"""Tier 4: a graph neural network trained on the raw address graph.

Run after `python -m src.train` (it reuses the frozen feature frame and edge
list) with:

    python -m src.gnn

Tiers 1-3 answer the research question with hand-crafted graph features: we
decided that degree, PageRank and neighbour risk are what "network context"
means, computed them, and handed them to a Random Forest. A GNN removes that
decision. It is given the raw graph -- all 44k addresses, including the one-hop
neighbours we never labelled -- and has to *learn* what network context means by
passing messages along the edges. If the hand-crafted features missed a signal,
this is where it would show up.

The model is a two-layer GraphSAGE. Two layers means every prediction sees two
hops of neighbourhood, one hop further than any tier-3 feature reaches.

What each node is given
-----------------------
Labelled addresses get their behaviour features -- the same 29 columns tier 2
trains on. The ~41k unlabelled neighbours were never collected address-by-
address, so they have no behaviour row; they get zeros plus a `has_behaviour`
flag saying so, and every node gets its degrees, which are readable straight off
the graph. Heavy-tailed columns go through a signed log before standardising,
because a neural network, unlike a forest, cares about scale.

Deliberately absent: `neighbour_risk_ratio` and every other label-derived
feature. The GNN must rediscover that signal through message passing or not at
all -- feeding it our answer would make the comparison with tier 3 circular.

Leakage
-------
Same rules as everywhere else in this project. The split is the identical
temporal split (`src.train.temporal_split`, same ordering, same cutoff). The
scaler is fit on training rows only. Only training-mask nodes contribute to the
loss; test labels exist nowhere in the tensors the model can see. Early stopping
peeks at a validation slice carved off the *end* of the training window, never
at the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch_geometric.nn import SAGEConv

from .features import BEHAVIOUR_FEATURES
from .graph import build_graph
from .train import RANDOM_STATE, evaluate, temporal_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

HIDDEN = 64
DROPOUT = 0.3
LR = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
PATIENCE = 30
VAL_FRACTION = 0.15  # newest slice of the training window, for early stopping

# Degrees are the only features an unlabelled neighbour has.
STRUCTURAL_FEATURES = ["log_degree", "log_in_degree", "log_out_degree", "has_behaviour"]


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x)


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_tensors(frame: pd.DataFrame, edges: pd.DataFrame):
    """Turn the edge list and feature frame into one full-graph tensor set.

    Returns the node feature matrix, the (undirected) edge index, the label
    vector (-1 where unknown), and the address -> node-index mapping.
    """
    graph = build_graph(edges)
    # Isolated labelled addresses (collection failed, no transactions) still
    # need a node so they can be scored; they simply receive no messages.
    for address in frame["address"]:
        graph.add_node(address)

    index_of = {address: i for i, address in enumerate(graph.nodes())}
    n = len(index_of)

    # Message passing runs both ways: who you paid and who paid you are both
    # context, exactly as the undirected views in src/graph.py treat them.
    src = [index_of[u] for u, v in graph.edges()]
    dst = [index_of[v] for u, v in graph.edges()]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    x = np.zeros((n, len(BEHAVIOUR_FEATURES) + len(STRUCTURAL_FEATURES)), dtype=np.float32)
    for address, i in index_of.items():
        x[i, -4] = np.log1p(graph.in_degree(address) + graph.out_degree(address))
        x[i, -3] = np.log1p(graph.in_degree(address))
        x[i, -2] = np.log1p(graph.out_degree(address))

    behaviour = signed_log1p(frame[BEHAVIOUR_FEATURES].to_numpy(dtype=np.float32))
    rows = np.array([index_of[a] for a in frame["address"]])
    x[rows, : len(BEHAVIOUR_FEATURES)] = behaviour
    x[rows, -1] = 1.0  # has_behaviour

    y = np.full(n, -1, dtype=np.int64)
    y[rows] = frame["label"].to_numpy(dtype=np.int64)

    return x, edge_index, torch.tensor(y), index_of


def masks_from_split(frame: pd.DataFrame, index_of: dict[str, int], n: int):
    """Fit/validation/test masks from the same temporal split as tiers 1-3."""
    train, test = temporal_split(frame)
    # The validation slice is the newest part of the training window, so early
    # stopping rehearses exactly what the test split demands: score addresses
    # newer than anything the loss has seen.
    fit, val = temporal_split(train, test_fraction=VAL_FRACTION)

    masks = {}
    for name, part in {"fit": fit, "val": val, "test": test}.items():
        mask = torch.zeros(n, dtype=torch.bool)
        mask[[index_of[a] for a in part["address"]]] = True
        masks[name] = mask
    return masks, train, test


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    features_path = PROCESSED / "features_full.csv"
    if not features_path.exists():
        raise SystemExit(f"No feature frame at {features_path}. Run: python -m src.train")

    frame = pd.read_csv(features_path)
    edges = pd.read_csv(PROCESSED / "edges.csv")
    print(f"Feature frame: {len(frame)} labelled addresses; edge list: {len(edges)} rows")

    x, edge_index, y, index_of = build_tensors(frame, edges)
    print(f"Graph tensors: {x.shape[0]} nodes, {edge_index.shape[1]} directed messages")

    masks, train, test = masks_from_split(frame, index_of, x.shape[0])

    # Standardise on training rows only -- the same discipline as the sklearn
    # pipelines, where the scaler lives inside the fitted pipeline.
    train_rows = np.array([index_of[a] for a in train["address"]])
    mean = x[train_rows].mean(axis=0)
    std = x[train_rows].std(axis=0)
    std[std == 0] = 1.0
    x = torch.tensor((x - mean) / std)

    fit_labels = y[masks["fit"]]
    counts = torch.bincount(fit_labels, minlength=2).float()
    class_weight = counts.sum() / (2 * counts)  # sklearn's "balanced"

    model = GraphSAGE(x.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state, best_val_f1, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimiser.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[masks["fit"]], y[masks["fit"]], weight=class_weight)
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x, edge_index)[masks["val"]].argmax(dim=1)
        val_f1 = f1_score(y[masks["val"]].numpy(), val_pred.numpy(), zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
        if epoch % 25 == 0:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  val F1 {val_f1:.3f}")

    model.load_state_dict(best_state)
    print(f"Early stop: best validation F1 {best_val_f1:.3f} at epoch {best_epoch}")

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        prob = F.softmax(logits, dim=1)[:, 1]

    y_test = y[masks["test"]].numpy()
    y_pred = logits[masks["test"]].argmax(dim=1).numpy()
    y_score = prob[masks["test"]].numpy()

    config = {
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
    }
    metrics = evaluate("gnn / graphsage", y_test, y_pred, y_score)
    metrics["tier"] = "gnn"
    metrics["cv_f1"] = float("nan")
    metrics["best_params"] = json.dumps(config)

    print(f"\nTier 4 GNN: F1 {metrics['f1']:.3f}  PR-AUC {metrics['pr_auc']:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["licit", "illicit"]))
    print("Confusion matrix (rows true, cols predicted):")
    print(confusion_matrix(y_test, y_pred))

    # Append to the shared metrics table so the report and the app pick the
    # tier up without re-running the sklearn training.
    metrics_path = REPORTS / "metrics.csv"
    table = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["tier"] != "gnn"]
    table = pd.concat([table, pd.DataFrame([metrics])], ignore_index=True)
    table.to_csv(metrics_path, index=False)
    print(f"\nAppended to {metrics_path}")

    MODELS.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "behaviour_features": BEHAVIOUR_FEATURES,
            "structural_features": STRUCTURAL_FEATURES,
            "feature_mean": mean,
            "feature_std": std,
            "metrics": metrics,
        },
        MODELS / "gnn.pt",
    )
    print(f"Saved {MODELS / 'gnn.pt'}")


if __name__ == "__main__":
    main()
