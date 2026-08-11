"""Train and evaluate the three model tiers, then save the winning pipeline.

Run end to end with:

    python -m src.train

It reads the frozen transaction cache, builds features, evaluates three tiers, and
writes models/model.joblib plus the metrics the report and notebooks quote.

The three tiers
---------------
1. Majority class. Always predicts "licit". It is the number every other model has
   to beat, and it is here because with a 23% positive rate a useless model still
   scores 77% accuracy -- which is exactly why accuracy is not our metric.
2. Behaviour features only. What an address does on its own: counts, values, gas,
   timing.
3. Behaviour plus graph features. Adds who the address is connected to.

The gap between tier 2 and tier 3 is the project's research question, so the tiers
share a split, a scaler, and a random seed. Only the feature set changes.

Metric
------
We select on F1 for the illicit class, not accuracy. The classes are imbalanced
(23% positive) and the two errors are not equally costly: a missed scam address
lets fraud continue, while a false alarm costs an analyst a few minutes of review.
F1 balances catching scams (recall) against not drowning the analyst in false
positives (precision). PR-AUC is reported alongside it because it summarises that
trade-off across every threshold rather than just the default 0.5.

The split
---------
Addresses are split by when they first appeared on chain: the earliest 70% train,
the most recent 30% test. A random split would let a model train on one wallet in a
scam cluster and be tested on its sibling, which flatters the score. The temporal
split mimics the real task -- learn from known history, then score addresses you
have not seen before.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    BEHAVIOUR_FEATURES,
    GRAPH_FEATURES,
    build_feature_frame,
    extract_edges,
)
from .graph import build_graph, graph_features, graph_summary

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

RANDOM_STATE = 404
TEST_FRACTION = 0.30


def temporal_split(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split addresses by first appearance: oldest train, newest test.

    Addresses whose collection failed have `first_block == 0`. Sorting would dump
    them all into the training set, so they are ranked as if they were the newest
    -- they carry no usable features either way, and this keeps them from
    distorting the boundary.
    """
    ordered = frame.copy()
    ordered["_sort_block"] = ordered["first_block"].replace(0, ordered["first_block"].max() + 1)
    ordered = ordered.sort_values("_sort_block").reset_index(drop=True)

    cutoff = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:cutoff].drop(columns="_sort_block")
    test = ordered.iloc[cutoff:].drop(columns="_sort_block")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(name: str, y_true, y_pred, y_score=None) -> dict:
    """Score one model on the illicit (positive) class."""
    metrics = {
        "model": name,
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def make_models() -> dict[str, tuple[Pipeline, dict]]:
    """The two classifiers, each as a full preprocessing + model pipeline.

    The scaler is inside the pipeline, not applied beforehand. That is what lets
    us save one joblib file that takes raw feature values and returns a
    prediction, so the app never has to re-implement preprocessing.

    `class_weight="balanced"` tells both models that the illicit class is rarer
    and that missing it should hurt more, which matters far more here than any
    hyperparameter.
    """
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logreg_grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}

    # Trees do not need scaling, but the scaler is kept in the pipeline so both
    # tiers are saved and loaded through an identical interface.
    forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest_grid = {
        "model__max_depth": [6, 12, None],
        "model__min_samples_leaf": [1, 5],
    }

    return {"logreg": (logreg, logreg_grid), "random_forest": (forest, forest_grid)}


def run_tier(name: str, columns: list[str], train: pd.DataFrame, test: pd.DataFrame):
    """Fit both classifiers on one feature set and return their test metrics."""
    X_train, y_train = train[columns], train["label"].to_numpy()
    X_test, y_test = test[columns], test["label"].to_numpy()

    results, fitted = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for model_name, (pipeline, grid) in make_models().items():
        # Cross-validated tuning on the training split only. The test split is
        # touched exactly once, at scoring time.
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_score = best.predict_proba(X_test)[:, 1]

        metrics = evaluate(f"{name} / {model_name}", y_test, y_pred, y_score)
        metrics["cv_f1"] = float(search.best_score_)
        metrics["best_params"] = json.dumps(search.best_params_)
        metrics["tier"] = name
        results.append(metrics)
        fitted[model_name] = best

    return results, fitted


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    cache = RAW / "transactions.jsonl.gz"
    if not cache.exists():
        raise SystemExit(f"No transaction cache at {cache}. Run: python -m src.collect")

    print("Building behavioural features from the frozen cache...")
    frame = build_feature_frame(cache)
    frame.to_csv(PROCESSED / "features_raw.csv", index=False)
    print(f"  {len(frame)} addresses, {int(frame['label'].sum())} illicit")

    print("Building the address graph...")
    edges = extract_edges(cache)
    edges.to_csv(PROCESSED / "edges.csv", index=False)
    graph = build_graph(edges)
    summary = graph_summary(graph)
    print(f"  {summary['nodes']} nodes, {summary['edges']} edges")

    train, test = temporal_split(frame)
    print(
        f"Temporal split: {len(train)} train / {len(test)} test "
        f"(train positives {int(train['label'].sum())}, "
        f"test positives {int(test['label'].sum())})"
    )

    # The critical leakage control: only training labels are visible when the
    # neighbour risk ratio is computed, for train and test rows alike.
    known_labels = dict(zip(train["address"], train["label"]))
    graph_frame = graph_features(graph, frame["address"].tolist(), known_labels)

    frame = frame.merge(graph_frame, on="address", how="left").fillna(0.0)
    frame.to_csv(PROCESSED / "features_full.csv", index=False)

    # Re-split so both halves carry the graph columns, using the same ordering.
    train, test = temporal_split(frame)

    all_results = []

    # Tier 1: the trivial baseline the spec requires us to beat and to report.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[BEHAVIOUR_FEATURES], train["label"])
    y_pred = dummy.predict(test[BEHAVIOUR_FEATURES])
    baseline = evaluate("baseline / majority_class", test["label"].to_numpy(), y_pred)
    baseline["tier"] = "baseline"
    baseline["cv_f1"] = float("nan")
    baseline["best_params"] = "-"
    all_results.append(baseline)
    print(f"\nTier 1 baseline: accuracy {baseline['accuracy']:.3f}, F1 {baseline['f1']:.3f}")

    print("\nTier 2: behaviour features only...")
    tier2, fitted2 = run_tier("behaviour", BEHAVIOUR_FEATURES, train, test)
    all_results += tier2
    for row in tier2:
        print(f"  {row['model']}: F1 {row['f1']:.3f}  PR-AUC {row['pr_auc']:.3f}")

    print("\nTier 3: behaviour + graph features...")
    combined = BEHAVIOUR_FEATURES + GRAPH_FEATURES
    tier3, fitted3 = run_tier("behaviour+graph", combined, train, test)
    all_results += tier3
    for row in tier3:
        print(f"  {row['model']}: F1 {row['f1']:.3f}  PR-AUC {row['pr_auc']:.3f}")

    results = pd.DataFrame(all_results)
    results.to_csv(REPORTS / "metrics.csv", index=False)

    # Pick the best tier-3 model by F1: the app ships the full-feature model.
    best_row = max(tier3, key=lambda r: r["f1"])
    best_name = best_row["model"].split(" / ")[1]
    best_model = fitted3[best_name]

    # Saving the whole pipeline, not just the classifier, is what the spec asks
    # for: the scaler is part of the model and travels with it.
    artefact = {
        "pipeline": best_model,
        "features": combined,
        "behaviour_features": BEHAVIOUR_FEATURES,
        "graph_features": GRAPH_FEATURES,
        "model_name": best_name,
        "metrics": best_row,
        "train_positive_rate": float(train["label"].mean()),
        "graph_summary": summary,
    }
    joblib.dump(artefact, MODELS / "model.joblib")

    lift = best_row["f1"] - max(r["f1"] for r in tier2)
    print(f"\nSaved {MODELS / 'model.joblib'} ({best_name})")
    print(f"Graph-feature lift in F1 over behaviour-only: {lift:+.3f}")
    print(f"Metrics table: {REPORTS / 'metrics.csv'}")

    print("\nClassification report for the shipped model:")
    y_pred = best_model.predict(test[combined])
    print(classification_report(test["label"], y_pred, target_names=["licit", "illicit"]))
    print("Confusion matrix (rows true, cols predicted):")
    print(confusion_matrix(test["label"], y_pred))


if __name__ == "__main__":
    main()
