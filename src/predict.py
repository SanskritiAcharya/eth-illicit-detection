"""Score an address with the trained pipeline.

This is the module the Streamlit app imports. It exists so that the app contains
no preprocessing logic of its own: the feature code here is the same code that
built the training set, imported from src.features, not copied.

Two ways in:

- `predict_from_address` takes a real Ethereum address, fetches its transactions
  live from Blockscout, builds features, and scores it.
- `predict_from_features` takes a dictionary of feature values, for exploring how
  the model responds without touching the network.

A live lookup can only see the address itself and its immediate transactions, so
the graph features that need the full training graph are not available. They are
filled with the neutral values described in `neutral_graph_features` and the
caller is told which ones were estimated, because a probability computed from
partial evidence should be presented as exactly that.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .blocktime import BlockClock
from .blockscout import fetch_transactions
from .features import address_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH) -> dict:
    """Load the saved pipeline bundle.

    The artefact holds the fitted pipeline *and* the feature list it was trained
    on, so the app can never disagree with the model about column order -- a
    mismatch there would silently produce nonsense predictions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one first with: python -m src.train"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def _clock() -> BlockClock:
    return BlockClock()


def neutral_graph_features(bundle: dict) -> dict:
    """Values to use for graph features when the training graph is unavailable.

    Zero is the honest default here. `neighbour_risk_ratio` of 0 with
    `n_labelled_neighbours` of 0 says "no confirmed-illicit neighbours are known",
    which is exactly the situation for an address we have just looked up, and it
    is the same encoding used during training for addresses with no labelled
    neighbours. So the model has seen this pattern before and is not being asked
    to extrapolate.
    """
    return {name: 0.0 for name in bundle["graph_features"]}


def build_row(features: dict, bundle: dict) -> pd.DataFrame:
    """Assemble a one-row frame in exactly the column order the model expects."""
    row = {name: float(features.get(name, 0.0)) for name in bundle["features"]}
    return pd.DataFrame([row], columns=bundle["features"])


def score(features: dict, bundle: dict | None = None) -> dict:
    """Return the model's verdict for one set of feature values."""
    bundle = bundle or load_model()
    frame = build_row(features, bundle)

    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    prediction = int(bundle["pipeline"].predict(frame)[0])

    return {
        "prediction": prediction,
        "label": "ILLICIT" if prediction else "LICIT",
        "probability_illicit": probability,
        "model": bundle["model_name"],
    }


def predict_from_features(features: dict) -> dict:
    """Score a hand-entered or slider-driven set of feature values."""
    return score(features)


def predict_from_address(address: str, max_pages: int = 8) -> dict:
    """Fetch an address live, engineer its features, and score it.

    The same `address_features` function that built the training data is used
    here, so what the model sees at prediction time is computed by identical code.
    """
    bundle = load_model()
    address = address.strip().lower()

    transactions = fetch_transactions(address, max_pages=max_pages)
    record = {"address": address, "label": 0, "transactions": transactions}
    features = address_features(record, _clock())

    # Graph features cannot be computed from a live lookup alone; they need the
    # graph built during training. They are filled neutrally and flagged.
    features.update(neutral_graph_features(bundle))

    result = score(features, bundle)
    result.update(
        {
            "address": address,
            "n_transactions": len(transactions),
            "features": features,
            "graph_features_estimated": True,
            "behaviour_only": True,
        }
    )
    return result


def feature_importances(bundle: dict | None = None) -> pd.DataFrame:
    """Rank features by how much the trained model relies on them.

    Random forests expose impurity-based importances; logistic regression exposes
    coefficients, whose sign also says which direction pushes towards illicit.
    Both are returned as a sorted frame so the app can show why a prediction
    came out the way it did.
    """
    bundle = bundle or load_model()
    model = bundle["pipeline"].named_steps["model"]
    names = bundle["features"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "importance"
    else:
        values = model.coef_[0]
        kind = "coefficient"

    frame = pd.DataFrame({"feature": names, kind: values})
    frame["magnitude"] = frame[kind].abs()
    return frame.sort_values("magnitude", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "0x09750ad360fdb7a2ee23669c4503c974d86d8694"
    outcome = predict_from_address(target)
    print(f"{outcome['address']}: {outcome['label']} "
          f"(p={outcome['probability_illicit']:.3f}, {outcome['n_transactions']} txs)")
