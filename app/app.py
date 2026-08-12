"""Streamlit front end for the illicit-address classifier.

Run with:

    streamlit run app/app.py

This file deliberately contains no feature engineering and no preprocessing. It
imports both from `src`, which is the same code that produced the training data.
That is a rule the project spec states outright, and it is also the only way the
numbers on screen can be trusted to mean what the training run meant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The app lives in app/ but imports from src/, so the project root has to be
# importable. This is the only path manipulation in the project.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (  # noqa: E402
    feature_importances,
    load_model,
    neutral_graph_features,
    predict_from_address,
    score,
)

st.set_page_config(
    page_title="Ethereum Illicit Address Detector",
    page_icon="🔍",
    layout="wide",
)

METRICS_PATH = ROOT / "reports" / "metrics.csv"


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> pd.DataFrame | None:
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


def risk_band(probability: float) -> tuple[str, str]:
    """Turn a probability into an analyst-facing priority band.

    A forensics team works through a queue, so what they need from a model is a
    place in that queue, not a bare yes or no. The thresholds are deliberately
    coarse: the model is not precise enough to justify finer distinctions.
    """
    if probability >= 0.75:
        return "HIGH", "🔴"
    if probability >= 0.45:
        return "MEDIUM", "🟠"
    if probability >= 0.20:
        return "LOW", "🟡"
    return "MINIMAL", "🟢"


def show_verdict(result: dict) -> None:
    """Render a prediction with its probability and priority band."""
    probability = result["probability_illicit"]
    band, icon = risk_band(probability)

    left, middle, right = st.columns(3)
    left.metric("Verdict", result["label"])
    middle.metric("Probability illicit", f"{probability:.1%}")
    right.metric("Review priority", f"{icon} {band}")

    st.progress(min(max(probability, 0.0), 1.0))


def explain(result: dict, bundle: dict, top_n: int = 12) -> None:
    """Show which features the model leans on, and this address's values."""
    importances = feature_importances(bundle).head(top_n)
    features = result.get("features", {})

    if features:
        importances = importances.assign(
            **{"this address": [features.get(name, 0.0) for name in importances["feature"]]}
        )

    st.dataframe(importances, width="stretch", hide_index=True)
    st.caption(
        "Importance is how much the trained model relies on each feature overall, "
        "not how much it drove this particular prediction."
    )


# ---------------------------------------------------------------- sidebar ----

st.sidebar.title("About")
st.sidebar.markdown(
    """
**Detecting Illicit Ethereum Transactions with Graph-Derived Features**

DCS 404 — Machine Learning and Artificial Intelligence
Sanskriti Acharya and Sweta Sharma

Labels come from the MyEtherWallet community darklist. Every **feature** is
derived by us from raw transaction data fetched through the Blockscout GraphQL
API, and the address graph is built by us in NetworkX.
"""
)

try:
    bundle = get_model()
    st.sidebar.success(f"Model loaded: {bundle['model_name']}")
    metrics = bundle.get("metrics", {})
    if metrics:
        st.sidebar.caption(
            f"Test F1 {metrics.get('f1', float('nan')):.3f} · "
            f"PR-AUC {metrics.get('pr_auc', float('nan')):.3f}"
        )
except FileNotFoundError as exc:
    st.sidebar.error(str(exc))
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

# ------------------------------------------------------------------- main ----

st.title("🔍 Ethereum Illicit Address Detector")
st.markdown(
    "Score an Ethereum address for signs of phishing or scam activity. "
    "The model was trained on behaviour derived from raw on-chain transactions "
    "plus features describing the address's position in the transaction graph."
)

lookup_tab, manual_tab, model_tab = st.tabs(
    ["Look up an address", "Explore features by hand", "Model performance"]
)

with lookup_tab:
    st.subheader("Score a live address")
    st.markdown(
        "Enter any Ethereum address. Its transactions are fetched live from "
        "Blockscout, the same features used in training are computed from them, "
        "and the saved pipeline scores the result."
    )

    examples = {
        "— choose an example —": "",
        "Known phishing wallet (darklist)": "0x09750ad360fdb7a2ee23669c4503c974d86d8694",
        "Ordinary 2017-era wallet": "0x7210a5388d8fdee1f628cbd92a55c0c5db775c51",
    }
    chosen = st.selectbox("Examples", list(examples), index=0)
    address = st.text_input(
        "Ethereum address",
        value=examples[chosen],
        placeholder="0x...",
    )

    if st.button("Analyse address", type="primary", disabled=not address.strip()):
        cleaned = address.strip()
        if not (cleaned.startswith("0x") and len(cleaned) == 42):
            st.error("That does not look like an Ethereum address (expected 0x + 40 hex characters).")
        else:
            with st.spinner("Fetching transactions from Blockscout..."):
                try:
                    result = predict_from_address(cleaned)
                except Exception as exc:
                    st.error(f"Could not fetch this address: {exc}")
                    result = None

            if result is not None:
                if result["n_transactions"] == 0:
                    st.warning(
                        "Blockscout returned no transactions for this address, so every "
                        "feature is zero and the prediction below carries no real evidence."
                    )
                show_verdict(result)

                st.info(
                    f"Scored from **{result['n_transactions']} transactions**. "
                    "Graph features are set to their neutral values here: they need the "
                    "full training graph, which a single live lookup cannot rebuild. "
                    "This prediction therefore rests on the address's own behaviour."
                )

                with st.expander("Computed features"):
                    frame = pd.DataFrame(
                        [
                            {"feature": k, "value": v}
                            for k, v in result["features"].items()
                            if isinstance(v, (int, float))
                        ]
                    )
                    st.dataframe(frame, width="stretch", hide_index=True)

                with st.expander("What the model pays attention to"):
                    explain(result, bundle)

with manual_tab:
    st.subheader("Explore the model's response")
    st.markdown(
        "Set feature values directly to see how the model behaves. This is useful "
        "for understanding the decision boundary without waiting on the network."
    )

    presets = {
        "Typical ordinary wallet": {
            "n_tx": 12, "n_sent": 6, "n_received": 6, "n_counterparties": 8,
            "total_eth_sent": 3.0, "total_eth_received": 3.2, "lifetime_days": 220.0,
            "counterparty_ratio": 0.67, "value_concentration": 0.35,
            "mean_gas_price_gwei": 22.0, "burstiness": 0.9, "tx_per_active_day": 0.05,
        },
        "Collect-and-sweep pattern": {
            "n_tx": 40, "n_sent": 3, "n_received": 37, "n_counterparties": 36,
            "total_eth_sent": 51.0, "total_eth_received": 52.0, "lifetime_days": 9.0,
            "counterparty_ratio": 0.9, "value_concentration": 0.95,
            "mean_gas_price_gwei": 60.0, "burstiness": 2.4, "tx_per_active_day": 4.4,
        },
    }

    preset_name = st.selectbox("Start from a preset", list(presets))
    preset = presets[preset_name]

    behaviour = bundle["behaviour_features"]
    values: dict[str, float] = {}

    columns = st.columns(3)
    for index, name in enumerate(behaviour):
        default = float(preset.get(name, 0.0))
        values[name] = columns[index % 3].number_input(
            name, value=default, step=0.1, format="%.3f", key=f"manual_{name}"
        )

    values.update(neutral_graph_features(bundle))

    if st.button("Score these values", type="primary"):
        show_verdict(score(values, bundle))
        st.caption(
            "Graph features are held at their neutral values, so this reflects "
            "behaviour alone."
        )

with model_tab:
    st.subheader("How well does it work?")

    metrics_frame = get_metrics()
    if metrics_frame is None:
        st.info("No metrics file yet. Run `python -m src.train` to generate it.")
    else:
        st.markdown(
            "Three tiers were trained on the same temporal split. The first is the "
            "trivial baseline every other model has to beat."
        )
        display = metrics_frame[
            ["model", "accuracy", "precision", "recall", "f1", "pr_auc"]
        ].round(3)
        st.dataframe(display, width="stretch", hide_index=True)

        st.markdown(
            """
**Why F1 and not accuracy.** About 23% of the addresses are illicit, so a model
that always says "licit" is right 77% of the time while being useless. F1 on the
illicit class rewards catching scams without burying an analyst in false alarms.
            """
        )

        best_behaviour = metrics_frame[metrics_frame["tier"] == "behaviour"]["f1"].max()
        best_combined = metrics_frame[metrics_frame["tier"] == "behaviour+graph"]["f1"].max()
        if pd.notna(best_behaviour) and pd.notna(best_combined):
            st.metric(
                "F1 gained by adding graph features",
                f"{best_combined:.3f}",
                delta=f"{best_combined - best_behaviour:+.3f}",
            )
            st.caption(
                "This difference is the project's research question: does an "
                "address's network of connections reveal fraud that its own "
                "behaviour does not?"
            )

    st.divider()
    st.subheader("Feature importances")
    st.dataframe(feature_importances(bundle).head(20), width="stretch", hide_index=True)

    graph_info = bundle.get("graph_summary", {})
    if graph_info:
        st.subheader("The transaction graph")
        left, middle, right = st.columns(3)
        left.metric("Addresses (nodes)", f"{graph_info.get('nodes', 0):,}")
        middle.metric("Transfers (edges)", f"{graph_info.get('edges', 0):,}")
        right.metric("Mean degree", f"{graph_info.get('mean_degree', 0):.1f}")

st.divider()
st.caption(
    "Predictions are a triage aid, not evidence of wrongdoing. Labels are "
    "community reports and are incomplete, so an address scored LICIT here has "
    "simply not been reported — it has not been cleared."
)
