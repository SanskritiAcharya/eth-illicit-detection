"""Streamlit front end for the illicit-address classifier.

Run with:

    streamlit run app/app.py

This file deliberately contains no feature engineering and no preprocessing. It
imports both from `src`, which is the same code that produced the training data.
That is a rule the project spec states outright, and it is also the only way the
numbers on screen can be trusted to mean what the training run meant.

Presentation rules the layout follows, so that later edits keep to them:

- One accent colour, carried by the theme in `.streamlit/config.toml`. Colour on
  screen means risk level and nothing else.
- Icons come from the Material Symbols set Streamlit ships, never from emoji, so
  they inherit text colour and stay legible in both light and dark mode.
- Long explanations live behind expanders. The first screen shows the answer.
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
    page_icon=":material/network_node:",
    layout="centered",
)

METRICS_PATH = ROOT / "reports" / "metrics.csv"
PROFILES_PATH = ROOT / "data" / "processed" / "features_full.csv"
REPO_URL = "https://github.com/SanskritiAcharya/eth-illicit-detection"

# (floor, name, badge colour, icon). A forensics team works through a queue, so
# what they need from a model is a place in that queue, not a bare yes or no.
# The thresholds are deliberately coarse: the model is not precise enough to
# justify finer distinctions.
BANDS = (
    (0.75, "High", "red", ":material/priority_high:"),
    (0.45, "Medium", "orange", ":material/warning:"),
    (0.20, "Low", "yellow", ":material/visibility:"),
    (0.00, "Minimal", "green", ":material/check:"),
)

BAND_LEGEND = "Priority bands: minimal below 20% · low 20–45% · medium 45–75% · high 75% and above."

# Behaviour features grouped the way an analyst reads them, so the explorer is a
# form rather than twenty-nine identical boxes in a row.
FEATURE_GROUPS = {
    "Activity": ["n_tx", "n_sent", "n_received", "sent_ratio"],
    "Counterparties": [
        "n_counterparties",
        "n_out_counterparties",
        "n_in_counterparties",
        "counterparty_ratio",
        "repeat_counterparty_ratio",
    ],
    "Value moved": [
        "total_eth_sent",
        "total_eth_received",
        "mean_eth_sent",
        "max_eth_sent",
        "mean_eth_received",
        "max_eth_received",
        "net_eth_flow",
        "value_concentration",
        "zero_value_ratio",
    ],
    "Gas and execution": [
        "mean_gas_price_gwei",
        "max_gas_price_gwei",
        "std_gas_price_gwei",
        "mean_gas_limit",
        "failed_ratio",
        "self_tx_ratio",
    ],
    "Timing": [
        "lifetime_days",
        "tx_per_active_day",
        "mean_days_between_tx",
        "min_days_between_tx",
        "burstiness",
    ],
}

GROUP_ICONS = {
    "Activity": ":material/swap_horiz:",
    "Counterparties": ":material/group:",
    "Value moved": ":material/payments:",
    "Gas and execution": ":material/local_gas_station:",
    "Timing": ":material/schedule:",
}

# Counts are integers; ratios are bounded at 0-1. Stepping them all by 0.1 in
# three decimal places, as an earlier version did, made every field look the same
# and made the integer ones awkward to type.
COUNT_FEATURES = {
    "n_tx",
    "n_sent",
    "n_received",
    "n_counterparties",
    "n_out_counterparties",
    "n_in_counterparties",
}
RATIO_FEATURES = {
    "sent_ratio",
    "counterparty_ratio",
    "repeat_counterparty_ratio",
    "value_concentration",
    "zero_value_ratio",
    "failed_ratio",
    "self_tx_ratio",
}

EXAMPLES = {
    "Darklist phishing wallet": "0x09750ad360fdb7a2ee23669c4503c974d86d8694",
    "Ordinary 2017-era wallet": "0x7210a5388d8fdee1f628cbd92a55c0c5db775c51",
}

# Used only when the processed feature table is unavailable -- the Docker image
# excludes data/processed, so the explorer has to work without it.
FALLBACK_PROFILES = {
    "Ordinary wallet": {
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


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> pd.DataFrame | None:
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


@st.cache_data
def get_profiles() -> dict[str, dict[str, float]]:
    """Starting points for the explorer, taken from the labelled data itself.

    A median row is a more honest demonstration than an invented one: it is what
    the model actually saw. The hand-written sweep pattern stays alongside them
    because it is the archetype the project set out to detect, and no single
    median row shows it cleanly.
    """
    profiles: dict[str, dict[str, float]] = {}
    if PROFILES_PATH.exists():
        frame = pd.read_csv(PROFILES_PATH)
        for label, name in ((0, "Median labelled-licit address"), (1, "Median labelled-illicit address")):
            rows = frame[frame["label"] == label]
            if not rows.empty:
                profiles[name] = rows.median(numeric_only=True).to_dict()
    profiles.update(FALLBACK_PROFILES)
    return profiles


def band_for(probability: float) -> tuple[str, str, str]:
    """Turn a probability into an analyst-facing priority band."""
    for floor, name, colour, icon in BANDS:
        if probability >= floor:
            return name, colour, icon
    return BANDS[-1][1:]


def show_verdict(result: dict, *, note: str | None = None) -> None:
    """Render a prediction as a single card: answer, priority, confidence."""
    probability = float(result["probability_illicit"])
    name, colour, icon = band_for(probability)
    verdict = "Illicit" if result["prediction"] else "Licit"

    with st.container(border=True):
        answer, priority = st.columns([3, 2], vertical_alignment="center")
        with answer:
            st.caption("Verdict")
            st.markdown(f"## {verdict}")
        with priority:
            st.caption("Review priority")
            st.badge(name, icon=icon, color=colour)

        st.progress(
            min(max(probability, 0.0), 1.0),
            text=f"**{probability:.1%}** probability illicit",
        )
        st.caption(note or BAND_LEGEND)


def table_height(rows: int) -> int:
    """Height that shows every row, so a table never nests its own scrollbar."""
    return (rows + 1) * 35 + 3


def show_importances(bundle: dict, top_n: int = 15, this_address: dict | None = None) -> None:
    """Rank the features the trained model leans on, as bars rather than digits."""
    frame = feature_importances(bundle).head(top_n).copy()
    kind = "importance" if "importance" in frame.columns else "coefficient"
    graph_features = set(bundle["graph_features"])
    frame["source"] = ["graph" if f in graph_features else "behaviour" for f in frame["feature"]]

    columns = {
        "feature": st.column_config.TextColumn("Feature"),
        "source": st.column_config.TextColumn("Derived from", width="small"),
        kind: st.column_config.ProgressColumn(
            kind.capitalize(),
            format="%.3f",
            min_value=float(min(0.0, frame[kind].min())),
            max_value=float(frame[kind].max()),
        ),
    }
    order = ["feature", "source", kind]

    if this_address:
        frame["value"] = [float(this_address.get(f, 0.0)) for f in frame["feature"]]
        columns["value"] = st.column_config.NumberColumn("This address", format="%.3f")
        order.append("value")

    st.dataframe(
        frame[order],
        width="stretch",
        hide_index=True,
        column_config=columns,
        height=table_height(len(frame)),
    )
    st.caption(
        "Importance is how much the trained model relies on each feature overall, "
        "not how much it drove any one prediction."
    )


def number_input_for(name: str, default: float, key: str, container) -> float:
    """One tuned input box, so counts, ratios and amounts each behave sensibly."""
    if name in COUNT_FEATURES:
        return container.number_input(name, value=float(default), step=1.0, format="%.0f", key=key)
    if name in RATIO_FEATURES:
        return container.number_input(
            name, value=float(min(max(default, 0.0), 1.0)),
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=key,
        )
    if name == "mean_gas_limit":
        return container.number_input(name, value=float(default), step=1000.0, format="%.0f", key=key)
    return container.number_input(name, value=float(default), step=0.1, format="%.3f", key=key)


def use_example() -> None:
    """Copy the chosen example into the address box before the form re-renders."""
    chosen = st.session_state.get("example_choice")
    if chosen:
        st.session_state["address_input"] = EXAMPLES[chosen]


# ---------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.markdown("#### Ethereum Illicit Address Detector")
    st.caption("DCS 404 — Machine Learning and Artificial Intelligence")
    st.caption("Sanskriti Acharya · Sweta Sharma")

    st.divider()

    try:
        bundle = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc), icon=":material/error:")
        st.error("No trained model found. Run `python -m src.train` first.", icon=":material/error:")
        st.stop()

    st.caption("Loaded model")
    st.markdown(f"**{bundle['model_name'].replace('_', ' ').title()}** · behaviour + graph")

    model_metrics = bundle.get("metrics", {})
    if model_metrics:
        f1_column, pr_column = st.columns(2)
        f1_column.metric("Test F1", f"{model_metrics.get('f1', float('nan')):.3f}")
        pr_column.metric("PR-AUC", f"{model_metrics.get('pr_auc', float('nan')):.3f}")

    st.divider()

    with st.expander("Where the data comes from", icon=":material/database:"):
        st.markdown(
            "Labels are the MyEtherWallet community darklist. Every **feature** is "
            "derived by us from raw transaction data fetched through the Blockscout "
            "API, and the address graph is built by us in NetworkX."
        )

    st.link_button("View source", REPO_URL, icon=":material/code:", width="stretch")

# ------------------------------------------------------------------- main ----

st.title("Ethereum Illicit Address Detector")
st.markdown(
    "Score an Ethereum address for signs of phishing or scam activity, from how it "
    "transacts and where it sits in the transaction graph."
)

lookup_tab, explore_tab, model_tab = st.tabs(
    [
        ":material/search: Address lookup",
        ":material/tune: Feature explorer",
        ":material/analytics: Model performance",
    ]
)

with lookup_tab:
    st.subheader("Score a live address")
    st.caption(
        "Transactions are fetched live from Blockscout, the training-time feature "
        "code runs on them, and the saved pipeline scores the result."
    )

    st.pills(
        "Try an example",
        list(EXAMPLES),
        key="example_choice",
        on_change=use_example,
        selection_mode="single",
    )

    with st.form("lookup", border=False):
        address = st.text_input(
            "Ethereum address",
            key="address_input",
            placeholder="0x…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Analyse address", type="primary", icon=":material/search:"
        )

    if submitted:
        cleaned = address.strip()
        if not cleaned:
            st.warning("Enter an address first.", icon=":material/edit:")
        elif not (cleaned.startswith("0x") and len(cleaned) == 42):
            st.error(
                "That does not look like an Ethereum address — expected `0x` followed "
                "by 40 hex characters.",
                icon=":material/error:",
            )
        else:
            with st.spinner("Fetching transactions from Blockscout…"):
                try:
                    result = predict_from_address(cleaned)
                except Exception as exc:
                    st.error(f"Could not fetch this address: {exc}", icon=":material/cloud_off:")
                    result = None

            if result is not None:
                count = result["n_transactions"]
                if count == 0:
                    st.warning(
                        "Blockscout returned no transactions for this address, so every "
                        "feature is zero and the score below carries no real evidence.",
                        icon=":material/warning:",
                    )

                show_verdict(
                    result,
                    note=(
                        f"Scored from {count:,} transactions. Graph features are held at "
                        "neutral values — a single lookup cannot rebuild the training "
                        "graph — so this rests on the address's own behaviour. "
                        + BAND_LEGEND
                    ),
                )

                with st.expander("Computed features", icon=":material/table_rows:"):
                    computed = result["features"]
                    frame = pd.DataFrame(
                        [
                            {"feature": name, "value": float(computed[name])}
                            for name in bundle["features"]
                            if isinstance(computed.get(name), (int, float))
                        ]
                    )
                    st.dataframe(
                        frame,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "feature": st.column_config.TextColumn("Feature"),
                            "value": st.column_config.NumberColumn("Value", format="%.3f"),
                        },
                    )

                with st.expander("What the model relies on", icon=":material/bar_chart:"):
                    show_importances(bundle, this_address=result.get("features"))

with explore_tab:
    st.subheader("Explore the model's response")
    st.caption(
        "Set feature values directly to see how the decision moves, without waiting "
        "on the network. Graph features are held at their neutral values throughout."
    )

    profiles = get_profiles()
    profile_name = st.selectbox("Start from", list(profiles), key="profile_choice")
    profile = profiles[profile_name]

    behaviour = bundle["behaviour_features"]
    grouped = {group: [f for f in names if f in behaviour] for group, names in FEATURE_GROUPS.items()}
    ungrouped = [f for f in behaviour if not any(f in names for names in grouped.values())]
    if ungrouped:
        grouped["Other"] = ungrouped

    values: dict[str, float] = {}
    for index, (group, names) in enumerate(grouped.items()):
        with st.expander(group, icon=GROUP_ICONS.get(group, ":material/list:"), expanded=index == 0):
            columns = st.columns(2)
            for position, name in enumerate(names):
                values[name] = number_input_for(
                    name,
                    float(profile.get(name, 0.0)),
                    # The preset name is part of the key so that switching preset
                    # actually reloads the boxes: Streamlit ignores `value` once a
                    # key already exists in session state.
                    key=f"manual::{profile_name}::{name}",
                    container=columns[position % 2],
                )

    values.update(neutral_graph_features(bundle))

    if st.button("Score these values", type="primary", icon=":material/calculate:"):
        show_verdict(
            score(values, bundle),
            note="Graph features held neutral, so this reflects behaviour alone. " + BAND_LEGEND,
        )

with model_tab:
    st.subheader("How well does it work?")

    metrics_frame = get_metrics()
    if metrics_frame is None:
        st.info(
            "No metrics file yet. Run `python -m src.train` to generate it.",
            icon=":material/info:",
        )
    else:
        st.caption(
            "Four tiers trained on the same temporal split. The first is the trivial "
            "baseline every other model has to beat; the last is a graph neural "
            "network (GraphSAGE) that learns network context from the raw address "
            "graph instead of hand-crafted graph features."
        )
        scores = metrics_frame[["model", "accuracy", "precision", "recall", "f1", "pr_auc"]].rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "pr_auc": "PR-AUC",
            }
        )
        # The baseline has no PR-AUC; a dash says so more plainly than "None".
        st.dataframe(
            scores.style.format("{:.3f}", subset=scores.columns[1:], na_rep="—"),
            width="stretch",
            hide_index=True,
            height=table_height(len(scores)),
        )

        best_behaviour = metrics_frame[metrics_frame["tier"] == "behaviour"]["f1"].max()
        best_combined = metrics_frame[metrics_frame["tier"] == "behaviour+graph"]["f1"].max()
        if pd.notna(best_behaviour) and pd.notna(best_combined):
            headline, blank = st.columns([2, 3], vertical_alignment="center")
            headline.metric(
                "Best F1, with graph features",
                f"{best_combined:.3f}",
                delta=f"{best_combined - best_behaviour:+.3f} vs behaviour alone",
                border=True,
            )
            blank.caption(
                "This difference is the project's research question: does an address's "
                "network of connections reveal fraud that its own behaviour does not?"
            )

        with st.expander("Why F1 and not accuracy", icon=":material/help:"):
            baseline = metrics_frame[metrics_frame["tier"] == "baseline"]["accuracy"].max()
            st.markdown(
                f"Illicit addresses are the minority class — {bundle['train_positive_rate']:.0%} "
                "of the training window. A model that always answers *licit* therefore "
                f"scores {baseline:.1%} accuracy on the test split while catching nothing. "
                "F1 on the illicit class rewards catching scams without burying an "
                "analyst in false alarms."
            )

    st.divider()

    st.subheader("Feature importances")
    show_importances(bundle, top_n=20)

    graph_info = bundle.get("graph_summary", {})
    if graph_info:
        st.divider()
        st.subheader("The transaction graph")
        nodes, edges, degree = st.columns(3)
        nodes.metric("Addresses", f"{graph_info.get('nodes', 0):,}", border=True)
        edges.metric("Transfers", f"{graph_info.get('edges', 0):,}", border=True)
        degree.metric("Mean degree", f"{graph_info.get('mean_degree', 0):.1f}", border=True)

st.divider()
st.caption(
    "Predictions are a triage aid, not evidence of wrongdoing. Labels are community "
    "reports and are incomplete, so an address scored *licit* here has simply not "
    "been reported — it has not been cleared."
)
