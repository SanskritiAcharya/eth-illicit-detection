"""Test whether the headline result survives the sampling bias we introduced.

Why this file exists
--------------------
The licit class is sampled by reading the transactions of random blocks and
keeping the addresses that appear in them. That is a *length-biased* sample: an
address that transacts often appears in more blocks, so it is more likely to be
picked. The illicit class has no such bias -- it is whatever the community
reported.

The consequence shows up plainly in the data. 61% of licit addresses hit the
100-transaction collection cap, against 8% of illicit ones, and `n_tx` on its own
scores an F1 of 0.836 against the full model's 0.890. A large part of what the
model has learned is "busy addresses are licit", which is a fact about how we
built the sample rather than a fact about fraud.

Some of that gap is real -- phishing wallets genuinely are short-lived, which is
a well-documented property -- but the sampling frame inflates it, and reporting
0.890 without saying so would be misleading.

What this does
--------------
Case-control matching. Each illicit address is paired with a licit address that
has a similar transaction count, and the model is retrained and re-evaluated on
those matched pairs alone. Within a matched pair, activity level carries no
information by construction, so whatever accuracy survives comes from *how* the
addresses behave rather than *how much*.

The number this prints is the conservative estimate of the model's real skill,
and it is reported alongside the headline in the report rather than instead of
it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BEHAVIOUR_FEATURES, GRAPH_FEATURES
from .train import RANDOM_STATE, run_tier, temporal_split

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Features that describe how *much* an address does, rather than how it behaves.
# These are the ones the length-biased sample contaminates.
ACTIVITY_FEATURES = [
    "n_tx", "n_sent", "n_received", "lifetime_days", "tx_per_active_day",
    "mean_days_between_tx", "min_days_between_tx", "burstiness",
    "n_counterparties", "n_out_counterparties", "n_in_counterparties",
    "degree", "in_degree", "out_degree", "pagerank", "community_size",
]


def match_on_activity(frame: pd.DataFrame, tolerance: float = 0.25) -> pd.DataFrame:
    """Pair each illicit address with a licit address of similar transaction count.

    Matching is greedy and without replacement: illicit addresses are taken in a
    fixed order and each claims the closest unused licit address whose `n_tx` is
    within `tolerance` (as a relative difference). Addresses with no acceptable
    partner are dropped, which is the point -- it removes the region of the
    activity distribution where only one class exists.
    """
    illicit = frame[frame["label"] == 1].sort_values("n_tx").reset_index(drop=True)
    licit = frame[frame["label"] == 0].sort_values("n_tx").reset_index(drop=True)

    licit_counts = licit["n_tx"].to_numpy(dtype=float)
    claimed = np.zeros(len(licit), dtype=bool)

    keep_illicit, keep_licit = [], []
    for _, row in illicit.iterrows():
        target = float(row["n_tx"])
        allowed = max(target * tolerance, 1.0)

        distances = np.abs(licit_counts - target)
        distances[claimed] = np.inf

        best = int(np.argmin(distances))
        if distances[best] > allowed:
            continue

        claimed[best] = True
        keep_illicit.append(row)
        keep_licit.append(licit.iloc[best])

    if not keep_illicit:
        return frame.iloc[0:0]

    matched = pd.concat(
        [pd.DataFrame(keep_illicit), pd.DataFrame(keep_licit)], ignore_index=True
    )
    return matched.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    frame = pd.read_csv(PROCESSED / "features_full.csv")
    combined = BEHAVIOUR_FEATURES + GRAPH_FEATURES
    results = {}

    print("=" * 68)
    print("SENSITIVITY ANALYSIS -- does the result survive the sampling bias?")
    print("=" * 68)

    at_cap = frame.groupby("label")["n_tx"].apply(lambda s: (s >= 100).mean())
    print("\nAddresses at the 100-transaction collection cap:")
    print(f"  licit   {at_cap.get(0, 0):.1%}")
    print(f"  illicit {at_cap.get(1, 0):.1%}")
    print("  -> the licit sample is biased towards busy addresses, because it was")
    print("     drawn from block transactions and busy addresses appear in more blocks")

    train, test = temporal_split(frame)

    print("\n[1] n_tx alone -- how much does raw activity explain?")
    single, _ = run_tier("n_tx only", ["n_tx"], train, test)
    for row in single:
        print(f"    {row['model']:<34} F1 {row['f1']:.3f}  PR-AUC {row['pr_auc']:.3f}")
    results["n_tx_only"] = max(r["f1"] for r in single)

    print("\n[2] every activity feature removed")
    lean = [c for c in combined if c not in ACTIVITY_FEATURES]
    print(f"    {len(lean)} of {len(combined)} features kept")
    ablated, _ = run_tier("no activity", lean, train, test)
    for row in ablated:
        print(f"    {row['model']:<34} F1 {row['f1']:.3f}  PR-AUC {row['pr_auc']:.3f}")
    results["no_activity"] = max(r["f1"] for r in ablated)

    print("\n[3] activity-matched pairs -- the conservative estimate")
    matched = match_on_activity(frame)
    if matched.empty or matched["label"].nunique() < 2:
        print("    not enough matched pairs to evaluate")
        results["matched"] = float("nan")
    else:
        counts = matched["label"].value_counts()
        print(f"    {len(matched)} addresses ({counts.get(1, 0)} illicit, "
              f"{counts.get(0, 0)} licit), matched on transaction count")
        median = matched.groupby("label")["n_tx"].median()
        print(f"    median n_tx after matching: licit {median.get(0, 0):.0f}, "
              f"illicit {median.get(1, 0):.0f}")

        m_train, m_test = temporal_split(matched)
        if m_test["label"].nunique() < 2:
            print("    matched test split has one class only; cannot evaluate")
            results["matched"] = float("nan")
        else:
            matched_results, _ = run_tier("matched", combined, m_train, m_test)
            for row in matched_results:
                print(f"    {row['model']:<34} F1 {row['f1']:.3f}  "
                      f"PR-AUC {row['pr_auc']:.3f}  recall {row['recall']:.3f}")
            results["matched"] = max(r["f1"] for r in matched_results)
            matched.to_csv(PROCESSED / "features_matched.csv", index=False)

    print("\n" + "=" * 68)
    print("Reading these together: the headline F1 is inflated by a licit sample")
    print("that skews busy. The matched figure is the honest estimate of how much")
    print("of the signal is behavioural rather than an artefact of sampling.")
    print("=" * 68)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "sensitivity.json").write_text(json.dumps(results, indent=1))
    print(f"\nWrote {REPORTS / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
