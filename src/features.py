"""Turn frozen raw transactions into one interpretable feature row per address.

Everything here is computed from the cached Blockscout responses. Nothing is
imported from an external feature set: the labels came from MEW, but every number
a model ever sees is derived in this file.

Two rules shape the design.

Interpretability. Sanskriti has to defend every column in a live demo, so each
feature is a plain count, ratio, or summary statistic with an obvious meaning.
No embeddings, no unnamed principal components.

Leakage. Absolute block numbers are deliberately *not* model features. They are
kept in the frame because the temporal split needs them, but a model given
`first_block` could separate the classes by era rather than by behaviour. The
MODEL_FEATURES list below is the single source of truth for what a model may see,
and the app and training code both read it from here.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .blocktime import BlockClock

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

WEI_PER_ETH = 1e18
WEI_PER_GWEI = 1e9

# Columns a model is allowed to use. Kept separate from the bookkeeping columns
# (address, label, first_block, last_block) so that nothing era-specific or
# identifying can leak into training by accident.
BEHAVIOUR_FEATURES = [
    "n_tx",
    "n_sent",
    "n_received",
    "sent_ratio",
    "n_counterparties",
    "n_out_counterparties",
    "n_in_counterparties",
    "counterparty_ratio",
    "repeat_counterparty_ratio",
    "total_eth_sent",
    "total_eth_received",
    "mean_eth_sent",
    "max_eth_sent",
    "mean_eth_received",
    "max_eth_received",
    "net_eth_flow",
    "value_concentration",
    "zero_value_ratio",
    "mean_gas_price_gwei",
    "max_gas_price_gwei",
    "std_gas_price_gwei",
    "mean_gas_limit",
    "failed_ratio",
    "self_tx_ratio",
    "lifetime_days",
    "tx_per_active_day",
    "mean_days_between_tx",
    "min_days_between_tx",
    "burstiness",
]

GRAPH_FEATURES = [
    "degree",
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
    "community_size",
    "neighbour_risk_ratio",
    "n_labelled_neighbours",
]

MODEL_FEATURES = BEHAVIOUR_FEATURES + GRAPH_FEATURES

# Bookkeeping columns: needed for splitting and inspection, never fed to a model.
META_COLUMNS = ["address", "label", "first_block", "last_block", "status"]


def _to_eth(wei: str | int | None) -> float:
    """Convert a wei string to ETH, tolerating nulls and malformed values."""
    if wei is None:
        return 0.0
    try:
        return int(wei) / WEI_PER_ETH
    except (TypeError, ValueError):
        return 0.0


def _to_gwei(wei: str | int | None) -> float:
    if wei is None:
        return 0.0
    try:
        return int(wei) / WEI_PER_GWEI
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning 0 when the denominator is zero.

    Addresses with a single transaction, or with no outgoing transactions at all,
    are common in this dataset, so almost every ratio here needs this guard.
    """
    return numerator / denominator if denominator else 0.0


def iter_records(path: Path):
    """Stream the frozen cache one address at a time."""
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def address_features(record: dict, clock: BlockClock) -> dict:
    """Compute every behavioural feature for one address."""
    address = record["address"]
    transactions = record.get("transactions") or []

    row: dict[str, float | str | int] = {
        "address": address,
        "label": record.get("label", 0),
        "status": record.get("status", "ok"),
        "n_tx": len(transactions),
    }

    if not transactions:
        # Addresses whose fetch failed or which genuinely have no transactions.
        # They are kept, with zeroed features, so the loss stays visible in the
        # data rather than silently shrinking the sample.
        for column in BEHAVIOUR_FEATURES:
            row.setdefault(column, 0.0)
        row["first_block"] = 0
        row["last_block"] = 0
        return row

    sent_values: list[float] = []
    received_values: list[float] = []
    out_counterparties: set[str] = set()
    in_counterparties: set[str] = set()
    counterparty_hits: list[str] = []
    gas_prices: list[float] = []
    gas_limits: list[float] = []
    blocks: list[int] = []
    n_failed = 0
    n_self = 0
    n_zero_value = 0

    for tx in transactions:
        sender = (tx.get("fromAddressHash") or "").lower()
        receiver = (tx.get("toAddressHash") or "").lower()
        value = _to_eth(tx.get("value"))

        gas_prices.append(_to_gwei(tx.get("gasPrice")))
        # `gas` is the limit and is always populated. `gasUsed` comes back as 0
        # on many ordinary transfers, so it is not trustworthy as a feature.
        try:
            gas_limits.append(float(tx.get("gas") or 0))
        except (TypeError, ValueError):
            gas_limits.append(0.0)

        if tx.get("blockNumber") is not None:
            blocks.append(int(tx["blockNumber"]))
        if tx.get("status") != "OK":
            n_failed += 1
        if value == 0:
            n_zero_value += 1

        if sender == receiver:
            n_self += 1
            continue

        if sender == address:
            sent_values.append(value)
            if receiver:
                out_counterparties.add(receiver)
                counterparty_hits.append(receiver)
        elif receiver == address:
            received_values.append(value)
            if sender:
                in_counterparties.add(sender)
                counterparty_hits.append(sender)

    n_tx = len(transactions)
    counterparties = out_counterparties | in_counterparties

    row["n_sent"] = len(sent_values)
    row["n_received"] = len(received_values)
    row["sent_ratio"] = _safe_ratio(len(sent_values), n_tx)
    row["n_counterparties"] = len(counterparties)
    row["n_out_counterparties"] = len(out_counterparties)
    row["n_in_counterparties"] = len(in_counterparties)

    # A low counterparty ratio means the address deals with the same few peers
    # over and over; a ratio near 1 means almost every transaction is with a new
    # address, which is the shape a collection or distribution wallet has.
    row["counterparty_ratio"] = _safe_ratio(len(counterparties), n_tx)
    row["repeat_counterparty_ratio"] = _safe_ratio(
        len(counterparty_hits) - len(counterparties), max(len(counterparty_hits), 1)
    )

    total_sent = float(np.sum(sent_values)) if sent_values else 0.0
    total_received = float(np.sum(received_values)) if received_values else 0.0
    row["total_eth_sent"] = total_sent
    row["total_eth_received"] = total_received
    row["mean_eth_sent"] = float(np.mean(sent_values)) if sent_values else 0.0
    row["max_eth_sent"] = float(np.max(sent_values)) if sent_values else 0.0
    row["mean_eth_received"] = float(np.mean(received_values)) if received_values else 0.0
    row["max_eth_received"] = float(np.max(received_values)) if received_values else 0.0
    row["net_eth_flow"] = total_received - total_sent

    # How much of everything sent went out in the single largest transfer. A
    # value near 1 is the signature of a wallet that gathers funds then empties
    # itself in one sweep.
    row["value_concentration"] = _safe_ratio(row["max_eth_sent"], total_sent)
    row["zero_value_ratio"] = _safe_ratio(n_zero_value, n_tx)

    row["mean_gas_price_gwei"] = float(np.mean(gas_prices)) if gas_prices else 0.0
    row["max_gas_price_gwei"] = float(np.max(gas_prices)) if gas_prices else 0.0
    row["std_gas_price_gwei"] = float(np.std(gas_prices)) if len(gas_prices) > 1 else 0.0
    row["mean_gas_limit"] = float(np.mean(gas_limits)) if gas_limits else 0.0

    row["failed_ratio"] = _safe_ratio(n_failed, n_tx)
    row["self_tx_ratio"] = _safe_ratio(n_self, n_tx)

    first_block = min(blocks) if blocks else 0
    last_block = max(blocks) if blocks else 0
    row["first_block"] = first_block
    row["last_block"] = last_block

    lifetime_days = clock.days_between(first_block, last_block) if blocks else 0.0
    row["lifetime_days"] = lifetime_days
    row["tx_per_active_day"] = _safe_ratio(n_tx, max(lifetime_days, 1.0))

    # Gaps between consecutive transactions, in days. Transactions arrive
    # newest-first, so the block list is sorted before differencing.
    if len(blocks) > 1:
        ordered = sorted(blocks)
        gaps = [
            clock.days_between(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)
        ]
        mean_gap = float(np.mean(gaps))
        row["mean_days_between_tx"] = mean_gap
        row["min_days_between_tx"] = float(np.min(gaps))
        # Burstiness contrasts the spread of the gaps with their mean: steady
        # activity scores near 0, while a wallet that fires many transactions in
        # a few minutes and then goes quiet scores high.
        std_gap = float(np.std(gaps))
        row["burstiness"] = _safe_ratio(std_gap, mean_gap) if mean_gap > 0 else 0.0
    else:
        row["mean_days_between_tx"] = 0.0
        row["min_days_between_tx"] = 0.0
        row["burstiness"] = 0.0

    return row


def build_feature_frame(cache_path: Path, clock: BlockClock | None = None) -> pd.DataFrame:
    """Build the behavioural feature table for every cached address."""
    clock = clock or BlockClock()
    rows = [address_features(record, clock) for record in iter_records(cache_path)]
    frame = pd.DataFrame(rows)

    # Duplicate addresses can appear if a collection run was resumed; the later
    # record is the complete one.
    frame = frame.drop_duplicates(subset="address", keep="last").reset_index(drop=True)

    for column in BEHAVIOUR_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def extract_edges(cache_path: Path) -> pd.DataFrame:
    """Pull the address-to-address edge list out of the frozen transactions.

    Each transaction is one directed edge. These edges are what the graph in
    src/graph.py is built from -- the network is constructed by us from raw data,
    not downloaded ready-made.
    """
    records = []
    for record in iter_records(cache_path):
        for tx in record.get("transactions") or []:
            sender = (tx.get("fromAddressHash") or "").lower()
            receiver = (tx.get("toAddressHash") or "").lower()
            if not sender or not receiver or sender == receiver:
                continue
            records.append(
                {
                    "src": sender,
                    "dst": receiver,
                    "eth": _to_eth(tx.get("value")),
                    "block": int(tx.get("blockNumber") or 0),
                }
            )
    return pd.DataFrame(records).drop_duplicates()


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = RAW / "transactions.jsonl.gz"

    frame = build_feature_frame(cache)
    frame.to_csv(PROCESSED / "features_raw.csv", index=False)

    edges = extract_edges(cache)
    edges.to_csv(PROCESSED / "edges.csv", index=False)

    print(f"features: {frame.shape[0]} addresses x {len(BEHAVIOUR_FEATURES)} behaviour features")
    print(f"  illicit: {int(frame['label'].sum())}  licit: {int((frame['label'] == 0).sum())}")
    print(f"edges:    {len(edges)} unique directed address-to-address edges")
    print(f"  -> {PROCESSED}/features_raw.csv, {PROCESSED}/edges.csv")


if __name__ == "__main__":
    main()
