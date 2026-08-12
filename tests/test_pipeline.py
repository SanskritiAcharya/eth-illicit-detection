"""Unit tests for the parts where a silent bug would not show up in the output.

A wrong feature value does not raise an exception; it just quietly shifts a
metric, and the run still looks fine. These tests pin down the arithmetic on
transactions we built by hand, so the expected answers can be checked by reading
them rather than by trusting the code that produced them.

Run with:

    python -m pytest tests/ -v
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.blockscout import PAGE_SIZE, TX_FIELDS, _cursor  # noqa: E402
from src.blocktime import BlockClock  # noqa: E402
from src.features import (  # noqa: E402
    BEHAVIOUR_FEATURES,
    GRAPH_FEATURES,
    address_features,
    _safe_ratio,
    _to_eth,
    _to_gwei,
)
from src.graph import build_graph, graph_features  # noqa: E402
from src.labels import dedupe_addresses  # noqa: E402

ADDRESS = "0xaaaa000000000000000000000000000000000001"
PEER_ONE = "0xbbbb000000000000000000000000000000000002"
PEER_TWO = "0xcccc000000000000000000000000000000000003"

ONE_ETH = str(10**18)


def tx(sender, receiver, value=ONE_ETH, block=4_000_000, gas="21000",
       gas_price="20000000000", status="OK"):
    """Build one transaction record in the shape the collector produces."""
    return {
        "hash": f"0x{block:064x}",
        "fromAddressHash": sender,
        "toAddressHash": receiver,
        "value": value,
        "gas": gas,
        "gasPrice": gas_price,
        "blockNumber": block,
        "status": status,
    }


@pytest.fixture(scope="module")
def clock():
    """A clock built from two anchors, so tests never touch the network."""
    return BlockClock({3_000_000: 1_484_500_000.0, 7_000_000: 1_546_300_000.0})


# --------------------------------------------------------------- API client --


class TestCursors:
    def test_cursor_is_base64_arrayconnection(self):
        """Quirk 3: we build cursors ourselves rather than paying to request them."""
        assert base64.b64decode(_cursor(0)).decode() == "arrayconnection:0"
        assert base64.b64decode(_cursor(41)).decode() == "arrayconnection:41"

    def test_page_size_fits_the_complexity_budget(self):
        """Quirk 2: cost is 12 per item for our field list, and the cap is 100.

        If someone adds a ninth field or raises the page size, this fails before
        the collector spends an hour discovering it against the live API.
        """
        n_fields = len(TX_FIELDS.split())
        assert n_fields == 8
        assert PAGE_SIZE * (n_fields + 4) <= 100


# ------------------------------------------------------------------ helpers --


class TestConversions:
    def test_wei_to_eth(self):
        assert _to_eth(str(10**18)) == 1.0
        assert _to_eth(str(5 * 10**17)) == 0.5

    def test_wei_to_gwei(self):
        assert _to_gwei("20000000000") == 20.0

    def test_conversions_tolerate_bad_input(self):
        """The API returns nulls and occasional junk; features must not explode."""
        assert _to_eth(None) == 0.0
        assert _to_eth("not a number") == 0.0
        assert _to_gwei(None) == 0.0

    def test_safe_ratio_guards_zero(self):
        """Addresses with no outgoing transactions are common, not exceptional."""
        assert _safe_ratio(1, 0) == 0.0
        assert _safe_ratio(1, 4) == 0.25


# ----------------------------------------------------------------- features --


class TestAddressFeatures:
    def test_counts_directions_separately(self, clock):
        record = {
            "address": ADDRESS,
            "label": 0,
            "transactions": [
                tx(ADDRESS, PEER_ONE),
                tx(ADDRESS, PEER_TWO),
                tx(PEER_ONE, ADDRESS),
            ],
        }
        row = address_features(record, clock)

        assert row["n_tx"] == 3
        assert row["n_sent"] == 2
        assert row["n_received"] == 1
        assert row["n_out_counterparties"] == 2
        assert row["n_in_counterparties"] == 1
        # PEER_ONE appears on both sides but is one counterparty.
        assert row["n_counterparties"] == 2

    def test_value_totals_and_concentration(self, clock):
        record = {
            "address": ADDRESS,
            "label": 0,
            "transactions": [
                tx(ADDRESS, PEER_ONE, value=str(10**18)),
                tx(ADDRESS, PEER_TWO, value=str(3 * 10**18)),
                tx(PEER_ONE, ADDRESS, value=str(2 * 10**18)),
            ],
        }
        row = address_features(record, clock)

        assert row["total_eth_sent"] == pytest.approx(4.0)
        assert row["total_eth_received"] == pytest.approx(2.0)
        assert row["net_eth_flow"] == pytest.approx(-2.0)
        assert row["max_eth_sent"] == pytest.approx(3.0)
        # The largest single transfer is 3 of the 4 ETH that left.
        assert row["value_concentration"] == pytest.approx(0.75)

    def test_sweep_pattern_concentrates_value(self, clock):
        """The shape we expect from a collection wallet that empties in one go."""
        transactions = [tx(f"0xpeer{i:036x}", ADDRESS, value=str(10**18), block=4_000_000 + i)
                        for i in range(10)]
        transactions.append(tx(ADDRESS, PEER_ONE, value=str(10 * 10**18), block=4_000_010))

        row = address_features({"address": ADDRESS, "label": 1,
                                "transactions": transactions}, clock)

        assert row["value_concentration"] == pytest.approx(1.0)
        assert row["n_received"] == 10
        assert row["n_sent"] == 1

    def test_failed_and_self_transactions(self, clock):
        record = {
            "address": ADDRESS,
            "label": 0,
            "transactions": [
                tx(ADDRESS, PEER_ONE, status="error"),
                tx(ADDRESS, PEER_ONE),
                tx(ADDRESS, ADDRESS),
                tx(ADDRESS, PEER_ONE),
            ],
        }
        row = address_features(record, clock)

        assert row["failed_ratio"] == pytest.approx(0.25)
        assert row["self_tx_ratio"] == pytest.approx(0.25)
        # A self-transaction has no counterparty and must not invent one.
        assert row["n_counterparties"] == 1

    def test_empty_address_yields_zeroed_row(self, clock):
        """Empty addresses are kept rather than dropped, so this path matters."""
        row = address_features({"address": ADDRESS, "label": 1, "transactions": []}, clock)

        assert row["n_tx"] == 0
        assert row["first_block"] == 0
        for name in BEHAVIOUR_FEATURES:
            assert name in row, f"{name} missing from an empty row"
            assert row[name] == 0.0

    def test_lifetime_uses_block_span(self, clock):
        record = {
            "address": ADDRESS,
            "label": 0,
            "transactions": [
                tx(ADDRESS, PEER_ONE, block=4_000_000),
                tx(PEER_ONE, ADDRESS, block=4_200_000),
            ],
        }
        row = address_features(record, clock)

        assert row["first_block"] == 4_000_000
        assert row["last_block"] == 4_200_000
        assert row["lifetime_days"] > 0

    def test_every_declared_feature_is_produced(self, clock):
        """Guards against a name being added to the list but never computed."""
        record = {
            "address": ADDRESS,
            "label": 0,
            "transactions": [tx(ADDRESS, PEER_ONE), tx(PEER_ONE, ADDRESS, block=4_100_000)],
        }
        row = address_features(record, clock)

        for name in BEHAVIOUR_FEATURES:
            assert name in row, f"{name} declared but never computed"


class TestNoBlockNumbersInFeatures:
    def test_absolute_blocks_are_not_model_features(self):
        """The leakage guard from the EDA: eras must not be learnable directly."""
        for banned in ("first_block", "last_block"):
            assert banned not in BEHAVIOUR_FEATURES
            assert banned not in GRAPH_FEATURES


# -------------------------------------------------------------------- clock --


class TestBlockClock:
    def test_interpolates_between_anchors(self, clock):
        midpoint = clock.timestamp(5_000_000)
        assert clock.timestamp(3_000_000) < midpoint < clock.timestamp(7_000_000)

    def test_is_monotonic(self, clock):
        assert clock.timestamp(4_000_000) < clock.timestamp(4_500_000)

    def test_extrapolates_outside_the_anchor_range(self, clock):
        """Blocks outside the anchors must still get an answer, not an exception."""
        assert clock.timestamp(1_000_000) < clock.timestamp(3_000_000)
        assert clock.timestamp(9_000_000) > clock.timestamp(7_000_000)

    def test_days_between_is_non_negative(self, clock):
        assert clock.days_between(5_000_000, 4_000_000) > 0


# -------------------------------------------------------------------- graph --


class TestGraph:
    def test_parallel_transfers_collapse_into_one_weighted_edge(self):
        import pandas as pd

        edges = pd.DataFrame([
            {"src": ADDRESS, "dst": PEER_ONE, "eth": 1.0, "block": 4_000_000},
            {"src": ADDRESS, "dst": PEER_ONE, "eth": 2.0, "block": 4_000_001},
            {"src": ADDRESS, "dst": PEER_TWO, "eth": 1.0, "block": 4_000_002},
        ])
        graph = build_graph(edges)

        assert graph.number_of_edges() == 2
        assert graph[ADDRESS][PEER_ONE]["weight"] == 2
        assert graph[ADDRESS][PEER_ONE]["eth"] == pytest.approx(3.0)

    def test_max_block_restricts_the_graph(self):
        import pandas as pd

        edges = pd.DataFrame([
            {"src": ADDRESS, "dst": PEER_ONE, "eth": 1.0, "block": 4_000_000},
            {"src": ADDRESS, "dst": PEER_TWO, "eth": 1.0, "block": 6_000_000},
        ])
        assert build_graph(edges, max_block=5_000_000).number_of_edges() == 1


class TestNeighbourRiskLeakage:
    """The most important tests here: this feature can leak the answer."""

    def _graph(self):
        graph = nx.DiGraph()
        graph.add_edge(ADDRESS, PEER_ONE, weight=1, eth=1.0)
        graph.add_edge(ADDRESS, PEER_TWO, weight=1, eth=1.0)
        return graph

    def test_only_known_labels_are_counted(self):
        """PEER_TWO's label is withheld, so it must not affect the ratio."""
        frame = graph_features(self._graph(), [ADDRESS], {PEER_ONE: 1})
        row = frame.iloc[0]

        assert row["n_labelled_neighbours"] == 1
        assert row["neighbour_risk_ratio"] == pytest.approx(1.0)

    def test_withheld_labels_change_the_answer(self):
        """The same graph gives a different ratio once the second label is visible."""
        frame = graph_features(self._graph(), [ADDRESS], {PEER_ONE: 1, PEER_TWO: 0})

        assert frame.iloc[0]["n_labelled_neighbours"] == 2
        assert frame.iloc[0]["neighbour_risk_ratio"] == pytest.approx(0.5)

    def test_address_never_sees_its_own_label(self):
        """A self-loop must not let an address read its own label back."""
        graph = self._graph()
        graph.add_edge(ADDRESS, ADDRESS, weight=1, eth=0.0)

        frame = graph_features(graph, [ADDRESS], {ADDRESS: 1, PEER_ONE: 0})
        row = frame.iloc[0]

        assert row["n_labelled_neighbours"] == 1
        assert row["neighbour_risk_ratio"] == pytest.approx(0.0)

    def test_no_labelled_neighbours_is_flagged_not_guessed(self):
        """Zero risk with zero evidence must be distinguishable from zero risk."""
        frame = graph_features(self._graph(), [ADDRESS], {})
        row = frame.iloc[0]

        assert row["neighbour_risk_ratio"] == 0.0
        assert row["n_labelled_neighbours"] == 0

    def test_address_missing_from_graph_gets_neutral_values(self):
        frame = graph_features(self._graph(), ["0xdead"], {PEER_ONE: 1})
        row = frame.iloc[0]

        assert row["degree"] == 0.0
        assert row["neighbour_risk_ratio"] == 0.0


# ------------------------------------------------------------------- labels --


class TestDedupe:
    def test_duplicate_addresses_collapse(self):
        """715 darklist rows become 652 addresses; duplicates would leak across splits."""
        records = [
            {"address": "0xABC", "comment": "scam one", "date": "2018-01-01"},
            {"address": "0xabc", "comment": "scam two", "date": "2018-06-01"},
            {"address": "0xdef", "comment": "scam three", "date": "2018-02-01"},
        ]
        deduped = dedupe_addresses(records)

        assert len(deduped) == 2
        assert {row["address"] for row in deduped} == {"0xabc", "0xdef"}
        assert all(row["label"] == 1 for row in deduped)


# -------------------------------------------------------------- saved model --


class TestSavedPipeline:
    """The spec is explicit that the scaler must travel with the model."""

    def test_bundle_contains_a_full_pipeline(self):
        model_path = ROOT / "models" / "model.joblib"
        if not model_path.exists():
            pytest.skip("no trained model yet; run python -m src.train")

        import joblib

        bundle = joblib.load(model_path)

        assert "pipeline" in bundle
        assert "scaler" in bundle["pipeline"].named_steps
        assert "model" in bundle["pipeline"].named_steps
        assert bundle["features"] == bundle["behaviour_features"] + bundle["graph_features"]

    def test_pipeline_scores_a_single_row(self):
        model_path = ROOT / "models" / "model.joblib"
        if not model_path.exists():
            pytest.skip("no trained model yet; run python -m src.train")

        from src.predict import build_row, load_model

        bundle = load_model(model_path)
        frame = build_row({name: 0.0 for name in bundle["features"]}, bundle)

        probability = bundle["pipeline"].predict_proba(frame)[0, 1]
        assert 0.0 <= probability <= 1.0
