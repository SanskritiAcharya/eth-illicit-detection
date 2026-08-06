"""Build the address-to-address graph and derive network features from it.

This module answers the project's actual research question: can an address's
network of connections reveal fraud that its own behaviour cannot?

The graph is built by us, from the transaction edges we collected. Every node is
an Ethereum address and every directed edge is "this address sent a transaction to
that address". Because we fetched the transactions of 2,852 labelled addresses and
each transaction names a counterparty, the graph is considerably larger than the
labelled set: it also contains the one-hop neighbours, which is exactly the
context the graph features are supposed to exploit.

Leakage
-------
`neighbour_risk_ratio` is the dangerous feature. It asks "what fraction of my
labelled neighbours are illicit?", and if it is allowed to see test-set labels
then the model is being told the answer through the back door: two scam addresses
that transact with each other would each reveal the other's label.

So `graph_features` takes a `known_labels` mapping and nothing else. Training
passes in the training split's labels only. A test address's own label, and the
labels of any other test address, are invisible while its features are computed.

Structural features (degree, PageRank, clustering, community) are computed on the
observed graph. Those are legitimately available at prediction time: an analyst
asked to score an address can look up who it has transacted with. What they cannot
look up is whether those neighbours have been *confirmed* as scams, which is
precisely the part we withhold.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

GRAPH_FEATURE_COLUMNS = [
    "degree",
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
    "community_size",
    "neighbour_risk_ratio",
    "n_labelled_neighbours",
]


def build_graph(edges: pd.DataFrame, max_block: int | None = None) -> nx.DiGraph:
    """Build a directed multigraph-free address graph from the edge list.

    Parallel transactions between the same pair collapse into one edge carrying a
    weight (how many transactions) and the total ETH moved. Collapsing keeps the
    graph small enough for PageRank to run in seconds, and the count is preserved
    as a weight so nothing is lost.

    `max_block` optionally restricts the graph to edges mined at or before a given
    block, which is how a strictly historical graph can be requested.
    """
    if max_block is not None:
        edges = edges[edges["block"] <= max_block]

    graph = nx.DiGraph()
    for src, dst, eth in zip(edges["src"], edges["dst"], edges["eth"]):
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += 1
            graph[src][dst]["eth"] += eth
        else:
            graph.add_edge(src, dst, weight=1, eth=eth)
    return graph


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Partition the graph into communities with the Louvain method.

    Louvain groups addresses that transact among themselves more than with the
    rest of the network. Scam operations often run several wallets that move funds
    between each other, so they tend to land in the same small, tight community --
    which is why community size is worth measuring.

    Louvain needs an undirected graph; direction is irrelevant to whether two
    addresses belong to the same cluster.
    """
    undirected = graph.to_undirected()
    communities = nx.algorithms.community.louvain_communities(
        undirected, weight="weight", seed=404
    )
    return {
        node: community_id
        for community_id, members in enumerate(communities)
        for node in members
    }


def graph_features(
    graph: nx.DiGraph,
    addresses: list[str],
    known_labels: dict[str, int],
) -> pd.DataFrame:
    """Compute network features for the given addresses.

    `known_labels` must contain only labels the model is allowed to see -- in
    training, that is the training split alone. Neighbours whose label is unknown
    are excluded from the risk ratio rather than assumed licit, so an address with
    no labelled neighbours scores 0 with `n_labelled_neighbours == 0`, and the
    model can learn to distrust the ratio when the count is low.
    """
    undirected = graph.to_undirected()

    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    # Clustering on the undirected view: what fraction of an address's neighbours
    # also transact with each other. Tight, mutually-connected clusters look very
    # different from a hub that pays out to many unrelated wallets.
    clustering = nx.clustering(undirected)
    community_of = detect_communities(graph) if graph.number_of_nodes() else {}

    community_sizes: dict[int, int] = {}
    for community_id in community_of.values():
        community_sizes[community_id] = community_sizes.get(community_id, 0) + 1

    rows = []
    for address in addresses:
        if address not in graph:
            rows.append(
                {"address": address, **{column: 0.0 for column in GRAPH_FEATURE_COLUMNS}}
            )
            continue

        neighbours = set(undirected.neighbors(address))
        # An address must never contribute its own label to its own feature.
        labelled = [
            known_labels[n] for n in neighbours if n in known_labels and n != address
        ]
        community_id = community_of.get(address)

        rows.append(
            {
                "address": address,
                "degree": undirected.degree(address),
                "in_degree": graph.in_degree(address),
                "out_degree": graph.out_degree(address),
                "pagerank": pagerank.get(address, 0.0),
                "clustering": clustering.get(address, 0.0),
                "community_size": community_sizes.get(community_id, 0),
                "neighbour_risk_ratio": (sum(labelled) / len(labelled)) if labelled else 0.0,
                "n_labelled_neighbours": len(labelled),
            }
        )

    return pd.DataFrame(rows)


def graph_summary(graph: nx.DiGraph) -> dict:
    """A few headline numbers about the graph, for the notebook and the report."""
    undirected = graph.to_undirected()
    degrees = [d for _, d in undirected.degree()]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "mean_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "components": nx.number_connected_components(undirected),
        "density": nx.density(graph),
    }


if __name__ == "__main__":
    edges = pd.read_csv(PROCESSED / "edges.csv")
    graph = build_graph(edges)
    for key, value in graph_summary(graph).items():
        print(f"  {key}: {value}")
