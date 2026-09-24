"""Tests for the temporal graph generators with planted higher-order structure."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from pathpyG.algorithms.temporal_generative_models import higher_order_block_walks, temporal_clusters
from pathpyG.core.multi_order_model import MultiOrderModel


def weighted_first_order_graph(graph):
    """The time-aggregated weighted graph, as a dictionary of edge multiplicities."""
    edges, counts = torch.unique(graph.data.edge_index.as_tensor(), dim=1, return_counts=True)
    return {(int(source), int(destination)): int(count) for source, destination, count in zip(*edges, counts)}


def walk_nodes(graph, order):
    """The (v_0, ..., v_order) node sequences of the generated walks."""
    edge_index = graph.data.edge_index.as_tensor()
    columns = [edge_index[0][step::order] for step in range(order)]
    columns.append(edge_index[1][order - 1 :: order])
    return torch.stack(columns)


def test_temporal_clusters_has_the_published_shape():
    graph = temporal_clusters(seed=0)
    assert graph.n == 30
    assert graph.m == 60000
    assert torch.unique(graph.data.edge_index.as_tensor(), dim=1).size(1) == 560
    assert graph.data.node_cluster.tolist() == [node // 10 for node in range(30)]


def test_temporal_clusters_preserves_the_first_order_graph():
    """Exchanging time stamps leaves every time-stamped edge in place, by construction."""
    arguments = dict(num_nodes=12, num_edges=60, num_clusters=3, num_walks=2000, seed=7)
    rewired = temporal_clusters(cluster_strength=1.0, **arguments)
    untouched = temporal_clusters(cluster_strength=0.0, **arguments)
    assert weighted_first_order_graph(rewired) == weighted_first_order_graph(untouched)


def test_temporal_clusters_plants_structure_only_in_the_causal_topology():
    graph = temporal_clusters(num_nodes=12, num_edges=60, num_clusters=3, num_walks=4000, seed=7)
    cluster = graph.data.node_cluster
    walks = walk_nodes(graph, order=2)

    first_order_agreement = float((cluster[walks[0]] == cluster[walks[1]]).float().mean())
    second_order_agreement = float((cluster[walks[0]] == cluster[walks[2]]).float().mean())
    # The first step is as uninformative as chance; the second is strongly clustered. How far
    # above chance the second step gets is limited by how many walks share each middle node, so
    # the assertion is on the gap rather than on an absolute level.
    assert first_order_agreement == pytest.approx(1 / 3, abs=0.06)
    assert second_order_agreement > 0.6
    assert second_order_agreement > 1.5 * first_order_agreement


@pytest.mark.parametrize("cluster_strength", [0.0, 0.5, 1.0])
def test_temporal_clusters_strength_controls_the_signal(cluster_strength):
    graph = temporal_clusters(
        num_nodes=12, num_edges=60, num_clusters=3, num_walks=4000, cluster_strength=cluster_strength, seed=7
    )
    cluster = graph.data.node_cluster
    walks = walk_nodes(graph, order=2)
    agreement = float((cluster[walks[0]] == cluster[walks[2]]).float().mean())
    if cluster_strength == 0.0:
        assert agreement == pytest.approx(1 / 3, abs=0.06)
    else:
        assert agreement > 1 / 3 + 0.1 * cluster_strength


def test_temporal_clusters_is_reproducible():
    arguments = dict(num_nodes=12, num_edges=60, num_clusters=3, num_walks=500)
    first = temporal_clusters(seed=3, **arguments)
    again = temporal_clusters(seed=3, **arguments)
    different = temporal_clusters(seed=4, **arguments)
    assert torch.equal(first.data.edge_index.as_tensor(), again.data.edge_index.as_tensor())
    assert not torch.equal(first.data.edge_index.as_tensor(), different.data.edge_index.as_tensor())


def test_temporal_clusters_detects_its_own_order():
    graph = temporal_clusters(num_nodes=12, num_edges=60, num_clusters=3, num_walks=4000, seed=7)
    model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=3)
    assert model.estimate_order(method="likelihood_ratio") == 2
    assert model.estimate_order(method="bayesian") == 2


def test_temporal_clusters_validates_its_arguments():
    with pytest.raises(ValueError, match="divide num_nodes"):
        temporal_clusters(num_nodes=10, num_clusters=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        temporal_clusters(num_nodes=5, num_edges=100, num_clusters=5)
    with pytest.raises(ValueError, match="cluster_strength"):
        temporal_clusters(num_nodes=12, num_edges=60, num_clusters=3, num_walks=10, cluster_strength=2.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_higher_order_block_walks_has_the_requested_order(order):
    """The lagged dependency makes the ground-truth Markov order exactly `order`."""
    max_order = order + 1
    graph = higher_order_block_walks(
        num_nodes=12,
        num_clusters=3,
        num_walks=4000,
        order=order,
        walk_length=max_order,
        intra_cluster_probability=0.9,
        walk_separation=3,
        seed=0,
    )
    model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=max_order)
    assert model.estimate_order(method="likelihood_ratio") == order
    assert model.estimate_order(method="bayesian") == order


def test_higher_order_block_walks_recovers_its_probability():
    probability = 0.75
    order = 2
    graph = higher_order_block_walks(
        num_nodes=12,
        num_clusters=3,
        num_walks=8000,
        order=order,
        intra_cluster_probability=probability,
        walk_separation=3,
        seed=1,
    )
    cluster = graph.data.node_cluster
    walks = walk_nodes(graph, order=order)
    # Drawing inside the cluster with probability p, and uniformly otherwise, agrees with the
    # lagged node's cluster with probability p + (1 - p) / num_clusters.
    expected = probability + (1 - probability) / 3
    assert float((cluster[walks[0]] == cluster[walks[2]]).float().mean()) == pytest.approx(expected, abs=0.02)
    # Everything below the order is uniform by construction.
    assert float((cluster[walks[0]] == cluster[walks[1]]).float().mean()) == pytest.approx(1 / 3, abs=0.02)


def test_higher_order_block_walks_attaches_clusters_and_validates():
    graph = higher_order_block_walks(num_nodes=12, num_clusters=4, num_walks=10, seed=0)
    assert graph.data.node_cluster.tolist() == [node // 3 for node in range(12)]
    with pytest.raises(ValueError, match="divide num_nodes"):
        higher_order_block_walks(num_nodes=10, num_clusters=3, num_walks=10)
    with pytest.raises(ValueError, match="must be at least order"):
        higher_order_block_walks(num_nodes=12, num_clusters=3, num_walks=10, order=3, walk_length=2)
    with pytest.raises(ValueError, match="intra_cluster_probability"):
        higher_order_block_walks(num_nodes=12, num_clusters=3, num_walks=10, intra_cluster_probability=1.5)
