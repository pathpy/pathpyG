from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric import EdgeIndex

from pathpyG.algorithms.temporal import extract_causal_paths, lift_order_temporal, temporal_shortest_paths
from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph


def walk_set(paths) -> list:
    """Return the walks of a PathData object as a sorted list of node ID tuples."""
    return sorted(paths.get_walk(i) for i in range(paths.num_paths))


def test_extract_causal_paths_branching(simple_temporal_graph):
    # (c,d,9) and (c,e,9) both continue (b,c,5), so the single path a->b->c branches in two.
    paths = extract_causal_paths(simple_temporal_graph, delta=4)
    assert walk_set(paths) == [("a", "b", "c", "d"), ("a", "b", "c", "e")]
    assert paths.data.dag_weight.tolist() == [1.0, 1.0]


def test_extract_causal_paths_delta_limits_continuation(simple_temporal_graph):
    # With delta=3 no edge can continue another, so every event is its own maximal path.
    paths = extract_causal_paths(simple_temporal_graph, delta=3)
    assert walk_set(paths) == [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e")]


def test_extract_causal_paths_merging():
    # Two distinct predecessors of (b,c,2) produce two maximal paths sharing the edge (b,c).
    g = TemporalGraph.from_edge_list([("a", "b", 1), ("x", "b", 1), ("b", "c", 2)])
    paths = extract_causal_paths(g, delta=1)
    assert walk_set(paths) == [("a", "b", "c"), ("x", "b", "c")]


def test_extract_causal_paths_repeated_node():
    # A node revisited at a later time must appear twice in the same path.
    g = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "a", 2), ("a", "c", 3)])
    paths = extract_causal_paths(g, delta=1)
    assert walk_set(paths) == [("a", "b", "a", "c")]


def test_extract_causal_paths_empty_graph():
    g = TemporalGraph.from_edge_list([])
    paths = extract_causal_paths(g, delta=1)
    assert paths.num_paths == 0


def test_extract_causal_paths_max_paths():
    with pytest.raises(RuntimeError, match="max_paths"):
        extract_causal_paths(
            TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("b", "d", 2)]), delta=1, max_paths=1
        )


def test_extract_causal_paths_covers_all_events(long_temporal_graph):
    # Every temporal edge must appear in at least one maximal path, and no path may be
    # shorter than a single edge.
    paths = extract_causal_paths(long_temporal_graph, delta=10)
    assert paths.num_paths > 0
    assert (paths.data.dag_num_nodes >= 2).all()

    walks = walk_set(paths)
    covered = {(w[i], w[i + 1]) for w in walks for i in range(len(w) - 1)}
    expected = {(s, d) for s, d, _ in long_temporal_graph.temporal_edges}
    assert expected.issubset(covered)


def test_lift_order_temporal(simple_temporal_graph):
    edge_index = lift_order_temporal(simple_temporal_graph, delta=5)
    event_graph = Graph.from_edge_index(edge_index)
    assert event_graph.n == simple_temporal_graph.m
    # for delta=5 we have three time-respecting paths (a,b,1) -> (b,c,5), (b,c,5) -> (c,d,9) and (b,c,5) -> (c,e,9)
    assert event_graph.m == 3
    assert torch.equal(event_graph.data.edge_index, EdgeIndex([[0, 1, 1], [1, 2, 3]]))


def test_temporal_shortest_paths(long_temporal_graph):
    dist, pred = temporal_shortest_paths(long_temporal_graph, delta=10)
    assert dist.shape == (long_temporal_graph.n, long_temporal_graph.n)
    assert pred.shape == (long_temporal_graph.n, long_temporal_graph.n)

    true_dist = np.array(
        [
            [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, float("inf")],
            [3.0, 0.0, 1.0, 2.0, 2.0, 1.0, 4.0, 5.0, 1.0],
            [2.0, float("inf"), 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
            ],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
            ],
            [1.0, float("inf"), float("inf"), float("inf"), float("inf"), 0.0, 2.0, 1.0, float("inf")],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                1.0,
                float("inf"),
            ],
            [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 1.0, float("inf"), 0.0, 1.0],
            [
                float("inf"),
                1.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
            ],
        ]
    )
    assert np.allclose(dist, true_dist, equal_nan=True)

    true_pred = np.array(
        [
            [0, 0, 0, 2, 2, 2, 0, 2, -1],
            [5, 1, 1, 2, 2, 1, 0, 6, 1],
            [5, -1, 2, 2, 2, 2, 0, 2, 2],
            [-1, -1, -1, 3, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 4, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, 5, 0, 5, -1],
            [-1, -1, -1, -1, -1, -1, 6, 6, -1],
            [-1, -1, -1, -1, -1, 7, -1, 7, 7],
            [-1, 8, -1, -1, -1, -1, -1, -1, 8],
        ]
    )
    assert np.allclose(pred, true_pred)
