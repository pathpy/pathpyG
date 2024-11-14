from __future__ import annotations

import numpy as np
import torch
from torch_geometric import EdgeIndex

from pathpyG.core.graph import Graph
from pathpyG.algorithms.temporal import temporal_shortest_paths, lift_order_temporal


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
