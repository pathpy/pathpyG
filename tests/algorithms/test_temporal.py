from __future__ import annotations

import numpy as np

from pathpyG.algorithms.temporal import temporal_shortest_paths


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
