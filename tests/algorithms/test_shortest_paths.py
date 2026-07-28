from __future__ import annotations

import numpy as _np

from pathpyG.algorithms.shortest_paths import avg_path_length, diameter, shortest_paths_dijkstra


def test_shortest_paths_dijkstra(simple_graph_sp):
    dist, pred = shortest_paths_dijkstra(simple_graph_sp)
    assert (dist == _np.matrix("0 1 2 2 3; 1 0 1 1 2; 2 1 0 2 1; 2 1 2 0 1; 3 2 1 1 0").A).all()
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i != j:
                assert dist[i, j] == dist[i, pred[i, j]] + 1


def test_diameter(simple_graph_sp):
    assert diameter(simple_graph_sp) == 3


def test_avg_path_length(simple_graph_sp):
    assert avg_path_length(simple_graph_sp) == 1.6
