from __future__ import annotations

import numpy as _np

from pathpyG.core.Graph import Graph
from pathpyG.visualisations.hist_plots import hist
from pathpyG.algorithms import centrality
from pathpyG.algorithms.shortest_paths import shortest_paths_dijkstra


def test_shortest_paths_dijkstra(simple_graph_sp):
    dist, pred = shortest_paths_dijkstra(simple_graph_sp)
    print(pred)
    assert (dist == _np.matrix('0 1 2 2 3; 1 0 1 1 2; 2 1 0 2 1; 2 1 2 0 1; 3 2 1 1 0').A).all()
    assert (pred == _np.matrix('-9999 0 1 1 2; 1 -9999 1 1 3; 1 2 -9999 1 2; 1 3 1 -9999 3; 1 2 4 4 -9999').A).all()
