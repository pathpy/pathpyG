"""Algorithms to calculate shortest paths in static networks

The functions  in this module allow to compute shortest paths
in static networks."""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

import numpy as _np
from scipy.sparse.csgraph import dijkstra

from pathpyG.core.graph import Graph


def shortest_paths_dijkstra(graph: Graph) -> (_np.ndarray, _np.ndarray):

    m = graph.sparse_adj_matrix()

    dist, pred = dijkstra(m, directed=graph.is_directed(), return_predecessors=True, unweighted=True)

    return dist, pred


def diameter(graph: Graph) -> float:

    m = graph.sparse_adj_matrix()

    dist = dijkstra(m, directed=graph.is_directed(), return_predecessors=False, unweighted=True)
    return _np.max(dist)


def avg_path_length(graph: Graph) -> float:

    m = graph.sparse_adj_matrix()

    dist = dijkstra(m, directed=graph.is_directed(), return_predecessors=False, unweighted=True)
    return _np.sum(dist) / (graph.n * (graph.n - 1))
