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

from pathpyG.core.Graph import Graph

from scipy.sparse.csgraph import dijkstra

def shortest_paths_dijkstra(graph: Graph) -> (_np.ndarray, _np.ndarray):

    m = graph.get_sparse_adj_matrix()

    # run disjktra for all source nodes
    dist, pred = dijkstra(m, directed=graph.is_directed(), return_predecessors=True, unweighted=True)

    return dist, pred
