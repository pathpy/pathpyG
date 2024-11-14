"""Algorithms to calculate connected components"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

from collections import Counter

import numpy as _np
from scipy.sparse.csgraph import connected_components as _cc

from pathpyG.core.graph import Graph

def connected_components(graph: Graph, connection='weak') -> (int, _np.ndarray):

    m = graph.sparse_adj_matrix()
    n, labels = _cc(m, directed=graph.is_directed(), connection=connection, return_labels=True)
    return n, labels

def largest_connected_component(graph: Graph, connection='weak') -> Graph:
    m = graph.sparse_adj_matrix()
    n, labels = _cc(m, directed=graph.is_directed(), connection=connection, return_labels=True)
    
    # find largest component C
    ctr = Counter(labels.tolist())
    x, x_c = ctr.most_common(1)[0]
    # create graph only consisting of nodes in C
    C = []
    for (v, w) in graph.edges:
        if labels[graph.mapping.to_idx(v)] == x and labels[graph.mapping.to_idx(w)] == x:
            C.append((v,w))
    return Graph.from_edge_list(C, is_undirected=graph.is_undirected(), num_nodes=x_c)


