"""Algorithms to calculate connected components."""

from collections import Counter
from typing import (
    Tuple,
)

import numpy as _np
from scipy.sparse.csgraph import connected_components as _cc

from pathpyG.core.graph import Graph


def connected_components(graph: Graph, connection="weak") -> Tuple[int, _np.ndarray]:
    """Compute the connected components of a graph.

    Args:
        graph (Graph): The input graph.
        connection (str, optional): Type of connection to consider. 
            Options are "weak" or "strong". Defaults to "weak".
    
    Returns:
        Tuple[int, np.ndarray]: A tuple containing the number of connected components and
            an array with component labels for each node.
    """
    m = graph.sparse_adj_matrix()
    n, labels = _cc(m, directed=graph.is_directed(), connection=connection, return_labels=True)
    return n, labels


def largest_connected_component(graph: Graph, connection="weak") -> Graph:
    """Extract the largest connected component from a graph.

    Args:
        graph (Graph): The input graph.
        connection (str, optional): Type of connection to consider. 
            Options are "weak" or "strong". Defaults to "weak".
    
    Returns:
        Graph: A new graph instance containing only the largest connected component.
    """
    m = graph.sparse_adj_matrix()
    _, labels = _cc(m, directed=graph.is_directed(), connection=connection, return_labels=True)

    # find largest component C
    ctr = Counter(labels.tolist())
    x, _ = ctr.most_common(1)[0]
    # create graph only consisting of nodes in C
    C = []
    for v, w in graph.edges:
        if labels[graph.mapping.to_idx(v)] == x and labels[graph.mapping.to_idx(w)] == x:
            C.append((v, w))
    return Graph.from_edge_list(C, is_undirected=graph.is_undirected())
