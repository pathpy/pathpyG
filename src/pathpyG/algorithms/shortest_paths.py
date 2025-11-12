"""Algorithms to calculate shortest paths in static networks.

The functions  in this module allow to compute shortest paths
in static networks.
"""

import numpy as _np
from scipy.sparse.csgraph import dijkstra

from pathpyG.core.graph import Graph


def shortest_paths_dijkstra(graph: Graph) -> tuple[_np.ndarray, _np.ndarray]:
    """Compute shortest paths using Dijkstra's algorithm.

    Args:
        graph (Graph): Input graph.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the distance matrix and the predecessor matrix.
    """
    m = graph.sparse_adj_matrix()
    dist, pred = dijkstra(m, directed=graph.is_directed(), return_predecessors=True, unweighted=True)
    return dist, pred


def diameter(graph: Graph) -> float:
    """Compute the diameter of the graph.

    Args:
        graph (Graph): Input graph.

    Returns:
        float: The diameter of the graph.
    """
    m = graph.sparse_adj_matrix()
    dist = dijkstra(m, directed=graph.is_directed(), return_predecessors=False, unweighted=True)
    return _np.max(dist)


def avg_path_length(graph: Graph) -> float:
    """Compute the average path length of the graph.

    Args:
        graph (Graph): Input graph.
        
    Returns:
        float: The average path length of the graph.
    """
    m = graph.sparse_adj_matrix()
    dist = dijkstra(m, directed=graph.is_directed(), return_predecessors=False, unweighted=True)
    return _np.sum(dist) / (graph.n * (graph.n - 1))
