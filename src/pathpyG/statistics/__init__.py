"""Functions to compute various graph statistics.

The functions in this module allow to compute
various statistics on graphs

Example:
    ```py
    import pathpyG as pp

    # Generate a toy example graph.
    g = pp.Graph.from_edge_list([("b", "c"), ("a", "b"), ("c", "d"), ("d", "a"), ("b", "d")])

    # Calculate degree distribution and raw moments
    d_dist = pp.statistics.degree_distribution(g)
    k_1 = pp.statistics.degree_raw_moment(g, k=1)
    k_2 = pp.statistics.degree_raw_moment(g, k=2)
    ```
"""

from . import node_similarities
from .clustering import avg_clustering_coefficient, closed_triads, local_clustering_coefficient
from .degrees import (
    degree_assortativity,
    degree_central_moment,
    degree_distribution,
    degree_generating_function,
    degree_raw_moment,
    degree_sequence,
    mean_degree,
    mean_neighbor_degree,
)

__all__ = [
    "avg_clustering_coefficient",
    "closed_triads",
    "local_clustering_coefficient",
    "degree_assortativity",
    "degree_central_moment",
    "degree_distribution",
    "degree_generating_function",
    "degree_raw_moment",
    "degree_sequence",
    "mean_degree",
    "mean_neighbor_degree",
    "node_similarities",
]
