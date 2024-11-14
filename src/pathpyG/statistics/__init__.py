"""Functions to compute various graph statistics.

The functions in this module allow to compute 
various statistics on graphs

Example:
    ```py
    import pathpyG as pp

    # Generate a toy example graph.
    g = pp.Graph.from_edge_list([
        ('b', 'c'),
        ('a', 'b'),
        ('c', 'd'),
        ('d', 'a'),
        ('b', 'd')
    ])

    # Calculate degree distribution and raw moments
    d_dist = pp.statistics.degree_distribution(g)
    k_1 = pp.statistics.degree_raw_moment(g, k=1)
    k_2 = pp.statistics.degree_raw_moment(g, k=2)
    ```
"""

from pathpyG.statistics.degrees import *
from pathpyG.statistics.clustering import *
from pathpyG.statistics import node_similarities
