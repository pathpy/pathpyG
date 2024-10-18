"""Algorithms for temporal path calculation and graph metrics.

The functions and submodules in this module allow to compute 
time-respecting or causal paths in temporal graphs and to
calculate (temporal) and higher-order graph metrics like centralities.

Example:
    ```py
    # Import pathpyG and configure your torch device if you want to use GPU .
    import pathpyG as pp
    pp.config['torch']['device'] = 'cuda'

    # Generate a toy example for a temporal graph.
    g = pp.TemporalGraph.from_edge_list([
        ('b', 'c', 2),
        ('a', 'b', 1),
        ('c', 'd', 3),
        ('d', 'a', 4),
        ('b', 'd', 2),
        ('d', 'a', 6),
        ('a', 'b', 7)
    ])

    # Extract DAG capturing causal interaction sequences in temporal graph.
    e_i = pp.algorithms.lift_order_temporal(g, delta=1)
    dag = pp.Graph.from_edge_index(e_i)
    print(dag)

    # Calculate shortest time-respecting pathas
    dist, pred = pp.algorithms.temporal.temporal_shortest_paths(g, delta=1)
    ```
"""

from pathpyG.algorithms.temporal import *
from pathpyG.algorithms import centrality
from pathpyG.algorithms.random_graphs import Watts_Strogatz
from pathpyG.algorithms import generative_models
from pathpyG.algorithms import shortest_paths
from pathpyG.algorithms.components import connected_components, largest_connected_component
from pathpyG.algorithms.rolling_time_window import RollingTimeWindow
from pathpyG.algorithms.weisfeiler_leman import WeisfeilerLeman_test

