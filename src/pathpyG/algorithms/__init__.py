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
        ['b', 'c', 2],
        ['a', 'b', 1],
        ['c', 'd', 3],
        ['d', 'a', 4],
        ['b', 'd', 2],
        ['d', 'a', 6],
        ['a', 'b', 7]
    ])

    # Extract DAG capturing causal interaction sequences in temporal graph.
    dag = pp.algorithms.temporal_graph_to_event_dag(g, delta=1)

    # Get path object to calculate statistics.
    paths = pp.PathData.from_temporal_dag(dag)

    # Generate time-aggregated static representation of temporal graph
    s = g.to_static_graph()

    # Calculate centrality on static graph using networkx
    c = pp.algorithms.centrality.closeness_centrality(s)
    ```
"""

from pathpyG.algorithms.temporal import *
from pathpyG.algorithms import centrality
from pathpyG.algorithms.RollingTimeWindow import RollingTimeWindow