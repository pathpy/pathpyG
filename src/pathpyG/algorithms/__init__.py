"""Algorithms for temporal path calculation and graph metrics.

The functions and submodules in this module allow to compute 
time-respecting or causal paths in temporal graphs and to
calculate (temporal) and higher-order graph metrics like centralities.

Usage Example:

    import pathpyG as pp
    pp.config['torch']['device'] = 'cuda'

    # Generate toy example for temporal graph
    g = pp.TemporalGraph.from_edge_list([
        ['b', 'c', 2],
        ['a', 'b', 1],
        ['c', 'd', 3],
        ['d', 'a', 4],
        ['b', 'd', 2],
        ['d', 'a', 6],
        ['a', 'b', 7]])

    # Extract DAG capturing causal interaction sequences in temporal graph
    dag = pp.algorithms.temporal_graph_to_event_dag(g, delta=1)

    # Calculate path statistics
    paths = pp.PathData.from_temporal_dag(dag)

    # Generate weighted (first-order) time-aggregated graph
    g = pp.HigherOrderGraph(paths, order=1)
    
    c = pp.algorithms.centralities.closeness_centrality(g)
"""

from pathpyG.algorithms.temporal import *
from pathpyG.algorithms import centrality
