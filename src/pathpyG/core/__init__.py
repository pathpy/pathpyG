"""Core classes for (temporal) graphs, paths, and higher-order De Bruijn graphs.

The classes in the `core` module can be used to implement integrated pipelines to
preprocess time-stamped network data, do inference and model selection of higher-order
De Bruijn graph models and address temporal graph learning tasks based on time-aware
graph neural networks.

Example:
    ```py
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

    # Compute first- and second-order De Bruijn graph model
    g1 = pp.HigherOrderGraph(paths, order=1, node_id=g.data["node_id"])
    g2 = pp.HigherOrderGraph(paths, order=2, node_id=g.data["node_id"])
    ```
"""
