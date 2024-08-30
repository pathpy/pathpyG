"""Core classes for (temporal) graphs, paths, and higher-order De Bruijn graphs.

The classes in the `core` module can be used to implement integrated pipelines to
preprocess time-stamped network data, do inference and model selection of higher-order
De Bruijn graph models and address temporal graph learning tasks based on time-aware
graph neural networks.

Example:
    ```py
    import pathpyG as pp
    pp.config['torch']['device'] = 'cuda'

    # Generate toy example temporal graph
    g = pp.TemporalGraph.from_edge_list([
        ('b', 'c', 2),
        ('a', 'b', 1),
        ('c', 'd', 3),
        ('d', 'a', 4),
        ('b', 'd', 2),
        ('d', 'a', 6),
        ('a', 'b', 7)])

    # Create Multi-Order model that models time-respecting paths
    m = pp.MultiOrderModel.from_temporal_graph(g, delta=1, max_order=3)
    print(m.layers[1])
    print(m.layers[2])
    print(m.layers[3])
    ```
"""
