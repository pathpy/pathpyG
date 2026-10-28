import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Working with Higher-Order De Bruijn Graphs

    ## Prerequisites

    First, we need to set up our Python environment that has PyTorch, PyTorch Geometric and PathpyG installed. Depending on where you are executing this notebook, this might already be (partially) done. E.g. Google Colab has PyTorch installed by default so we only need to install the remaining dependencies. The DevContainer that is part of our GitHub Repository on the other hand already has all of the necessary dependencies installed.

    In the following, we install the packages for usage in Google Colab using Jupyter magic commands. For other environments comment in or out the commands as necessary. For more details on how to install `pathpyG` especially if you want to install it with GPU-support, we refer to our [documentation](https://www.pathpy.net/dev/getting_started/). Note that `%%capture` discards the full output of the cell to not clutter this tutorial with unnecessary installation details. If you want to print the output, you can comment `%%capture` out.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # # !pip install torch
    # # !pip install torch_geometric
    # # !pip install git+https://github.com/pathpy/pathpyG.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation and Learning Objectives

    In [Path Data and Higher-Order Models](../paths_higher_order/), we have introduced $k$-th order De Bruijn graphs, whose nodes are paths of $k$ nodes and whose edges capture paths of length $k$, and we have used the layers of a `MultiOrderModel` to obtain them. Each of those layers is an instance of the class `HigherOrderGraph`, a subclass of `Graph` whose nodes are paths of first-order nodes rather than the first-order nodes themselves.

    In this tutorial, we take a closer look at the `HigherOrderGraph` class. You will learn how to

    - create a De Bruijn graph of a given order directly from path data or a graph,
    - map higher-order nodes back to the first-order paths they represent,
    - build a higher-order graph from an edge index, which lets us contrast the paths that were actually *observed* with the paths that the topology of a graph *permits*,
    - use the memoryless model of order zero,
    - project a higher-order graph back onto the first-order nodes, and
    - connect higher-order nodes to first-order nodes, which is the basis of causality-aware graph neural networks.
    """)
    return


@app.cell
def _():
    import pathpyG as pp

    return (pp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the same example as in [Path Data and Higher-Order Models](../paths_higher_order/): a directed graph with five nodes, in which we observe four walks `a` -> `c` -> `d` and four walks `b` -> `c` -> `e`.
    """)
    return


@app.cell
def _(pp):
    g = pp.Graph.from_edge_list([('a', 'c'), ('b', 'c'), ('c', 'd'), ('c', 'e')])

    paths = pp.PathData(g.mapping)
    paths.append_walk(('a', 'c', 'd'), weight=4.0)
    paths.append_walk(('b', 'c', 'e'), weight=4.0)
    pp.plot(g);
    return g, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Higher-Order Graphs

    If we are only interested in a single order $k$, we can directly create the corresponding De Bruijn graph using the class method `HigherOrderGraph.from_path_data`. For $k=2$, we obtain:
    """)
    return


@app.cell
def _(paths, pp):
    g2 = pp.HigherOrderGraph.from_path_data(paths, order=2)
    print(g2)
    return (g2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The summary tells us the order of the graph as well as the number of first-order nodes from which the higher-order nodes are built. The nodes of a higher-order graph with order $k>1$ are identified by tuples of $k$ first-order node IDs, and its edges are pairs of such tuples. The `edge_weight` attribute counts how often each path of length $k$ was observed:
    """)
    return


@app.cell
def _(g2):
    print(g2.order)
    print(g2.nodes)
    for _e in g2.edges:
        print(_e, g2['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Internally, the first-order paths represented by higher-order nodes are stored in the tensor `node_sequence`, which has shape `(n, k)` and contains the first-order node indices of each path. The function `node_id` uses this tensor along with the `first_order_mapping` to map the index of a higher-order node to the path it represents:
    """)
    return


@app.cell
def _(g2):
    print(g2.data.node_sequence)
    print(g2.node_id(0))
    print(g2.first_order_mapping)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only a `HigherOrderGraph` has an order and a node sequence. A plain `Graph` is always a first-order graph whose nodes are entities rather than paths, and creating a `Graph` from a `Data` object that contains a `node_sequence` raises an error. Since `HigherOrderGraph` is a subclass of `Graph`, everything you have learned about graphs in the previous tutorials, e.g. traversing successors, computing degrees, or plotting, works for higher-order graphs as well:
    """)
    return


@app.cell
def _(g2, pp):
    print(list(g2.successors(('a', 'c'))))
    pp.plot(g2);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that each call of `from_path_data` constructs all lower orders internally before returning the graph of the requested order. If you need several orders, it is more efficient to create a `MultiOrderModel` once and access its layers, which are `HigherOrderGraph` instances as well:
    """)
    return


@app.cell
def _(paths, pp):
    m = pp.MultiOrderModel.from_path_data(paths, max_order=2)
    print(type(m.layers[2]).__name__)
    print(m.layers[2].nodes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The First-Order Case

    A De Bruijn graph of order one is simply the weighted first-order graph. Different from other orders, its nodes are identified by plain node IDs rather than tuples. We can create it either from path data, where edge weights count how often each edge was traversed, or from a `Graph` object using `HigherOrderGraph.from_graph`, which coalesces multi-edges into weighted edges:
    """)
    return


@app.cell
def _(g, paths, pp):
    g1 = pp.HigherOrderGraph.from_path_data(paths, order=1)
    print(g1.order, g1.nodes)
    print(g1.data.edge_weight)

    g1_from_graph = pp.HigherOrderGraph.from_graph(g)
    print(g1_from_graph.order, g1_from_graph.nodes)
    return (g1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we ask for an order that is larger than the longest observed path, we obtain a higher-order graph without edges. In our example, the longest walks have length two, so the third-order graph has two nodes (the two walks of length two) but no edges:
    """)
    return


@app.cell
def _(paths, pp):
    print(pp.HigherOrderGraph.from_path_data(paths, order=3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combining Higher-Order Graphs

    We can combine two higher-order graphs of the same order using the `+` operator. This is useful, e.g., if we have observed paths in two separate data sets. Nodes are matched by the paths they represent, and edges that occur in both graphs, i.e. repeated observations of the same path, are merged into a single edge whose weight is the sum of both weights. Let us assume that we have additionally observed the walk `a` -> `c` -> `d` once more as well as a new walk `c` -> `d` -> `a`:
    """)
    return


@app.cell
def _(g2, pp):
    more_paths = pp.PathData(pp.IndexMap(list('abcde')))
    more_paths.append_walk(('a', 'c', 'd'), weight=1.0)
    more_paths.append_walk(('c', 'd', 'a'), weight=1.0)

    g2_combined = g2 + pp.HigherOrderGraph.from_path_data(more_paths, order=2)
    for _e in g2_combined.edges:
        print(_e, g2_combined['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observed vs. Possible Paths

    So far, we have created higher-order graphs from observed paths. We can also create a higher-order graph directly from an edge index whose nodes are paths, using the class method `HigherOrderGraph.aggregate`. It merges all nodes that represent the same path and sums the weights of edges between them.

    Let us use this to construct the second-order graph of all paths of length two that the *topology* of our first-order graph `g1` permits. The function `lift_order_step` performs a line graph transformation, i.e. it turns the edges of `g1` into nodes and connects two of them whenever they overlap in a first-order node, while extending the node sequences accordingly:
    """)
    return


@app.cell
def _(g1, pp):
    _edge_index, _node_sequence, _edge_weight = pp.algorithms.lift_order.lift_order_step(
        g1.data.edge_index.as_tensor(), g1.data.node_sequence, edge_weight=g1.data.edge_weight
    )
    g2_possible = pp.HigherOrderGraph.aggregate(
        _edge_index,
        _node_sequence,
        first_order_mapping=g1.first_order_mapping,
        edge_weight=_edge_weight,
        n_first_order=g1.n_first_order,
    )
    print(g2_possible)
    for _e in g2_possible.edges:
        print(_e)
    pp.plot(g2_possible);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This graph contains all four paths of length two through the center node `c`, while the second-order graph `g2` that we created from the observed walks contains only two of them. The two "missing" edges `(a, c) -> (c, e)` and `(b, c) -> (c, d)` correspond to paths that are possible in the topology of the first-order graph, but that never occur in the data. This is exactly the information about the **causal topology** that is lost in a first-order graph, and that De Bruijn graphs of higher order preserve.

    For details on how the line graph transformation and the aggregation are implemented based on tensor operations, see [Implementation Concepts](../implementation_concepts/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Memoryless Model of Order Zero

    The order $k$ of a De Bruijn graph can be viewed as the *memory* of a model for paths: in a model of order $k$, the next node of a path depends on the last $k$ nodes. A first-order model thus only remembers the current node, i.e. it corresponds to a Markov chain on the first-order graph.

    We can go one step further and consider a model of order zero without any memory, in which each node of a path is drawn independently from a fixed distribution over the first-order nodes, irrespective of the edges of the graph. For path data, we can create this model with `order=0`:
    """)
    return


@app.cell
def _(paths, pp):
    g0 = pp.HigherOrderGraph.from_path_data(paths, order=0)
    print(g0)
    print(g0.nodes)
    return (g0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The order-zero graph has a single node, the empty path `()`, with one self-loop for each first-order node. Since all loops share the same endpoints, the edge attribute `edge_first_order_node` stores which first-order node each loop represents, while `edge_weight` counts how often that node is visited by the observed paths. Node `c` is visited by all eight walks, all other nodes by four walks each. The transition probabilities of the self-loops are thus the probabilities to visit each of the first-order nodes:
    """)
    return


@app.cell
def _(g0):
    print(g0.data.edge_first_order_node)
    print(g0.data.edge_weight)
    print(g0.transition_probabilities(edge_attr='edge_weight'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also create an order-zero graph from given node weights using the class method `HigherOrderGraph.from_node_weights`. A `MultiOrderModel` created from path data always contains the order-zero layer, which allows the method `estimate_order` introduced in [Path Data and Higher-Order Models](../paths_higher_order/) to detect that a set of paths does not even warrant a first-order model. Let us test this for walks in which each node is chosen uniformly at random, independently of the previous nodes:
    """)
    return


@app.cell
def _(mo, pp):
    import random

    random.seed(42)
    random_paths = pp.PathData(pp.IndexMap(list('abc')))
    for _ in range(200):
        random_paths.append_walk(tuple(random.choice('abc') for _ in range(4)))

    # Hide a warning about the beta status of sparse tensors in PyTorch
    with mo.capture_stderr():
        m_random = pp.MultiOrderModel.from_path_data(random_paths, max_order=2)
        estimated_order = m_random.estimate_order(random_paths, significance_threshold=0.01)
    print(sorted(m_random.layers))
    print(estimated_order)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the order-zero model is only defined for path data. For temporal graphs, `MultiOrderModel.from_temporal_graph` starts with the first-order layer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projecting Higher-Order Graphs to First-Order Graphs

    The function `to_first_order` projects a higher-order graph onto the first-order nodes. Each higher-order node is replaced by one of the nodes of the path it represents, and the weights of all higher-order edges that are mapped to the same first-order edge are summed up. With `mode='last'`, each higher-order node is represented by the last node of its path. The second-order edge `(a, c) -> (c, d)` is thus projected onto the first-order edge `c -> d`:
    """)
    return


@app.cell
def _(g2):
    g_last = g2.to_first_order(mode='last')
    for _e in g_last.edges:
        print(_e, g_last['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `mode='first'`, each higher-order node is instead represented by the first node of its path, i.e. the edge `(a, c) -> (c, d)` is projected onto `a -> c`:
    """)
    return


@app.cell
def _(g2):
    g_first = g2.to_first_order(mode='first')
    for _e in g_first.edges:
        print(_e, g_first['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First-order nodes that do not represent any higher-order node remain in the projected graph as isolated nodes, i.e. `a` and `b` for `mode='last'`, and `d` and `e` for `mode='first'`.

    Note that `to_first_order` is a projection rather than the inverse of the construction of a higher-order graph: neither projection above recovers the four edges of our original first-order graph `g`. Since an order-zero graph has no first-order nodes to project onto, `to_first_order` cannot be applied to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connecting Higher-Order and First-Order Nodes

    Many applications, e.g. node-level predictions with causality-aware graph neural networks, require us to combine information from higher-order nodes with first-order nodes. The function `bipartite_edge_index` returns an edge index that connects each higher-order node to the first-order node at the end (`mapping='last'`), at the start (`mapping='first'`) or at both ends (`mapping='both'`) of its path. The first row contains the indices of higher-order nodes, the second row the indices of first-order nodes:
    """)
    return


@app.cell
def _(g2):
    bipartite = g2.bipartite_edge_index(mapping='last')
    print(bipartite)
    for _ho, _fo in bipartite.t().tolist():
        print(g2.node_id(_ho), '->', g2.first_order_mapping.to_id(_fo))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    De Bruijn Graph Neural Networks use exactly this bipartite edge index to map the latent representations of higher-order nodes to first-order nodes, as we will see in [De Bruijn Graph Neural Networks](../dbgnn/).

    So far, we have built higher-order graphs from path data, where the paths are given explicitly. In the next tutorial, [Event Graphs](../event_graphs/), we show how we can construct `HigherOrderGraph` objects for time-respecting paths in temporal graphs.
    """)
    return


if __name__ == "__main__":
    app.run()
