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
    # Basic pathpyG Concepts

    ## Prerequisites

    First, we need to set up our Python environment that has PyTorch, PyTorch Geometric and PathpyG installed. Depending on where you are executing this notebook, this might already be (partially) done. E.g. Google Colab has PyTorch installed by default so we only need to install the remaining dependencies. The DevContainer that is part of our GitHub Repository on the other hand already has all of the necessary dependencies installed.

    In the following, we install the packages for usage in Google Colab using Jupyter magic commands. For other environments comment in or out the commands as necessary. For more details on how to install `pathpyG` especially if you want to install it with GPU-support, we refer to our [documentation](https://www.pathpy.net/dev/getting_started/). Note that `%%capture` discards the full output of the cell to not clutter this tutorial with unnecessary installation details. If you want to print the output, you can comment `%%capture` out.
    """)
    return


@app.cell
def _():
    # %%capture
    # # !pip install torch
    # # !pip install torch_geometric
    # # !pip install git+https://github.com/pathpy/pathpyG.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation and Learning Objectives

    This first step of our multi-stage introductory tutorial introduces key concepts of `pathpyG`. While `pathpyG` targets GPU-accelerated analysis and learning using higher-order graph models for time series data on graphs, it can also be used to represent, analyze and interactively visualize static graphs. For this, it provides a `Graph` class that is build around the `torch_geometric.data.Data` object, which has the advantage that we can directly apply `pyG` transforms and use the `Graph` object for deep graph learning.

    In this tutorial you will learn how we can use `pathpyG` to represent static graphs. We start with basic features to create directed and undirected graphs with node-, edge-, and graph-level attributes. We also show how we can read and write graph data and how we can implement graph algorithms that are based on a traversal of nodes and edges.

    We first import the modules `torch`, `torch_geometric` and `pathpyG`.
    """)
    return


@app.cell
def _():
    import os
    import tempfile

    import pandas as pd
    import torch
    from torch_geometric.data import Data

    import pathpyG as pp

    return Data, os, pd, pp, tempfile, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Graph objects

    Let's start by generating a simple, directed graph with three nodes `a`, `b`, `c` and three edges `(a,b)`, `(b,c)` and `(a,b)`. The three nodes `a`, `b`, and `c` can be represented by integer indices $0, 1$ and $2$ respectively. Following the tensor-based representation in `pyG`, we use an `edge_index` tensor with shape `(2,m)` to represent the `m` edges of a graph.
    We can then add this to a `Data` object that can hold additional node and edge attributes. We finally pass the `Data` object to the constructor of the `Graph` class.

    Using the mapping of node names to indices specified above, the following code generates a directed `Graph` with three edges `(a,c)`, `(b,c)` and `(a,b)`.
    """)
    return


@app.cell
def _(Data, pp, torch):
    d = Data(edge_index = torch.tensor([[0,1,0], [2,2,1]]))
    g = pp.Graph(d)
    print(g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we do not need additional node or edge attributes, we can use the class function `Graph.from_edge_index` to directly create a graph based on an edge index:
    """)
    return


@app.cell
def _(pp, torch):
    g_1 = pp.Graph.from_edge_index(torch.tensor([[0, 1, 0], [2, 2, 1]]))
    print(g_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We may want to inlude isolated nodes that do not have an edge. We can do so by passing a `num_nodes` parameter. The following graph thus contains a fourth node (which we could name as `d`) that is not connected to any of the other nodes.
    """)
    return


@app.cell
def _(pp, torch):
    g_2 = pp.Graph.from_edge_index(torch.tensor([[0, 1, 0], [2, 2, 1]]), num_nodes=4)
    print(g_2)
    return (g_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In both cases, the `Graph` instance has a property `g.data` that stores a `pyG` `Data` object that includes the edge index as well as any further node-, edge- or graph-level attributes.
    """)
    return


@app.cell
def _(g_2):
    print(g_2.data)
    return


@app.cell
def _(g_2):
    print(g_2.data.edge_index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the `edge_index` is actually of type `pyG.EdgeIndex`, which is a subclass of `torch.Tensor`. Any tensor passed as an edge index in the constructor of `Graph` will automatically be converted to an `EdgeIndex` instance, as this internally allows us to provide efficient edge traveral routines based on sparse matrix operations. To support this, the edge index will be automatically sorted by row when the `Graph` object is created. To avoid this additional sort operation, you can pass an already sorted `EdgeIndex` object in the `Data` object in the constructor or using the `from_edge_index` class function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the generators `nodes` and `edges` to iterate through the nodes and edges of a graph as follows:
    """)
    return


@app.cell
def _(g_2):
    for _v in g_2.nodes:
        print(_v)
    for _e in g_2.edges:
        print(_e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the index-based representation of nodes allows for efficient tensor-based operations, it is often convenient to use string identifiers to refer to nodes. To simplify the handling of graphs with such node identifiers, `pathpyG` provides a class `IndexMap` that transparently maps string identifiers to integer indices. For our small example graph, we can create an `IndexMap` that associates node indices with string IDs. For our example, we can create a mapping as follows:
    """)
    return


@app.cell
def _(pp):
    m = pp.IndexMap(['a', 'b', 'c', 'd'])
    print(m)
    return (m,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the functions `IndexMap.to_id` or `IndexMap.to_idx` to map a node to an index or an ID:
    """)
    return


@app.cell
def _(m):
    print(m.to_id(0))
    return


@app.cell
def _(m):
    print(m.to_idx('b'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pathpyG` can apply this mapping transparently for the user. For this, we can add a mapping to a `Graph` object, either by passing it in the constructor or by setting the `mapping` attribute of an existing `Graph` instance.
    """)
    return


@app.cell
def _(g_2, m):
    g_2.mapping = m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we now iterate through the nodes and edges of the graph, we get:
    """)
    return


@app.cell
def _(g_2):
    for _v in g_2.nodes:
        print(_v)
    for _e in g_2.edges:
        print(_e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also pass an `IndexMap` object to the constructor of the `Graph` class. This transparently applies the mapping in all future operations on this graph instance.
    """)
    return


@app.cell
def _(m, pp, torch):
    g_3 = pp.Graph.from_edge_index(torch.tensor([[0, 1, 0], [2, 2, 1]]), num_nodes=4, mapping=m)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Above, we have created a graph based on an edge index tensor and we then additionally applied a mapping that we manually defined. We often have data in the form on an edge list, where edges are given as tuples of non-numeric node identifiers. The class function `Graph.from_edge_list` simplifies the construction of a `Graph` from such edge lists. It automatically creates an internal integer-based representation of the edge index along with the associated `IndexMap`, where integer node indices are based on the lexicographic order of node IDs.
    """)
    return


@app.cell
def _(pp):
    g_4 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c')])
    print(g_4)
    print(g_4.data.edge_index)
    print(g_4.mapping)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also pass a custom index mapping, e.g. mapping node `c` to idex 1 and node `b` to index 2 (thus deviating from a lexicographic order):
    """)
    return


@app.cell
def _(pp):
    g_5 = pp.Graph.from_edge_list([('a', 'b'), ('a', 'c'), ('b', 'c')], mapping=pp.IndexMap(['a', 'c', 'b']))
    print(g_5.data.edge_index)
    print(g_5.mapping)
    return (g_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Traversing Graphs

    The `Graph` object provides `get_successors` and `get_predecessors` functions, which return the indices of nodes that are connected to a node with a given index. Based on cached CSR (compressed sparse row) and CSC (compressed sparse column) representations cached for the sorted `EdgeIndex`, access to the successors and predecessors of a node works in constant time, i.e. it does not require to enumerate the `edge_index` tensor.

    For node `a` with index $0$ in our directed network we obtain:
    """)
    return


@app.cell
def _(g_5):
    g_5.get_successors(0)
    return


@app.cell
def _(g_5):
    g_5.get_predecessors(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that, even if a mapping is defined, the `get_successors` and `get_predecessors` functions always return a tensor with node indices, rather than node IDs. This is useful to support fast tensor-based operations on the list of successors and predecessors. We can however manually map node indices using the `IndexMap` object stored in the `mapping` attribute.

    If we instead want to traverse graphs based on string node IDs, we can use the `successors` and `predecessors` generators of the `Graph` object, which -- if a mapping is defined - yield the string IDs of successor or predecessor nodes for a given node (also identified by its string identifier).
    """)
    return


@app.cell
def _(g_5):
    for _v in g_5.successors('a'):
        print(_v)
    return


@app.cell
def _(g_5):
    for _v in g_5.predecessors('c'):
        print(_v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To check (in constant time) whether an edge exists in the graph, we can call the `is_edge` function:
    """)
    return


@app.cell
def _(g_5):
    g_5.is_edge('a', 'b')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, we can use the following function to check (in constant time) whether node `b` is a successor of `a`
    """)
    return


@app.cell
def _(g_5):
    'b' in g_5.successors('a')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, graph objects in `pathpyG` are directed, i.e. for the graph above, the edge `(b,a)` does not exist, which we can verify as follows:
    """)
    return


@app.cell
def _(g_5):
    print('a' in g_5.successors('b'))
    print(g_5.is_edge('b', 'a'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate (directed) in- and out-degrees of nodes, we can use the properties `in_degrees` and `out_degrees`, which return a dictionary that maps node IDs to their degrees:
    """)
    return


@app.cell
def _(g_5):
    for _v in g_5.nodes:
        print(f'{_v} -> {g_5.in_degrees[_v]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `in_degree` and `out_degree` properties are shortcuts to a general `degree` function that can be used to calculate (weighted) in- and outdegrees.
    """)
    return


@app.cell
def _(g_5):
    g_5.degrees(mode='in')
    return


@app.cell
def _(g_5):
    g_5.degrees(mode='out')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Degrees can be alternatively returned as torch.tensors.
    """)
    return


@app.cell
def _(g_5):
    g_5.degrees(mode='in', return_tensor=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also use arbitrary numerical edge attributes that will be used for a weighted (in- or out) degree calculation.
    """)
    return


@app.cell
def _(g_5, torch):
    g_5.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
    return


@app.cell
def _(g_5):
    g_5.degrees(mode='in', edge_attr='edge_weight', return_tensor=True)
    return


@app.cell
def _(g_5):
    g_5.degrees(mode='out', edge_attr='edge_weight', return_tensor=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Importantly, irrespective of how we have generated the graph object, the actual node and edge data are always stored as a `pyG` data object. This allows us to use the full power of `torch` and `pyG`, including the application of transforms, splits, or any easy migration between CPU and GPU-based computation.
    """)
    return


@app.cell
def _(g_5):
    g_5.data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In general, `pathpyG` handles device placement (i.e. if a tensor should be placed on CPU or GPU memory) similar to `pytorch`. By default, all tensors are created on the CPU, as we can see below:
    """)
    return


@app.cell
def _(g_5):
    g_5.data.is_cuda
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we instead want to create a graph on the GPU, we can specify the device during graph creation.
    """)
    return


@app.cell
def _(pp, torch):
    if torch.cuda.is_available():
        g_6 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c')], device='cuda')
        _out = g_6.data.is_cuda
    else:
        g_6 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c')])
        _out = 'CUDA not available'
    _out
    return (g_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can move a graph that is stored on the GPU back to the CPU using the familiar `to` function:
    """)
    return


@app.cell
def _(g_6):
    g_7 = g_6.to('cpu')
    return (g_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node-, Edge- or Graph-Level Attributes

    Real-world graphs often have node-, edge-, or graph-level attributes. In `pathpyG`, we can add attributes as tensors, either by directly assigning them to the `pyG` data object of an existing graph (or by adding them to the `Data` object passed to the constructor). Following the `pyG` semantics of attribute names, we use the prefixes `node_` and `edge_` to refer to node- and edge-level attributes. Attributes without those prefixes are assumed to refer to graph-level attributes.
    """)
    return


@app.cell
def _(g_7, torch):
    g_7.data['node_class'] = torch.tensor([[0], [0], [1]])
    g_7.data['edge_weight'] = torch.tensor([[1], [2], [3]])
    g_7.data['feature'] = torch.tensor([3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we have added attributes to nodes, edges, or the graph, those attributes, along with their type and shape will be shown when you print a string representation of the graph object:
    """)
    return


@app.cell
def _(g_7):
    print(g_7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To simplify access to attribute values, the `Graph` class provides getter and setter functions that allow to access attribute values based on node identifiers. To access the feature `node_feature` of node `a`, we can write:
    """)
    return


@app.cell
def _(g_7):
    g_7['node_class', 'a']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To access the weight of edge `(a, b)` we can write:
    """)
    return


@app.cell
def _(g_7):
    g_7['edge_weight', 'a', 'b']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, graph-based attributes can accessed as follows:
    """)
    return


@app.cell
def _(g_7):
    g_7['feature']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also use the setter functions to change attributes:
    """)
    return


@app.cell
def _(g_7, torch):
    g_7['node_class'] = torch.tensor([[7], [2], [3]])
    return


@app.cell
def _(g_7):
    g_7['node_class', 'a']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create sparse adjacency matrix representations of graphs, we can use the following function:
    """)
    return


@app.cell
def _(g_7):
    print(g_7.sparse_adj_matrix())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This returns a `scipy.sparse.coo_matrix` object, which can be turned into a dense `numpy` matrix as follows:
    """)
    return


@app.cell
def _(g_7):
    print(g_7.sparse_adj_matrix().todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By passing the name of the attribute, we can use edge attributes in the creation of the adjacency matrix. To create a sparse, weighted adjacency matrix that uses the `edge_weight` attribute of our graph object we can simply write:
    """)
    return


@app.cell
def _(g_7):
    print(g_7.sparse_adj_matrix(edge_attr='edge_weight').todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, graphs in `pathpyG` are directed. To represent undirected edges, we must add edges in both directions. We can use the `to_undirected()` function to make a directed graph undirected, which adds all (missing) edges that point in the opposite direction. This will also automatically duplicate and assign the corresponding edge attributes to the newly formed (directed) edges, i.e. edges are assumed to have the same attributes in both directions.
    """)
    return


@app.cell
def _(g_7):
    g_u = g_7.to_undirected()
    print(g_u)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, the `Graph` object can contain multiple identical edges, so the following is possible:
    """)
    return


@app.cell
def _(pp):
    g_8 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('a', 'b')])
    print(g_8.data.edge_index)
    return (g_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is often convenient, to coalesce multi-edges into weighted single-edges, i.e. in the example above we may prefer a graph where each edge occurs once in the edge index, but the edge `a->b` has a weight attribute of two, while the two other edges have one.

    In `pathpyG` we can do this by turning a graph into a weighted graph, which will coalesce edges and add an edge weight attribute that counts multi-edges in the original istance.
    """)
    return


@app.cell
def _(g_8):
    g_w = g_8.to_weighted_graph()
    print(g_w.data.edge_index)
    print(g_w['edge_weight', 'a', 'b'])
    print(g_w['edge_weight', 'b', 'c'])
    print(g_w['edge_weight', 'c', 'a'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we will see in a separate notebook focusing on the advanced (temporal) graph visualization features of `pathpyG`, it is easy to generate (interactive) HTML plots of graphs, that are embedded into jupyter notebooks. You can simply call the `pp.plot` function on the Graph object:
    """)
    return


@app.cell
def _(g_8, pp):
    pp.plot(g_8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading and writing graph data

    `pathpyG` supports reading and writing graph data in various formats using `pandas.DataFrame`s as an interface. For example, we can create a graph from an edge list and then transform it to a `DataFrame` that holds the edge list:
    """)
    return


@app.cell
def _(pp):
    g_9 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a')])
    print(g_9)
    _df = pp.io.graph_to_df(g_9)
    print(_df)
    return (g_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All edge attributes are preserved in this operation, as provided in the following example:
    """)
    return


@app.cell
def _(g_9, pp, torch):
    g_9.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
    print(g_9)
    _df = pp.io.graph_to_df(g_9)
    print(_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the `pp.io.graph_to_df` function only includes `edge`-level data. To also include `node`-level attributes, you need to create a separate `DataFrame` for those attributes.
    """)
    return


@app.cell
def _(pd):
    node_attr = pd.DataFrame({'v': ['b', 'a', 'c'], 'node_size': [5.0, 2.0, 1.0]})
    print(node_attr)
    return (node_attr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can add node attributes from a `DataFrame` to a graph using the `add_node_attributes` function:
    """)
    return


@app.cell
def _(g_9, node_attr, pp):
    pp.io.add_node_attributes(node_attr, g_9)
    print(g_9.data.node_size)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, you can also add additional edge attributes from a `DataFrame` using the `add_edge_attributes` function:
    """)
    return


@app.cell
def _(g_9, pd, pp):
    edge_attr = pd.DataFrame({'v': ['c', 'a', 'b'], 'w': ['a', 'b', 'c'], 'edge_weight': [2.0, 3.0, 5.0]})
    print(edge_attr)
    pp.io.add_edge_attributes(edge_attr, g_9)
    print(g_9.data.edge_index)
    print(g_9.data.edge_weight)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you do not want to use the `uid`s of the nodes, you can also specify the parameter `node_indices=True`, which will use the integer indices of nodes instead.
    """)
    return


@app.cell
def _(g_9, pp):
    _df = pp.io.graph_to_df(g_9, node_indices=True)
    print(_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading and writing from and to `.csv` files

    As mentioned above, we can read and write graph data into various common formats. A common format to store edge lists is the `.csv` format. We can easily read a graph from a `.csv` file using the `pp.io.read_csv_graph` function and write a graph to a `.csv` file using the `pp.io.write_csv` function.
    """)
    return


@app.cell
def _(g_9, os, pp, tempfile):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        tmp_file = os.path.join(tmpdirname, 'test_graph.csv')
        pp.io.write_csv(g_9, path_or_buf=tmp_file)
        csv_g = pp.io.read_csv_graph(filename=tmp_file)
    pp.plot(csv_g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="admonition tip">
        <p class="admonition-title">Hint</p>

            Note that the edges in the above visualisation have varying thickness. This corresponds to the edge weights in the graph.


    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `networkx` Delegate Mechanism

    To calculate node centralities, we can use a `networkx` delegate mechanism implemented in the module `pathpyG.algorithms.centrality`. Simply speaking, you can call any function implented in the `networkx.centrality` module whose name ends with `_centrality`. The `pathpyG` graph object will be internally converted to a `networkx.DiGraph` object, the corresponding centrality function (with all of its parameters) will be called, and the result will be mapped back to nodes based on node IDs.

    In order to calculate the closeness centralities of all nodes for the graph above, we can call:
    """)
    return


@app.cell
def _(g_9, pp):
    pp.algorithms.centrality.closeness_centrality(g_9)
    return


@app.cell
def _(g_9, pp):
    pp.algorithms.centrality.eigenvector_centrality(g_9)
    return


@app.cell
def _(g_9, pp):
    pp.algorithms.centrality.katz_centrality(g_9)
    return


if __name__ == "__main__":
    app.run()
