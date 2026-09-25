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
    # Path Data and Higher-Order De Bruijn Graphs

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
    ## Motivation and Learning Objective

    While `pathpyG` is useful to handle and visualize static graphs - as the name suggests - its main advantage is that it facilitates the analysis of time series data that can be used to calculate **paths** in a graph. As we shall see in the following tutorial, there are various situations in which we naturally have access to data on paths, including data on (random) walks or trajectories, traces of dynamical processes giving rise to node sequences or directed acyclic graphs, or time-respecting paths in temporal graphs. `pathpyG` can be used to model patterns in such data based on higher-order De Bruijn graph models.

    In this first unit, we will show how `pathpyG` supports to represent data on paths in graphs. Like graphs, such data are internally stored as tensors, which facilitates GPU-based operations to create higher-order De Bruijn graphs.

    We first import the modules `torch` and `pathpyG`. By setting the device used by `torch`, we can specify whether we want to run our code on the CPU or on the GPU.
    """)
    return


@app.cell
def _():
    import torch

    import pathpyG as pp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device, pp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the following examples, we consider a simple directed graph with five nodes `a`, `b`, `c`, `d`, `e` and four edges:
    """)
    return


@app.cell
def _(pp):
    g = pp.Graph.from_edge_list([('a', 'c'),
                                 ('b', 'c'),
                                 ('c', 'd'),
                                 ('c', 'e')])
    pp.plot(g);
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using `PathData` to store walks or paths in a graph

    Assume that we have time series data that captures observations of trajectories (i.e. walks or paths) in the graph above. For example, we could observe four walks of length two, four each of the following:

    - 4 x `a` -> `c` -> `d`
    - 4 x `b` -> `c` -> `e`

    Note that we define the length of a walk or path as the number of edges that are traversed, i.e. a sequence that consists of a single node, e.g. `a`, is considered a walk of length zero, while every edge in a graph is a walk of length one.

    `pp.PathData` supports to store and model such sequential data. We first create an instance of the `PathData` class. To consistently map node IDs to indices across `Graph` and `PathData` objects, we can pass the `IndexMap` object from the `Graph` above in the constructor. We then use the `append_walk` function to add observations of our two walks, where the `weight` argument is used to indicate the number of times each path or walk has been observed.
    """)
    return


@app.cell
def _(g, pp):
    paths = pp.PathData(g.mapping)

    paths.append_walk(('a', 'c', 'd'), weight=4.0)
    paths.append_walk(('b', 'c', 'e'), weight=4.0)
    print(paths)
    return (paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us inspect how those walks are internally stored in the `PathData` object. We find that the class internally stores a `pyG.Data` object, which contains the properties `edge_index`, `node_sequence`, `dag_weight`, `dag_num_edges` and `dag_num_nodes` that can be used to access all of the individual paths.
    """)
    return


@app.cell
def _(paths):
    paths.data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `edge_index` tensor represents an ordered sequence of edges traversed by the walk, where the indices of nodes map to the `node_sequence` tensor. This additional mapping is neccessary since walks can traverse the same edge multiple times. Moreover, it allows to internally concatenate multiple walks into a single `Data` object, which is needed for fast GPU-based operations on path data. We can access the first path as follows:
    """)
    return


@app.cell
def _(paths):
    paths.data.edge_index[:, :paths.data.dag_num_edges[0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `node_sequence` tensor tells us that the node with index `1` in the `edge_index` maps to the node in the graph with index `2`, which is node `c`.
    """)
    return


@app.cell
def _(paths):
    paths.data.node_sequence[:paths.data.dag_num_nodes[0]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can index the second edge index accordingly and can see that the `node_sequence` tensor maps to the sequence `b -> c -> e`:
    """)
    return


@app.cell
def _(paths):
    paths.data.node_sequence[paths.data.edge_index[:, paths.data.dag_num_edges[0]:]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can actually see a collection of walks as a higher-order generalization of the usual way to define graphs as a collection of dyadic edges (which are simply walks of length one). From this point of view, a standard static (weighted) graph is simply a first-order model of node sequences, which only considers the frequency at which edges are traversed.

    To generate such a first-order model, we can use the class `MultiOrderModel` and use the first-layer of the model, which is simply a weighted static graph where edge weights count the number of times each edge is traversed by a path. We will explain the class `MultiOrderModel`, which generalizes this concept to higher-order graph models for any order $k$ in a moment. For now, we can just use it to generate a first-order weighted graph as follows.

    The generated graph is again based on a `pyG.Data` object that contains an edge_index and edge weights. As we can see, for the example above the edge_index is just a concatenation of the edge indices of individual walks, where the node indices have been mapped to the correct nodes.
    """)
    return


@app.cell
def _(paths, pp):
    _m = pp.MultiOrderModel.from_path_data(paths, max_order=1)
    g_1 = _m.layers[1]
    print(g_1.data.edge_index)
    print(g_1.data.edge_weight)
    pp.plot(g_1)
    return (g_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why are data on paths and walks interesting in the first place? The answer is that they provide information on the **causal topology of complex systems**, i.e. which nodes can possibly causally influence each other via paths that follow the arrow of time. This information is lost if we were to split paths into an (unordered) collection of dyadic interactions between pairs of nodes, i.e. if we were to only onsider links.

    To illustrate this, let us assume that the four walks above tell us which paths information (or whatever you may be interested in) can take in the simple graph above. That is, we observe something moving from `a` via `c` to `d` and from `b` via `c` to `e`, and each of those events occur four times. However, we never observed that something moving from `a` to `c` ended up in `d`. And neither did we observe that something moving from `b` to `c` ended up in `e`. This means that - assuming that we completely observed all walks or paths - there is no way that `a` can causally influence `e` or that `b` could causally influence `d` via the center node `c`. Note that this is not what we would assume if we consider possible paths in the topology of the underlying graph, where paths of length two exist between all four pairs of nodes (`a`, `d`), (`a`, `e`), (`b`, `d`), (`b`, `e`).

    Hence, we can use data capturing actually observed paths or walks ion a network in contrast to which paths or walks would theoretically be possible based on the topology.

    As a contrast, consider the following observations of walks in the same graph.
    """)
    return


@app.cell
def _(g_1, pp):
    paths_2 = pp.PathData(g_1.mapping)
    paths_2.append_walk(('a', 'c', 'd'), weight=2)
    paths_2.append_walk(('a', 'c', 'e'), weight=2)
    paths_2.append_walk(('b', 'c', 'd'), weight=2)
    paths_2.append_walk(('b', 'c', 'e'), weight=2)
    print(paths_2)
    return (paths_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we have observed walks along all four possible paths of length two, each walk occurring only two times. Like in the example before, each edge was traversed exactly four times and thus the weighted edge index of a first-order graph model is identical to the one before:
    """)
    return


@app.cell
def _(paths_2, pp):
    _m = pp.MultiOrderModel.from_path_data(paths_2, max_order=1)
    g_2 = _m.layers[1]
    print(g_2.data.edge_index)
    print(g_2.data.edge_weight)
    pp.plot(g_2)
    return (g_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a first-order graph representation, as it only captures the (weighted) edges in the underlying path data, i.e. we could say that we only count the frequency of paths (or walks) of length one. This naturally gives rise to an `edge_index` tensor with shape $(2,m)$, where $m$ is the number of unique edges in the graph that are traversed by the paths.

    ## From Graphs to Higher-Order De Bruijn Graph Models

    As we have seen above, the use of a first-order graph model discards information in path data, which capture which nodes can possibly causally influence each other via paths. A key feature of `pathpyG` is it allows to generalize this first-order modelling perspective to $k$-th order De Bruijn graph models for paths, where the nodes in a $k$-th order De Bruijn graph model are sequences of $k$ nodes. Edges connect pairs of nodes that overlap in $k-1$ nodes and capture paths of length $k$.

    A De Bruijn graph of order $k=1$ is simply a normal (weighted) static graph consisting of nodes and edges. Pairs of nodes connected by edges overlap in $k-1=0$ nodes and capture paths of length $k=1$, i.e. simple dyadic edges in the underlying path data.

    For a De Bruijn graph with order $k=2$, in our example above, an edge connects a pair of nodes $(a,b)$ and $(b,c)$ that overlaps in the $k-1=1$ node $b$. Such an edge represents the path $a -> b -> c$ of length two. We can use the `MultiOderModel` class to generate a second-order De Bruijn graph representation of the path data above. We just have to set the `max_order` parameter to two and use the second layer of the resulting `MultiOrderModel` instance.
    """)
    return


@app.cell
def _(paths, pp):
    _m = pp.MultiOrderModel.from_path_data(paths, max_order=2)
    pp.plot(_m.layers[2], edge_size=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $k=2$, we obtain a second-order De Bruijn graph where second-order nodes are first-order edges and second-order edges represent walks of length two in the original graph. Edge weights capture observation frequencies of those walks. In our example, we have two different walks of length two ($a$ -> $c$ -> $d$ and $b$ -> $c$ -> $e$), represented by two edges $(a-c, c-d)$ and $(b-c, c-e)$. Each of those walks appears four times so the weights of both edges are four.
    """)
    return


@app.cell
def _(g_2):
    print(g_2.mapping)
    print(g_2.data.edge_index)
    print(g_2.data.edge_weight)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While this goes beyond the scope of this tutorial, thanks to the tensor-based representation of paths, the construction ofhigher-order De Bruijn graphs can be done based on efficient GPU operations, i.e. we can scale it up to large graphs.

    Let us have a closer look at our examples above. While the first-order edge indices of the two path objects `paths` and `paths_2` are the same, we find that the second-order edge indices are actually different. For `paths_2` we have four different paths of length two, each occurring twice. Hence, our second-order De Bruijn graph has four edges, each with weight two. These edges correspond to all possible paths of length two in the underlying graph.
    """)
    return


@app.cell
def _(paths_2, pp):
    _m = pp.MultiOrderModel.from_path_data(paths_2, max_order=2)
    pp.plot(_m.layers[2])
    return


@app.cell
def _(g_2):
    print(g_2.mapping)
    print(g_2.data.edge_index)
    print(g_2.data.edge_weight)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We thus find that the second-order De Bruijn graph representation of paths is sensitive to the differences in the causal topology, while a first-order graph is not. This is the basis to generalize network analysis and graph learning to causality-aware graph models for various kinds of time series data on graphs. In particular, as we shall see in more detail in a later tutorial, we can use paths to generate k-th order graphs that can be used to generalize Graph Neural Networks to higher-order De Bruijn Graphs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that all higher-order graphs are simply `Graph` objects, which means that we can iterate through the nodes of a higher-order graph just like for normal graphs. Node indices are automatically mapped, yielding tuples of first-order node identifiers.
    """)
    return


@app.cell
def _(g_2):
    for n in g_2.nodes:
        print(n)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Edges are tuples with two elements, where each element is a k-th order node, i.e. a tuple of node IDs of length $k$. I.e. for a second-order model the edges are tuples of length two, each entry containing s tuple of length two.
    """)
    return


@app.cell
def _(g_2):
    for _e in g_2.edges:
        print(_e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weight attribute stores a tensor whose entries capture the frequencies of edges, i.e. the frequencies of paths of length $k$.
    """)
    return


@app.cell
def _(g_2):
    for _e in g_2.edges:
        print(_e, g_2['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can finally plot a higher-order De Bruijn graph in the same way as a first-order graph.
    """)
    return


@app.cell
def _(g_2, pp):
    pp.plot(g_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us compare this to a second-order graph model of the second path data set `paths_2` from above, which corresponds to a network where all possible paths of length two actually occur. Hence, different from the data in `paths`, all pairs of nodes in this graph can causally influence each other via paths of length two.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detecting the Optimal Order of Higher-Order De Bruijn Graph Models

    The fact that we can model the same set of paths with higher-order De Bruijn graph models with different orders $k$ raises an important question: What is the **optimal order** to model a given `PathData` instance. It is actually easy to answer this question in our example above.

    For the data contained in `paths`, we observe only two of the four possible paths, which is different from what we would expect based on a first-order graph model. To capture this pattern in the node sequenves, a first-order graph model is not sufficient and we need a second-order De Bruoijn graph model.

    For `paths_2` this is different: Here we observe all four paths of length two with the same frequency, which is exactly what we would expect based on the first-order weighted graph, where all edge weights are the same. Hence, for `paths_2` the second-order De Bruijn graph model contains no additional information compared to a first-order weighted graph, which means a first-order model is sufficient.

    While it is easy to see this in the toy example, for real data we need a principled method to automatically determine the optimal order of a higher-order De Bruijn graph model. Luckily, this can be achieved based on statistical model selection in a multi-order De Bruijn graph model. Here we cannot explain the details of this method, so we kindly refer you to the following paper:

    [I Scholtes: When is a Network a Network?: Multi-Order Graphical Model Selection in Pathways and Temporal Networks, In Proc. of SIGKDD 2017, August 2017](https://dl.acm.org/doi/10.1145/3097983.3098145)

    The method introduced in this paper is implemented in `pathpyG`. To determine the optimal order of a higher-order De Bruijn graph model for the node sequences contained in a given `PathData` instance, we can use the `MultiOderModel.estimate_order` method. Since the method is based on statistical hypothesis testing, we can also pass a significance threshold, which - in line with the interpretation of p-values - bounds the type I error rate of our test, i.e. the rate at which we wrongly reject the null hypothesis that the true optimal order of a data set is $k-1$ in favor of the alternative hypothesis that the order is $k$.

    Let us test this for our toy example. Using a significance threshold of $0.01$, we determine the optimal order for the data set `paths` that should actually warrant a second-order model:
    """)
    return


@app.cell
def _(paths, pp):
    m1 = pp.MultiOrderModel.from_path_data(paths, max_order=2)
    print(m1.estimate_order(paths, significance_threshold=0.01))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For `paths_2` where the observed paths are in line what we would expect based on the weighted edges in the first-order graph model, we correctly find that we do not need to consider a second-order model
    """)
    return


@app.cell
def _(paths_2, pp):
    m2 = pp.MultiOrderModel.from_path_data(paths_2, max_order=2)
    print(m2.estimate_order(paths_2, significance_threshold=0.01))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Admittedly, the situation in the `paths` data is extreme insofar as two of the possible paths of lengths two have not been observed at all. We can also have more subtle deviations from the expectation based on the first-order graph model. Consider the following case, where two of the paths are observed more often than two other paths. Note that we have assigned the path frequencies such that again all edges are traversed with exactly the same frequencies, i.e. all edge weights are again equal in the first-order graph.
    """)
    return


@app.cell
def _(pp):
    g_3 = pp.Graph.from_edge_list([('a', 'c'), ('b', 'c'), ('c', 'd'), ('c', 'e')])
    paths_3 = pp.PathData(g_3.mapping)
    paths_3.append_walk(('a', 'c', 'd'), weight=6.0)
    paths_3.append_walk(('a', 'c', 'e'), weight=2.0)
    paths_3.append_walk(('b', 'c', 'e'), weight=6.0)
    paths_3.append_walk(('b', 'c', 'd'), weight=2.0)
    m3 = pp.MultiOrderModel.from_path_data(paths_3, max_order=2)
    print(m3.layers[1].data.edge_weight)
    print(m3.estimate_order(paths_3, significance_threshold=0.01))
    return g_3, m3, paths_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, due to the relatively small number of observations, the deviations from the expected baseline are still not strong enough to detect the optimal order of two (at least not for a significance threshold of $0.01$). We would need to raise the signififance threshold to $0.15$ to detect order two:
    """)
    return


@app.cell
def _(m3, paths_3):
    print(m3.estimate_order(paths_3, significance_threshold=0.15))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, we are able to detect a significant deviation from a first-order model for a significance threshold of $0.01$ if we make the deviations more extreme. If we observe two fo the paths seven times, while the other two are only observed once we find that order two is significant at a sigificance threshold of $0.01$.
    """)
    return


@app.cell
def _(g_3, pp):
    paths_3_1 = pp.PathData(g_3.mapping)
    paths_3_1.append_walk(('a', 'c', 'd'), weight=7.0)
    paths_3_1.append_walk(('a', 'c', 'e'), weight=1.0)
    paths_3_1.append_walk(('b', 'c', 'e'), weight=7.0)
    paths_3_1.append_walk(('b', 'c', 'd'), weight=1.0)
    m3_1 = pp.MultiOrderModel.from_path_data(paths_3_1, max_order=2)
    print(m3_1.layers[1].data.edge_weight)
    print(m3_1.estimate_order(paths_3_1, significance_threshold=0.01))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ability to detect the optimal higher-order for a given data set in a statistically principled way is a powerful feature of the modelling framework of higher-order De Bruijn Graphs. Admittedly, the likelihood-based model selection approach has its limitations especially for small data sets or partially observed graphs, where it can both over- or underfit. To address this, we have developed an alternative Bayesian model selection technique that is explained and evaluated in the following paper:

    [L Petrovic, I Scholtes: Learning the Markov order of paths, In Proc. of the ACM Web Conference (WWW'22), April 2022](https://dl.acm.org/doi/10.1145/3485447.3512091)

    Unfortunately, this method has not yet been implemented in `pathpyG` but we are planning to add it soon.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading empirical path data from N-Gram Files

    For real data on walks in graphs it is not convenient to manually construct and add walks based on edge tensors. We can instead use the `pp.io.read_csv_path_data` function to load such data from an n-gram file, i.e. a text file where each line corresponds to one observed walk consisting of comma-separated node IDs. If we set the argument `weight=True`, the last component of each line is considered to be the observation frequency of that particular walk.

    As an example, the file `data/tube_paths_train.ngram` contains observed passenger itineraries between nodes in a graph that representes the network of London Tube stations. Each of those itineraries is associated with an observation frequencies. The following is an excerpt from that file:

    ```
    Southwark,Waterloo,212.0
    Liverpool Street,Bank / Monument,1271.0
    Barking,West Ham,283.0
    Tufnell Park,Kentish Town,103.0
    ...
    ```

    Note that this will automatically create an internal mapping of node IDs to indices.

    <div class="admonition note">
        <p class="admonition-title">Dataset Availability</p>

            Depending on how you are executing this notebook, the dataset may not be available locally. We use <code>pp.io.example_data</code> to resolve the file name: it returns the path of a local copy in the <code>docs/data</code> folder of the pathpyG repository, or - if you are running outside of a clone, e.g. in Google Colab - a URL pointing to the GitHub repository. Both can be passed directly to the <code>pp.io.read_csv_*</code> functions.


    </div>
    """)
    return


@app.cell
def _(pp):
    paths_tube = pp.io.read_csv_path_data(path_or_buf=pp.io.example_data('tube_paths_train.ngram'), sep=',', weight=True)
    print(paths_tube)
    return (paths_tube,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To plot a (first-order) graph representation of the London Tube metro network, we can use the following code:
    """)
    return


@app.cell
def _(paths_tube, pp):
    _m = pp.MultiOrderModel.from_path_data(paths_tube, max_order=1)
    g_4 = _m.layers[1]
    g_4.data.edge_weight = 5 * g_4.data.edge_weight / g_4.data.edge_weight.max()
    pp.plot(g_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In general, the maximum size of a higher-order model could grow exponentially with the number of nodes in the underlying graph. However, for most empirical data sets, higher-order models are actually sparse, which allows us to efficiently construct them using GPU-based operations. We demonstrate this in the London Tube data sets by constructing all higher-order De Bruijn graph models up to order 20, which takes approx. 25 seconds on a mobile GPU (RTX A2000) and yields reasonably sized higher-order models.
    """)
    return


@app.cell
def _(device, paths_tube, pp):
    paths_tube.to(device)
    _m = pp.MultiOrderModel.from_path_data(paths_tube, max_order=20)
    print(_m.layers[20])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we shall see in the following two units, the `MultiOrderGraph` class is also the basis for the GPU-based analysis and modelling of causal structures in temporal graphs. In particular, the underlying generalization of first-order static graph models to higher-order De Bruijn graphs allows us to easily build causality-aware graph neural network architectures that consider both the topology and the temoral ordering of time-stamped edges in a temporal graph. We will
    """)
    return


if __name__ == "__main__":
    app.run()
