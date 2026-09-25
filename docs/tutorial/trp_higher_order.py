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
    # Higher-Order Models for Time-Respecting Paths in Temporal Graphs

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

    In the previous tutorial, we have seen how we can use higher-order models to model paths in complex networks. In this example, paths were directly given in terms of sequences of nodes traversed by some process (like a random walk). We have further seen that higher-order De Bruijn graph models can be used to capture patterns that influence the **causal topology** of a complex network, i.e. which nodes can possibly influence each other via paths. The same is true for time-respecting paths in a temporal graph. Due to the fact that time-stamped edges need to occur in the correct temporal ordering (and within a given time interval based on the maximum time difference $\delta$), the causal topology given by time-respecting paths can be very different from what we would expect from the (static) topology of links.

    In the following, we will show how we can easiy and efficiently construct higher-order models for time-respecting paths in a temporal graph. To illustrate this, we use the same toy example as before:
    """)
    return


@app.cell
def _():
    import torch

    import pathpyG as pp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, pp


@app.cell
def _(pp):
    tedges = [('a', 'b', 1),('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4),
                  ('c', 'd', 5), ('b', 'a', 5), ('c', 'b', 6)]
    t = pp.TemporalGraph.from_edge_list(tedges)
    print(t)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To better understand this temporal graph, we can again create a directed acyclic graph that represents the topology of time-respecting paths:
    """)
    return


@app.cell
def _(pp, t):
    e_i = pp.core.event_graph.EventGraph.build_edge_index(t, delta=1)
    mapping = pp.IndexMap([f'{v}-{w}-{time}' for v, w, time in t.temporal_edges])
    dag = pp.Graph.from_edge_index(e_i, mapping=mapping)
    pp.plot(dag);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $\delta=1$, we again have the following time-respecting paths:

    Length one:
        a -> b
        b -> a
        b -> c
        c -> b
        c -> d
        d -> c
    Length two:
        a -> b -> a (twice)
        b -> a -> b
        a -> b -> c
        b -> c -> b
        c -> b -> a
        d -> c -> d
    Length three:
        a -> b -> a -> b
        b -> a -> b -> a
        a -> b -> c -> b
        b -> c -> b -> a
    Length four:
        a -> b -> a -> b -> a
        a -> b -> c -> b -> a
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, these time-respecting paths are actually very similar to the paths data that we have previously represented using the `PathData` object. In fact, we could - in theory - first extract all time-respecting paths of all lengths, add them to a `PathData` object and then use the `MultiOderModel` class to generate higher-order De Bruijn graph models of all orders. In the example above, since we have paths of length one to four, we could create higher-order models with orders from one to four.

    However, this approach would not be efficient for large temporal graphs, as it is computationally expensive to calculate all possible time-respecting paths as well as subpaths of length $k$, especially for larger values of $\delta$. To avoid this bottleneck, `pathpyG` uses a smarter, GPU-based algorithm to calculate time-respecting paths of length $k$ that are needed for a given order $k$.

    For the example above, we can generate all higher-order models up to order four as follows:
    """)
    return


@app.cell
def _(pp, t):
    m = pp.MultiOrderModel.from_temporal_graph(t, delta=1, max_order=4)

    print(m.layers[3])
    print(m.layers[4])
    return (m,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remember that in a $k$-th order model nodes capture paths of length $k-1$, while edges capture paths of length $k$.

    This implies that the first-order model has four nodes and six edges, which simply corresponds to the time-aggregated weighted graph for our example temporal network.
    """)
    return


@app.cell
def _(m, pp):
    print(m.layers[1])
    pp.plot(m.layers[1]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the second-order model, we have six nodes, which map to the six different edges (each edge trivially being a time-respecting path of length one) of the temporal graph. The six edges in the second-order model represent the six different time-respecting paths of length two (see above). Since the time-respecting path $a \rightarrow b \rightarrow a$  occurs twice at different times, we have one edge with weight two.
    """)
    return


@app.cell
def _(m, pp):
    print(m.layers[2])
    print(m.layers[2].data.edge_weight)
    pp.plot(m.layers[2]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the third-oder mode, we have four edges representing the four different time-respecting paths of length three in the temporal graph above:
    """)
    return


@app.cell
def _(m, pp):
    print(m.layers[3])
    pp.plot(m.layers[3]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, for the model with order $k=4$ we only have two edges, representing the two time-respecting paths $a \rightarrow b \rightarrow a \rightarrow b \rightarrow a$ and $a \rightarrow b \rightarrow c \rightarrow b \rightarrow a$:
    """)
    return


@app.cell
def _(m, pp):
    print(m.layers[4])
    pp.plot(m.layers[4]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Intuitively, since in our example there are no time-respecting paths longer than four, if we were to generate a multi-order model with De Bruijn graphs with orders larger than four, those graphs cannot contain any edges. We see this in the following example. The first-order graph is simply the time-aggregated weighted graph, i.e. the number of nodes is equal to the number of nodes in the temporal graph and the number of edges is equal to the number of different time-stamped edges. In each graph of order $k>1$, the number of nodes corresponds to the number of edges in the graph with order $k-1$, since each of those nodes corresponds to a time-respecting path of length $k-1$, which are represented by edges in a $k-1$-th order gaph. This implies that the graph with order five has two nodes, which are the two time-respecting paths of length four. Those nodes are not connected since there is no time-respecting path with length five.
    """)
    return


@app.cell
def _(pp, t):
    m_1 = pp.MultiOrderModel.from_temporal_graph(t, delta=1, max_order=5)
    print(m_1.layers[4])
    print(m_1.layers[5])
    pp.plot(m_1.layers[5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constructing Higher-Order De Bruijn Graph Models for Empirical Temporal Networks

    Let us now use `pathpyG` to construct higher-order De Bruijn graph models for time-respecting paths in empirical temporal network. For this, we first read a number of temporal graphs using the Netzschleuder interface. In the following, we use a publicly available data set:

    - Face-to-face interactions in a highschool [(Mastrandrea, Fournet, Barrat, 2015)](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/)
    """)
    return


@app.cell
def _(pp):
    t_sp = pp.io.read_netzschleuder_graph("sp_high_school", "proximity", time_attr="time")
    print(t_sp)
    return (t_sp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To generate a `MultiOrderModel` consisting of multiple layers of higher-order De Bruijn graph models, we can use the `MultiOrderModel.from_temporal_graph` method. We can further specify the maximum order of the highest-order layer, as well as the maximum time difference $\delta$ for time-respecting paths.

    We use a maximum time difference of 15 minutes. As you can see below, we can efficiently generate a 5-th order model despite using a temporal graph with more than 188,000 time-stamped edges and considering all time-respecting paths up to length five with a large maximum time difference. Thanks to the use of GPU-accelerated operations, creating such a model takes less than 12 seconds on an (old) RTX 2090 GPU.
    """)
    return


@app.cell
def _(device, pp, t_sp):
    m_2 = pp.MultiOrderModel.from_temporal_graph(t_sp.to(device), delta=900, max_order=5)
    return (m_2,)


@app.cell
def _(m_2):
    print(m_2.layers[1])
    print(m_2.layers[3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How can we use such higher-order graph models for graph learning tasks? We will demonstrate this in the next unit of our tutorial.
    """)
    return


if __name__ == "__main__":
    app.run()
