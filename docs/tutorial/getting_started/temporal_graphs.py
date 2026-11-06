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
    # Temporal Graph Analysis

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

    In this tutorial we will introduce the representation of temporal graph data using the `TemporalGraph` class. We show how we can create temporal graphs, restrict them to time windows, aggregate them to static graphs, and read and write temporal graph data.
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
    We can create a temporal graph object from a list of time-stamped edges. Since `TemporalGraph` is a subclass of the `Graph` class, the internal structures are very similar:
    """)
    return


@app.cell
def _(pp):
    _tedges = [('a', 'b', 1), ('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4), ('c', 'd', 5), ('b', 'a', 5), ('c', 'b', 6)]
    t = pp.TemporalGraph.from_edge_list(_tedges)
    print(t.mapping)
    print(t.n)
    print(t.m)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By default, all temporal graphs are directed. We can create an undirected version a temporal graph as follows:
    """)
    return


@app.cell
def _(t):
    x = t.to_undirected()
    print(x.mapping)
    print(x.n)
    print(x.m)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also directly create a temporal graph from a `pyG` `Data` object that has an `edge_index` and a `time` tensor
    """)
    return


@app.cell
def _(Data, pp, torch):
    td = Data(edge_index=torch.Tensor([[0, 1, 2, 0], [1, 2, 3, 1]]).long(), time=torch.Tensor([0, 1, 2, 3]), num_nodes=4)
    print(td)
    t2 = pp.TemporalGraph(td)
    print(t2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can restrict a temporal graph to a time window, which returns a temporal graph that only contains time-stamped edges in the given time interval.
    """)
    return


@app.cell
def _(t):
    _t1 = t.get_window(0, 4)
    print(_t1)
    print(_t1.m)
    print(_t1.start_time)
    print(_t1.end_time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also extract a TemporalGraph object for a batch of temporal edges, which is defined by the start and end index of the edges defining the batch.
    """)
    return


@app.cell
def _(t):
    _t1 = t.get_batch(1, 6)
    print(_t1)
    print(_t1.m)
    print(_t1.start_time)
    print(_t1.end_time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can easily convert a temporal graph into a weighted time-aggregated static graph, where edge weights count the number of occurrences of an edge across all timestamps.
    """)
    return


@app.cell
def _(t):
    _g = t.to_static_graph(weighted=True)
    print(_g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also aggregate a temporal graph within a certain time window:
    """)
    return


@app.cell
def _(t):
    _g = t.to_static_graph(time_window=(1, 3), weighted=True)
    print(_g)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can use the class `RollingTimeWindow` to perform a rolling window analysis. The class returns an iterable object, where each iteration yields a time-aggregated weighted graph object as well as the corresponding time window.
    """)
    return


@app.cell
def _(pp, t):
    r = pp.algorithms.RollingTimeWindow(t, window_size=3, step_size=1, return_window=True)
    for _g, _w in r:
        print('Time window ', _w)
        print(_g)
        print(_g.data.edge_index)
        print('---')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can visualize temporal graphs using the plot function just like static graphs:
    """)
    return


@app.cell
def _(pp, t):
    pp.plot(t);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The source nodes, destination nodes and timestamps of time-stamped edges are stored in a `pyG` `Data` object with an `edge_index` and a `time` tensor, which we can access in the following way.
    """)
    return


@app.cell
def _(t):
    t.data
    return


@app.cell
def _(t):
    print(t.data.edge_index)
    return


@app.cell
def _(t):
    print(t.data.time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the generator functions `edges` and `temporal_edges` we can iterate through the time-ordered (temporal) multi-edges of a temporal graph.
    """)
    return


@app.cell
def _(t):
    for _v, _w in t.edges:
        print(_v, _w)
    return


@app.cell
def _(t):
    for _v, _w, time in t.temporal_edges:
        print(_v, _w, time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading and writing temporal graph data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similar to simple graphs, we can read and write temporal graph data into various common formats. For example, we can easily convert our temporal graph to a Pandas DataFrame using the `temporal_graph_to_df` function and create a temporal graph from a DataFrame using the `df_to_temporal_graph` function:
    """)
    return


@app.cell
def _(pp):
    _tedges = [('a', 'b', 1), ('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4), ('c', 'd', 5), ('b', 'a', 5), ('c', 'b', 6)]
    t_1 = pp.TemporalGraph.from_edge_list(_tedges)
    df = pp.io.temporal_graph_to_df(t_1)
    print(df)
    return (df,)


@app.cell
def _(df, pp):
    t_2 = pp.io.df_to_temporal_graph(df)
    print(t_2)
    return (t_2,)


@app.cell
def _(os, pp, t_2, tempfile):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        tmp_file = os.path.join(tmpdirname, 'test_graph.csv')
        pp.io.write_csv(t_2, path_or_buf=tmp_file)
        csv_t = pp.io.read_csv_temporal_graph(filename=tmp_file)
    pp.plot(csv_t);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the columns of a `DataFrame` are not named, `df_to_temporal_graph` assumes that the first three columns contain the source node, target node, and timestamp of each time-stamped edge:
    """)
    return


@app.cell
def _(pd, pp):
    df_1 = pd.DataFrame([['a', 'b', 1], ['b', 'c', 2], ['a', 'c', 3]])
    print(df_1)
    t_3 = pp.io.df_to_temporal_graph(df_1)
    print(t_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The real power of temporal graphs comes into play when we consider **time-respecting paths**, i.e. paths that follow the temporal ordering of time-stamped edges. We introduce time-respecting paths, as well as temporal centralities and communities that build on them, in the tutorial [Time-Respecting Paths, Centralities and Communities](../../advanced/trp_centralities_communities/).
    """)
    return


if __name__ == "__main__":
    app.run()
