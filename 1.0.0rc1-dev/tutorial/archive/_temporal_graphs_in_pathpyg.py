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
    # Temporal Graphs in pathpyG

    *August 4 2026*
    *Training Workshop: Causality-Aware Temporal Networks*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we will introduce the representation of temporal graph data using the `TemporalGraph` class and how such data can be used to calculate shortest time respecting paths between nodes as well temporal node cemtralities.
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
    We can also directly create a temporal graph from an instance of `pyG.TemporalData`
    """)
    return


@app.cell
def _(Data, pp, torch):
    td = Data(
        edge_index = torch.Tensor([[0,1,2,0],[1,2,3,1]]).long(),
        time = torch.Tensor([0,1,2,3])
    )
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
    pp.plot(t, node_label=t.nodes, edge_color='lightgray');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Besides the standard formatting options available in pathpyG, temporal plots come with specific options tailored to their unique nature. These specialized settings allow for precise control over the time dimension of the visualization. The delta option lets you adjust the progression speed through the time steps of your visualization. Here, a value of 1000 translates to a one-second interval, providing a way to calibrate the pace at which the temporal data unfolds.
    """)
    return


@app.cell
def _(pp, t):
    color = {"a": "blue", "b": "red", "c": "green", "d": "yellow"}
    pp.plot(t, node_color=color, delta=2500);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The source nodes, destination nodes and timestamps of time-stamped edges are stored as a `pyG TemporalData` object, which we can access in the following way.
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
    return


@app.cell
def _(pd, pp):
    df_1 = pd.DataFrame([['a', 'b', 1], ['b', 'c', 2], ['a', 'c', 3]])
    print(df_1)
    t_3 = pp.io.df_to_temporal_graph(df_1)
    print(t_3)
    return (t_3,)


@app.cell
def _(os, pp, t_3, tempfile):
    tmpdirname = tempfile.mkdtemp()
    tmp_file = os.path.join(tmpdirname, 'test_temporal_graph.csv')
    pp.io.write_csv(t_3, path_or_buf=tmp_file)
    return (tmp_file,)


@app.cell
def _(pp, tmp_file):
    t_4 = pp.io.read_csv_temporal_graph(tmp_file)
    print(t_4)
    return


if __name__ == "__main__":
    app.run()
