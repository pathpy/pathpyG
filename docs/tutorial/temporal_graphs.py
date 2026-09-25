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
    # %%capture
    # # !pip install torch
    # # !pip install torch_geometric
    # # !pip install git+https://github.com/pathpy/pathpyG.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation and Learning Objectives

    In this tutorial we will introduce the representation of temporal graph data using the `TemporalGraph` class and how such data can be used to calculate shortest time respecting paths between nodes as well temporal node cemtralities.
    """)
    return


@app.cell
def _():
    import os
    import tempfile

    import torch
    from torch_geometric.data import Data

    import pathpyG as pp

    return Data, os, pp, tempfile, torch


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
    ## Extracting Time-Respecting Paths in Temporal Networks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are often interested in time-respecting paths in a temporal graph. A time-respecting path consists of a sequence of nodes $v_0,...,v_l$ where consecutive nodes are connected by time-stamped edges that occur (i) in the right temporal ordering, and (ii) within a maximum time difference of $\delta\in N$.

    To calculate time-respecting paths in a temporal graph, we can construct a directed acyclic graph (DAG), where each time-stamped edge $(u,v;t)$ in the temporal graph is represented by a node and two nodes representing time-stamped edges $(u,v;t_1)$ and $(v,w;t_2)$ are connected by an edge iff $0 < t_2-t_1 \leq \delta$. This implies that (i) each edge in the resulting DAG represents a time-respecting path of length two, and (ii) time-respecting paths of any lengths are represented by paths in this DAG.

    We can construct such a DAG using the function `pp.core.event_graph.EventGraph.build_edge_index`, which returns an edge_index. We can pass this to the constructor of a `Graph` object, which we can use to visualize the resulting DAG.
    """)
    return


@app.cell
def _(pp, t):
    e_i = pp.core.event_graph.EventGraph.build_edge_index(t, delta=1)
    return (e_i,)


@app.cell
def _(e_i, pp, t):
    dag_mapping = pp.IndexMap([f"{v}->{w}: {time}" for v, w, time in t.temporal_edges ])
    dag = pp.Graph.from_edge_index(e_i, mapping=dag_mapping)
    pp.plot(dag);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $\delta=1$, this DAG with four connected components tells us that the underlying temporal graph has  the following time-respecting paths (of different lengths):
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Length one:
        a -> b
        b -> a
        b -> c
        c -> b
        c -> d
        d -> c

    Length two:
        a -> b -> a (twice, starting at time 2 and time 4)
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
    We can can use the function `pp.algorithms.temporal.temporal_shortest_paths` to calculate shortest time-respecting path distances between any pair of nodes. This also returns a predecessor matrix, which can be used to reconstruct all shortest time-respecting paths (in analogy to the Dijkstra algorithm for static graphs):
    """)
    return


@app.cell
def _(pp, t):
    dist, pred = pp.algorithms.temporal_shortest_paths(t, delta=1)
    return dist, pred


@app.cell
def _(dist, pred, t):
    print(t.mapping)
    print(dist)
    print(pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the example above, the four `inf` values indicate that there is no time-respecting paths between the four node pairs (a, d), (b, d), (d,a) and (d, b). This is not something we would expect based on the (strongly connected) topology of the time-aggregated graph, which is shown below:
    """)
    return


@app.cell
def _(pp, t):
    _g = t.to_static_graph(weighted=True)
    pp.plot(_g)
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
    pp.plot(csv_t)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal Centralities in Empirical Temporal Networks

    `pathpyG`'s ability to calculate (shortest) time-respecting paths enables us to calulate different notions of temporal centralities for nodes in empirical temporal networks. We can download an empirical temporal graph from Netzschleuder:
    """)
    return


@app.cell
def _(pp):
    t_baboons = pp.io.read_netzschleuder_graph('sp_baboons', 'observational', time_attr='time')
    t_baboons.data.time = (t_baboons.data.time - t_baboons.data.time.min()) // 60 # convert to minutes
    print(t_baboons)
    return (t_baboons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate the temporal closeness centrality, which is defined based on the length of shortest time-respecting paths of a node to all other nodes, we can write the following:
    """)
    return


@app.cell
def _(pp, t_baboons):
    cl = pp.algorithms.centrality.temporal_closeness_centrality(t_baboons, delta=24*60)
    return (cl,)


@app.cell
def _(cl, pp, t_baboons):
    print(cl)
    _node_size = {v: 15 * (x / max(cl.values())) for v, x in cl.items()}
    pp.plot(t_baboons, node_size=_node_size, node_color=t_baboons.nodes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The definition of time-respecting paths depends on our maximum time difference parameter $\delta$, which implies that different values of this parameter also yield different centralities. This means that we can calculate temporal node centralities for different "time scales" of a temporal graph.
    """)
    return


@app.cell
def _(pp, t_baboons):
    cl_1 = pp.algorithms.centrality.temporal_closeness_centrality(t_baboons, delta=1)
    return (cl_1,)


@app.cell
def _(cl_1, pp, t_baboons):
    print(cl_1)
    _node_size = {v: 15 * (x / max(cl_1.values())) for v, x in cl_1.items()}
    pp.plot(t_baboons, node_size=_node_size, node_color=t_baboons.nodes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also calculate the temporal betweenness centrality, which is based on the number of shortest time-respecting paths between pairs of nodes that pass through a given node. Again, this centrality score is sensitive to the time scale parameter $\delta$.
    """)
    return


@app.cell
def _(pp, t_baboons):
    bw = pp.algorithms.centrality.temporal_betweenness_centrality(t_baboons, delta=24*60)
    return (bw,)


@app.cell
def _(bw, pp, t_baboons):
    print(bw)
    _node_size = {v: 15 * (x / max(bw.values())) for v, x in bw.items()}
    pp.plot(t_baboons, node_size=_node_size, node_color=t_baboons.nodes)
    return


@app.cell
def _(pp, t_baboons):
    bw_1 = pp.algorithms.centrality.temporal_betweenness_centrality(t_baboons, delta=1)
    return (bw_1,)


@app.cell
def _(bw_1, pp, t_baboons):
    print(bw_1)
    _node_size = {v: 15 * (x / max(bw_1.values())) for v, x in bw_1.items()}
    pp.plot(t_baboons, node_size=_node_size, node_color=t_baboons.nodes)
    return


if __name__ == "__main__":
    app.run()
