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
    # Event Graphs

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

    In [Time-Respecting Paths, Centralities and Communities](../trp_centralities_communities/), we have represented the time-respecting paths in a temporal graph by a directed acyclic graph, whose nodes are the time-stamped edges (or *events*) of the temporal graph, and where two events are connected if the second one can continue a time-respecting path that ends with the first one. There, we have built the edge index of this **event graph** using the low-level function `EventGraph.build_edge_index` and manually created a mapping of node IDs.

    The class `EventGraph` wraps this construction into a subclass of `Graph` that additionally keeps track of the timestamp and the first-order edge of each event, as well as of the maximum time difference $\delta$. In this tutorial, you will learn how to

    - create an event graph from a temporal graph,
    - access the timestamps and time differences of events,
    - change the maximum time difference $\delta$ of an existing event graph,
    - reuse an event graph to calculate shortest time-respecting paths, and
    - aggregate an event graph into a higher-order De Bruijn graph.

    Note the difference between an `EventGraph` and the `HigherOrderGraph` introduced in [Working with Higher-Order De Bruijn Graphs](../higher_order_graphs/): an event graph has one node per *event*, i.e. it keeps every single time-stamped interaction along with its timestamp. A `HigherOrderGraph` instead aggregates all occurrences of the same path into a single weighted edge, discarding the timestamps.
    """)
    return


@app.cell
def _():
    import torch

    import pathpyG as pp

    return pp, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We again use the small example temporal graph from [Time-Respecting Paths, Centralities and Communities](../trp_centralities_communities/). To illustrate how attributes are handled, we additionally assign a (made-up) duration to each time-stamped edge:
    """)
    return


@app.cell
def _(pp, torch):
    tedges = [('a', 'b', 1), ('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4),
              ('c', 'd', 5), ('b', 'a', 5), ('c', 'b', 6)]
    t = pp.TemporalGraph.from_edge_list(tedges)
    t.data.edge_duration = torch.tensor([10.0, 20.0, 5.0, 15.0, 30.0, 10.0, 25.0, 5.0, 20.0, 10.0])
    print(t)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Event Graphs

    We can create an event graph for a given maximum time difference $\delta$ using the class method `EventGraph.from_temporal_graph`:
    """)
    return


@app.cell
def _(mo, pp, t):
    # Hide the progress output
    with mo.capture_stdout(), mo.capture_stderr():
        eg = pp.EventGraph.from_temporal_graph(t, delta=1)
    print(eg)
    return (eg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The event graph has one node for each of the ten time-stamped edges in the temporal graph. Nodes are automatically assigned IDs of the form `v->w@t`, so we no longer need to create a mapping ourselves:
    """)
    return


@app.cell
def _(eg, pp):
    print(eg.nodes)
    pp.plot(eg);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The seven edges of the event graph are the time-respecting paths of length two for $\delta=1$. Note that the path $a \rightarrow b \rightarrow a$ occurs twice (starting at times two and four), and that each occurrence is a separate edge in the event graph:
    """)
    return


@app.cell
def _(eg):
    for _e in eg.edges:
        print(_e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each event, the tensors `node_sequence` and `node_time` store the (first-order) source and target node indices and the timestamp, respectively. The event graph further provides some convenience properties to access the number of events and first-order nodes, as well as the time span of the events:
    """)
    return


@app.cell
def _(eg):
    print(eg.data.node_sequence.t())
    print(eg.data.node_time)
    print(eg.num_events, eg.n_first_order)
    print(eg.start_time, eg.end_time)
    print(eg.event_time(eg.mapping.to_idx('c->d@5')))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Edge attributes of the temporal graph are attributes of individual events. When we create an event graph, they thus become node attributes, and their prefix is changed from `edge_` to `node_`. In our example, the attribute `edge_duration` of the temporal graph becomes the attribute `node_duration` of the event graph, which we can access via event IDs. Node attributes of the temporal graph refer to first-order nodes, which have no counterpart in the event graph, so they are not propagated.
    """)
    return


@app.cell
def _(eg):
    print(eg.data.node_duration)
    print(eg['node_duration', 'd->c@4'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the event graph keeps the underlying first-order edges and timestamps of all events, we can always recover the temporal graph from which it was built:
    """)
    return


@app.cell
def _(eg):
    print(eg.to_temporal_graph())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Differences and the Parameter $\delta$

    For each edge of the event graph, the attribute `edge_delta` stores the time difference between the two events that it connects. For $\delta=1$, all time differences are trivially equal to one. Let us create an event graph with a larger maximum time difference $\delta=2$:
    """)
    return


@app.cell
def _(mo, pp, t):
    # Hide the progress output
    with mo.capture_stdout(), mo.capture_stderr():
        eg_2 = pp.EventGraph.from_temporal_graph(t, delta=2)
    print(eg_2)
    for _e, _d in zip(eg_2.edges, eg_2.data.edge_delta.tolist()):
        print(_e, _d)
    return (eg_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With $\delta=2$, events that are two time steps apart can also follow each other, which gives rise to four additional time-respecting paths of length two.

    We often want to compare time-respecting paths for different values of $\delta$. Rather than constructing a new event graph from scratch, we can use the function `reduce_delta`, which removes all edges whose time difference exceeds the new maximum time difference. Reducing $\delta=2$ by one gives us the same seven edges that we obtained for $\delta=1$ above:
    """)
    return


@app.cell
def _(eg_2):
    eg_reduced = eg_2.reduce_delta(1)
    print(eg_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shortest Time-Respecting Paths

    Since the paths in an event graph are exactly the time-respecting paths of the temporal graph, we can calculate shortest time-respecting paths directly on the event graph. The function `shortest_paths` returns the same distance and predecessor matrices as `pp.algorithms.temporal_shortest_paths`, but it does not need to construct the event graph again. This is useful if we calculate several measures based on the same event graph:
    """)
    return


@app.cell
def _(eg, mo, pp, t):
    # Hide the progress output
    with mo.capture_stdout(), mo.capture_stderr():
        dist, pred = eg.shortest_paths()
        _dist_t, _ = pp.algorithms.temporal_shortest_paths(t, delta=1)
    print(t.mapping)
    print(dist)
    print((dist == _dist_t).all())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From Event Graphs to Higher-Order De Bruijn Graphs

    An event graph grows with the number of events, which can be very large for empirical temporal graphs. For many purposes, we are not interested in *when* exactly a time-respecting path occurred, but only in *which* paths occur and how often. We can aggregate an event graph into a De Bruijn graph of order $k$ using the class method `HigherOrderGraph.from_event_graph`, which uses the maximum time difference $\delta$ of the event graph. For $k=2$, the two separate edges representing the time-respecting path $a \rightarrow b \rightarrow a$ are merged into a single edge with weight two:
    """)
    return


@app.cell
def _(eg, mo, pp):
    # Hide the progress output
    with mo.capture_stdout(), mo.capture_stderr():
        g2 = pp.HigherOrderGraph.from_event_graph(eg, order=2)
    print(g2)
    for _e in g2.edges:
        print(_e, g2['edge_weight', _e[0], _e[1]].item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is equivalent to calling `HigherOrderGraph.from_temporal_graph(t, order=2, delta=1)`, which constructs the event graph internally. Similarly, `MultiOrderModel.from_event_graph` creates the De Bruijn graphs of all orders up to a given maximum order from an event graph.

    In the next tutorial, [Higher-Order Models for Time-Respecting Paths](../trp_higher_order/), we use multi-order models to analyze time-respecting paths of different lengths, both in our toy example and in empirical temporal networks. For details on how the edge index of event graphs is constructed efficiently, see [Implementation Concepts](../implementation_concepts/).
    """)
    return


if __name__ == "__main__":
    app.run()
