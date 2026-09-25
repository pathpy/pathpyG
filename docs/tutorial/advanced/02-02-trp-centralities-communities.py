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
    # Time-respecting paths, centralities and communities

    *August 4 2026*
    *Training Workshop: Causality-Aware Temporal Networks*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we will show how temporal network can be used to calculate (shortest) time respecting paths between nodes. We also show how we can assess temporal node centralities and temporal community structures.
    """)
    return


@app.cell
def _():
    import pandas as pd

    import pathpyG as pp

    return pd, pp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To illustrate time-respecting paths, we reuse the small example temporal graph from the previous notebook:
    """)
    return


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
    We are often interested in time-respecting paths in a temporal graph. A time-respecting path consists of a sequence of nodes $v_0,...,v_l$ where consecutive nodes are connected by time-stamped edges that occur (i) in the right temporal ordering, and (ii) within a maximum time difference of $\delta\in \N$.

    To calculate time-respecting paths in a temporal graph, we can construct a directed acyclic graph (DAG), where each time-stamped edge $(u,v;t)$ in the temporal graph is represented by a node and two nodes representing time-stamped edges $(u,v;t_1)$ and $(v,w;t_2)$ are connected by an edge iff $0 < t_2-t_1 \leq \delta$. This implies that (i) each edge in the resulting DAG represents a time-respecting path of length two, and (ii) time-respecting paths of any lenghts are represented by paths in this DAG.

    We can construct such a DAG using the function `pp.core.event_graph.EventGraph.build_edge_index`, which returns an edge_index. We can pass this to the constructor of a `Graph` object, which we can use to visualize the resulting DAG.

    We call such a DAG a directed acyclic temporal event graph and we will cover it in more detail later in the tutorial.
    """)
    return


@app.cell
def _(pp, t):
    e_i = pp.core.event_graph.EventGraph.build_edge_index(t, delta=1)
    dag = pp.Graph.from_edge_index(e_i)
    pp.plot(dag, node_label = [f'{v}-{w}-{time}' for v, w, time in t.temporal_edges]);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $\delta=1$, this DAG with three connected components tells us that the underlying temporal graph has  the following time-respecting paths (of different lengths):
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
    g = t.to_static_graph(weighted=True)
    pp.plot(g, node_label=g.mapping.node_ids.tolist());
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pathpyG`'s ability to calculate (shortest) time-respecting paths enables us to calulate different notions of temporal centralities for nodes in empirial temporal networks. We can read an empirical temporal graph based on CSV data, where each line contains the source, target, and timestamp of an edge as comma-separated value:
    """)
    return


@app.cell
def _(pp):
    t_ants = pp.io.read_csv_temporal_graph(pp.io.example_data('ants_1_1.tedges'), header=False)
    print(t_ants)
    return (t_ants,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate the temporal closeness centrality, which is defined based on the length of shortest time-respecting paths of a node to all other nodes, we can write the following:
    """)
    return


@app.cell
def _(pp, t_ants):
    _cl = pp.algorithms.centrality.temporal_closeness_centrality(t_ants, delta=60)
    print(_cl)
    _mx = max(_cl.values())
    _mn = min(_cl.values())
    _node_size = {v: 50 * (x / (_mx - _mn)) for v, x in _cl.items()}
    pp.plot(t_ants, node_size=_node_size, edge_color='red', edge_size=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The definition of time-respecting paths depends on our maximum time difference parameter $\delta$, which implies that different values of this parameter also yield different centralities. This means that we can calculate temporal node centralities for different "time scales" of a temporal graph.
    """)
    return


@app.cell
def _(pp, t_ants):
    _cl = pp.algorithms.centrality.temporal_closeness_centrality(t_ants, delta=20)
    print(_cl)
    _mx = max(_cl.values())
    _mn = min(_cl.values())
    _node_size = {v: 50 * (x / (_mx - _mn)) for v, x in _cl.items()}
    pp.plot(t_ants, node_size=_node_size, edge_color='red', edge_size=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also calculate the temporal betweenness centrality, which is based on the number of shortest time-respecting paths between pairs of nodes that pass through a given node. Again, this centrality score is sensitive to the time scale parameter $\delta$.
    """)
    return


@app.cell
def _(pp, t_ants):
    _bw = pp.algorithms.centrality.temporal_betweenness_centrality(t_ants, delta=60)
    print(_bw)
    _mx = max(_bw.values())
    _mn = min(_bw.values())
    _node_size = {v: 50 * (x / (_mx - _mn)) for v, x in _bw.items()}
    pp.plot(t_ants, node_size=_node_size, edge_color='red', edge_size=4)
    return


@app.cell
def _(pp, t_ants):
    _bw = pp.algorithms.centrality.temporal_betweenness_centrality(t_ants, delta=20)
    print(_bw)
    _mx = max(_bw.values())
    _mn = min(_bw.values())
    _node_size = {v: 50 * (x / (_mx - _mn)) for v, x in _bw.items()}
    pp.plot(t_ants, node_size=_node_size, edge_color='red', edge_size=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal vs. Static Centrality

    How different are these temporal centralities from what we would get if we simply ignored the timing of interactions and computed standard centrality measures on the time-aggregated static graph? To find out, we compute the static closeness and betweenness centrality of the (weighted) time-aggregated ants network, and compare the top-10 ranked nodes to those obtained from the temporal centralities above ($\delta=60$).
    """)
    return


@app.cell
def _(pp, t_ants):
    _cl = pp.algorithms.centrality.temporal_closeness_centrality(t_ants, delta=60)
    _bw = pp.algorithms.centrality.temporal_betweenness_centrality(t_ants, delta=60)
    g_ants = t_ants.to_static_graph(weighted=True)
    static_cl = pp.algorithms.centrality.closeness_centrality(g_ants)
    static_bw = pp.algorithms.centrality.betweenness_centrality(g_ants)

    def top_k(centrality, k=10):
        """Return the k nodes with the highest centrality score."""
        return [node for node, _ in sorted(centrality.items(), key=lambda x: -x[1])[:k]]
    top_temporal_cl = top_k(_cl)
    top_static_cl = top_k(static_cl)
    top_temporal_bw = top_k(_bw)
    top_static_bw = top_k(static_bw)
    print('Top-10 nodes by temporal closeness:  ', top_temporal_cl)
    print('Top-10 nodes by static closeness:    ', top_static_cl)
    print('Overlap:', len(set(top_temporal_cl) & set(top_static_cl)), '/ 10')
    print()
    print('Top-10 nodes by temporal betweenness:', top_temporal_bw)
    print('Top-10 nodes by static betweenness:  ', top_static_bw)
    print('Overlap:', len(set(top_temporal_bw) & set(top_static_bw)), '/ 10')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rankings only partially agree. For closeness centrality, 8 of the top-10 nodes are shared between the temporal and static ranking, since closeness mainly reflects overall reachability, which is fairly robust to whether we account for time or not. For betweenness centrality, however, only half of the top-10 nodes agree: since betweenness depends on the exact set of shortest paths, and time-respecting paths can differ substantially from static shortest paths (recall the four `inf` entries we found earlier in our toy example), a node that appears to be an important "bridge" in the static topology may in fact rarely lie on an actual time-respecting path, and vice versa. This illustrates that ignoring the timing of interactions does not just add noise to centrality rankings, it can identify systematically different nodes as important.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Temporal Communities

    Time-respecting paths also let us uncover **temporal communities**: groups of nodes that are connected by many time-respecting paths, even in cases where this pattern is not visible in the topology of the time-aggregated graph. This is exactly the same phenomenon that motivates the higher-order De Bruijn graph models used in the previous notebooks, but here we take a complementary, simulation-based perspective.

    We load a synthetic temporal graph with 30 nodes and 60,000 time-stamped interactions, which was generated with a planted community structure (nodes 0-9, 10-19, and 20-29 each form a temporal community). Note that the CSV file uses the column names `source`, `target`, `time` instead of the `v`, `w`, `t` expected by `pp.io.read_csv_temporal_graph`, so we read it with `pandas` and rename the columns ourselves before converting it to a `TemporalGraph`:
    """)
    return


@app.cell
def _(pd, pp):
    df = pd.read_csv(pp.io.example_data('temporal_clusters.tedges'))
    t_clusters = pp.io.df_to_temporal_graph(df)
    print(t_clusters)
    return (t_clusters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now simulate a random walker that takes time-respecting steps through this temporal graph. We repeatedly pick a random time-stamped edge $(u,v;t_1)$ as the first step of a walk. The walk can only continue to a second edge $(v,w;t_2)$ if it is time-respecting for $\delta=1$, i.e. if $0 < t_2-t_1 \leq \delta$. If a valid continuation exists, we have simulated a time-respecting walk of length two from $u$ to $w$, and we record this by incrementing the entry $(u,w)$ of a matrix $M$. Walks that cannot be continued are simply discarded. Repeating this many times and normalizing by the number of successful walks gives us a matrix whose entries capture how often pairs of nodes are connected by a time-respecting walk of length two.
    """)
    return


@app.cell
def _(t_clusters):
    import numpy as np

    np.random.seed(0)

    delta = 1
    num_walks = 20000

    # temporal_edges yields (v, w, time) tuples in temporal order
    events = list(t_clusters.temporal_edges)

    M = np.zeros((t_clusters.n, t_clusters.n))
    num_successful = 0

    for i in np.random.randint(0, len(events) - 1, size=num_walks):
        u, v, t1 = events[i]
        v2, w, t2 = events[i + 1]

        # the walk can only continue if the second edge starts where the first one ended
        # and is time-respecting for delta=1
        if v2 == v and 0 < (t2 - t1) <= delta:
            M[t_clusters.mapping.to_idx(u), t_clusters.mapping.to_idx(w)] += 1
            num_successful += 1

    print(f'{num_successful} of {num_walks} simulated walks could be continued to length two')
    return M, delta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's plot the resulting matrix $M$, using the (numerical) node identifiers to order the rows and columns:
    """)
    return


@app.cell
def _(M, delta):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap='viridis')
    plt.colorbar(label='number of simulated walks of length two')
    plt.xlabel('target node $w$')
    plt.ylabel('source node $u$')
    plt.title(f'Time-respecting walks of length two ($\\delta={delta}$)')
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though we never told the walker anything about node groups, three brighter blocks emerge along the diagonal of the matrix, corresponding exactly to the three planted communities (nodes 0-9, 10-19, and 20-29): time-respecting walks of length two much more frequently stay within one of these groups than they cross between groups. This is despite the fact that the topology of the time-aggregated graph itself does not show any such community structure (there is nothing special about the *static* connections between these nodes). The community structure only becomes visible once we take the temporal ordering of interactions into account, i.e. it is a genuinely **temporal-topological** pattern rather than a purely topological one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For comparison, let's look at the same kind of matrix, but this time simply counting the number of (time-aggregated) edges between each pair of nodes, ignoring their timing entirely. We can obtain this using the `to_static_graph` function that we already used earlier in this notebook, which lets us access the edge counts as a weighted adjacency matrix:
    """)
    return


@app.cell
def _(plt, t_clusters):
    g_clusters = t_clusters.to_static_graph(weighted=True)
    A = g_clusters.sparse_adj_matrix(edge_attr='edge_weight').todense()

    plt.figure(figsize=(6, 5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='number of edges')
    plt.xlabel('target node $w$')
    plt.ylabel('source node $u$')
    plt.title('Time-aggregated edge counts (static topology)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unlike the matrix of time-respecting walks of length two, the plain edge-count matrix shows no visible block structure: edges are essentially uniformly scattered across all pairs of nodes, regardless of community membership. This confirms that the community structure we found above is a genuinely temporal phenomenon: it is encoded in the *order* in which edges occur, not in *which* edges exist. A purely static analysis of this graph based on its edges alone would completely miss the three communities.
    """)
    return


if __name__ == "__main__":
    app.run()
