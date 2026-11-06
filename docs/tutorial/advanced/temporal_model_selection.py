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
    # Model Selection for Time-Respecting Paths

    ## Prerequisites

    First, we need to set up our Python environment that has PyTorch, PyTorch Geometric and PathpyG installed. Depending on where you are executing this notebook, this might already be (partially) done. E.g. Google Colab has PyTorch installed by default so we only need to install the remaining dependencies. The DevContainer that is part of our GitHub Repository on the other hand already has all of the necessary dependencies installed.
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

    In the tutorial on [path data and higher-order models](paths_higher_order.ipynb) we used statistical model selection to answer the question *how much memory does this data have?* for a collection of observed paths. In the tutorial on [higher-order models for time-respecting paths](trp_higher_order.ipynb) we built De Bruijn graph models directly from a temporal graph, without ever materialising the paths.

    This tutorial puts the two together: how to detect the optimal order of a multi-order model for the **time-respecting paths of a temporal graph**, without first extracting them.

    We will see that this needs one extra decision that does not arise for explicitly observed paths, namely *what counts as an observation*, and why getting that decision wrong quietly invalidates the statistical test.
    """)
    return


@app.cell
def _():
    import torch

    import pathpyG as pp
    from pathpyG.algorithms import extract_time_respecting_walks, temporal_clusters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return extract_time_respecting_walks, pp, temporal_clusters, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Exactly Are We Fitting the Model To?

    A likelihood ratio test compares two hypotheses about *the same data*. When we test whether a second-order model explains our data better than a first-order one, both likelihoods have to be likelihoods of one fixed set of observations.

    For a `PathData` object that is obvious: the observations are the paths we were given. For a temporal graph it is not, because a temporal graph does not come with a set of paths — it comes with time-stamped edges, and which time-respecting walks we read off them is our choice.

    The choice `pathpyG` makes is to take **all time-respecting walks of exactly `walk_length` events** as the observed data, where a walk of $L$ events visits $L+1$ nodes and consecutive events are at most `delta` apart. Crucially, `walk_length` is fixed independently of the order being tested. Counting, say, the distinct walks of $k$ events instead would make the data itself a function of $k$, and the test would compare two likelihoods of two different data sets.

    `walk_length` defaults to `max_order`, and the model remembers it:
    """)
    return


@app.cell
def _(pp):
    tedges = [("a", "b", 1), ("b", "c", 2), ("c", "d", 3), ("c", "e", 3), ("a", "b", 4), ("b", "c", 5)]
    t = pp.TemporalGraph.from_edge_list(tedges)

    m = pp.MultiOrderModel.from_temporal_graph(t, delta=1, max_order=3)
    print(m)
    print("number of observations:", m.num_observations)
    return m, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Those observations are exactly the time-respecting walks of three events. We never build them, but we can, which is a useful way to check what the model is actually fitted to:
    """)
    return


@app.cell
def _(extract_time_respecting_walks, t):
    walks = extract_time_respecting_walks(t, delta=1, length=3)
    print(walks)
    for i in range(walks.num_paths):
        print(walks.get_walk(i))
    return (walks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitting a model to those extracted walks gives exactly the same layers, order for order. The difference is that the model above never materialised a single walk: it counts them with a dynamic program over the event graph, which stays linear in the size of that graph no matter how many walks there are.
    """)
    return


@app.cell
def _(m, pp, torch, walks):
    from_walks = pp.MultiOrderModel.from_path_data(walks, max_order=3)
    for _order in (1, 2, 3):
        same_weights = torch.allclose(m.layers[_order].data.edge_weight.double(), from_walks.layers[_order].data.edge_weight.double())
        print(f'order {_order}: layers agree = {same_weights}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="admonition warning">
        <p class="admonition-title">Models with different <code>walk_length</code> are not comparable</p>

            Two models built with a different <code>walk_length</code> describe different observation sets, so their likelihoods, evidences and Bayes factors cannot be compared with each other. Within a single model everything shares one <code>walk_length</code>, which is why <code>estimate_order</code> is always well defined.


    </div>

    ### Two Kinds of Edge Weight

    Because the observation set is fixed, the layer weights are no longer simply the number of distinct time-respecting walks an edge represents. Each layer therefore carries both:

    - `edge_weight` counts how often the walk is realized *inside an observation*, once per position it can take. This is the statistic model selection uses. The order-0 layer counts the visits of each node in the same way: each observation visits the source of each of its events and the target of its last event.
    - `edge_num_walks` counts the distinct time-respecting walks of that length, i.e. the classical De Bruijn graph weight.

    They agree at the maximum order of the model and can differ below it.
    """)
    return


@app.cell
def _(m):
    for _order in (1, 2, 3):
        layer = m.layers[_order]
        print(f'order {_order}')
        print('   edge_weight   :', layer.data.edge_weight.tolist())
        print('   edge_num_walks:', layer.data.edge_num_walks.tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating the Optimal Order

    We now generate a temporal graph with a *planted* second-order pattern. The generator `temporal_clusters` draws a random graph, splits its nodes into clusters, samples causal walks of two events, and then exchanges time stamps between walks through the same node so that a walk tends to end in the cluster it started in.

    The exchange leaves every time-stamped edge in place, so the time-aggregated weighted graph is exactly the graph a model without the planted structure would have. A first-order model therefore cannot see the pattern at all — which is precisely what makes this a good test of model selection.
    """)
    return


@app.cell
def _(temporal_clusters):
    g = temporal_clusters(
        num_nodes=15,
        num_edges=120,
        num_clusters=3,
        num_walks=4000,
        seed=0,
    )
    cluster = g.data.node_cluster
    print(g)
    return cluster, g


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us confirm that the pattern really is invisible at the first order. We compare the cluster of the first node of each causal walk with the cluster of the second node, and with the cluster of the third:
    """)
    return


@app.cell
def _(cluster, g):
    edge_index = g.data.edge_index.as_tensor()
    first, second, third = edge_index[0][0::2], edge_index[1][0::2], edge_index[1][1::2]

    print("P(cluster of v1 == cluster of v0):", round(float((cluster[first] == cluster[second]).float().mean()), 3))
    print("P(cluster of v2 == cluster of v0):", round(float((cluster[first] == cluster[third]).float().mean()), 3))
    print("chance level                     :", round(1 / 3, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first step is indistinguishable from chance, the second is strongly clustered. Now we build a multi-order model and ask it for the optimal order:
    """)
    return


@app.cell
def _(g, pp):
    m_1 = pp.MultiOrderModel.from_temporal_graph(g, delta=1, max_order=3)
    print(m_1)
    print('estimated order:', m_1.estimate_order())
    return (m_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model correctly reports order two. We can look at the evidence behind that decision, either through the $p$-values of the likelihood ratio tests or through the model evidence of each order:
    """)
    return


@app.cell
def _(m_1):
    for _order in range(1, 4):
        rejected, p_value = m_1.likelihood_ratio_test(null_max_order=_order - 1, max_order=_order)
        print(f'order {_order - 1} vs {_order}: p = {p_value:.3g}, reject the simpler model = {rejected}')
    print()
    for _order in range(4):
        print(f'log evidence for order {_order}: {m_1.log_evidence(max_order=_order):.1f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing Between the Two Selectors

    `estimate_order` offers a frequentist and a Bayesian selector. The likelihood ratio test compares nested models through a $\chi^2$ test; the Bayesian selector integrates the transition probabilities out under a Dirichlet prior and compares the resulting model evidences through Bayes factors.

    The Bayesian variant needs considerably less data before it can tell two orders apart. Repeating the experiment above with fewer and fewer causal walks shows where the two part ways:
    """)
    return


@app.cell
def _(pp, temporal_clusters):
    for num_walks in (200, 400, 1500, 4000):
        smaller = temporal_clusters(num_nodes=15, num_edges=120, num_clusters=3, num_walks=num_walks, seed=0)
        model = pp.MultiOrderModel.from_temporal_graph(smaller, delta=1, max_order=3)
        print(
            f"{num_walks:5d} walks: "
            f"likelihood ratio -> {model.estimate_order(method='likelihood_ratio')}, "
            f"Bayesian -> {model.estimate_order(method='bayesian')}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - For a temporal graph, the observed data is the set of all time-respecting walks of `walk_length` events, and `walk_length` must not depend on the order being tested.
    - The statistics are computed by a dynamic program over the event graph, so no walk is ever materialised; `extract_time_respecting_walks` builds them explicitly when that is what you want.
    - Each layer carries both the observation statistic (`edge_weight`) and the classical De Bruijn weight (`edge_num_walks`); the order-0 layer counts node visits.
    - `estimate_order` offers a frequentist and a Bayesian selector. Both recover the planted second order here, but the Bayesian one already does so from a fraction of the data.
    """)
    return


if __name__ == "__main__":
    app.run()
