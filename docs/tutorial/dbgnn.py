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
    # Causality-Aware Graph Neural Networks

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

    In previous tutorials, we have introduced causal paths in temporal graphs, and how we can use them to generate higher-order De Bruijn graph models that capture temporal-topological patterns in time series data. In this tutorial, we will show how we can use De Bruijn Graph Neural Networks, a causality-aware deep learning architecture for temporal graph data. The details of this approach are introduced [in this paper](https://proceedings.mlr.press/v198/qarkaxhija22a.html). The architecture is implemented in pathpyG and can be readily applied to temporal graph data.

    Below we illustrate this method in a supervised node classification task, i.e. given a temporal graph we will use the temporal-topological patterns in the graph to classify nodes.

    We start by importing a few modules:
    """)
    return


@app.cell
def _():
    from copy import deepcopy

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import torch
    from sklearn.manifold import TSNE
    from sklearn.metrics import balanced_accuracy_score
    from torch_geometric.transforms import RandomNodeSplit

    import pathpyG as pp
    from pathpyG.nn.dbgnn import DBGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (
        DBGNN,
        RandomNodeSplit,
        TSNE,
        balanced_accuracy_score,
        deepcopy,
        device,
        np,
        plt,
        pp,
        sp,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal-Topological Clusters in Temporal Graphs

    Let us load a small synthetic toy example for a temporal graph with 60.000 time-stamped interactions between 30 nodes. We use the `TemporalGraph` class to load this example from a file containing edges with discrete time-stamps.

    <div class="admonition note">
        <p class="admonition-title">Dataset Availability</p>

            Depending on how you are executing this notebook, the dataset may not be available locally. We use <code>pp.io.example_data</code> to resolve the file name: it returns the path of a local copy in the <code>docs/data</code> folder of the pathpyG repository, or - if you are running outside of a clone, e.g. in Google Colab - a URL pointing to the GitHub repository. Both can be passed directly to the <code>pp.io.read_csv_*</code> functions.


    </div>
    """)
    return


@app.cell
def _(device, pp):
    t = pp.io.read_csv_temporal_graph(pp.io.example_data('temporal_clusters.tedges'))

    t = t.to(device)
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This example has created in such a way that the nodes naturally form three clusters, which are highlighted in the interactive visualization below:
    """)
    return


@app.cell
def _(pp, t):
    style = {}
    style["node_color"] = ["green"] * 10 + ["red"] * 10 + ["blue"] * 10
    pp.plot(t, **style, show_labels=False);
    return (style,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelling Causal Structures with Higher-Order De Bruijn Graphs

    But what is the origin for the cluster pattern? In the visualization above, you will notice that the time-stamped edges randomly interconnect nodes within and across clusters, actually there is no correlation whatsoever between the topology of links and the cluster membership of the nodes. Hence, the notion of clusters does not correspond to the common idea of cluster patterns in static graphs, which we can highlight further by plotting the static time-aggregated network:
    """)
    return


@app.cell
def _(pp, style, t):
    pp.plot(t.to_static_graph(), **style, show_labels=False);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In fact, the topology of this graph corresponds to that of a random graph, i.e. there are not patterns whatsoever in the topology of links. Nevertheless, the temporal graph contains a cluster pattern in the topology of causal or time-respecting paths. In particular, the temporal ordering of time-stamped edges is such that nodes with the same cluster label are more frequently connected by time-respecting paths than nodes with different cluster labels. Hence, nodes within the same clusters can more strongly influence each other in a causal way, i.e. via multiple interactions that follow the arrow of time.

    Traditional (temporal) graph neural networks will not be able to learn from this pattern, as it is due to the specific microscopic temporal ordering of edges. Using higher-order De Bruijn graph models implemented in pathpyG, we can learn from temporal graph data that contains such patterns. Let us explain this step by step.

    Referring to the previous tutorial on causal paths in temporal graphs, we first create a node-time directed acyclic graph that captures the causal structure of the temporal graph. In this small example, we will only consider two time-stamped edges $(u,v;t)$ and $(v,w;t')$ to contribute to a causal path iff $0 < t'-t \leq 1$, i.e. we use a delta for the maximum time difference of one time step.
    """)
    return


@app.cell
def _(pp, t):
    m = pp.MultiOrderModel.from_temporal_graph(t, max_order=2)
    return (m,)


@app.cell
def _(m):
    print(m)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can get the first and second order networks from the Multi Order Network object. The first order network is the network of nodes and edges, while the second order network is the network of first order edges as second order nodes and second order edges. The second order network is a De Bruijn graph that captures the temporal-topological patterns in the data.
    """)
    return


@app.cell
def _(m):
    g = m.layers[1]
    g2 = m.layers[2]
    return g, g2


@app.cell
def _(g, pp):
    pp.plot(g, edge_size=1, show_labels=False);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since it does not consider patterns in the causal topology of the temporal graph, this is not a meaningful model. We can instead use a second-order De Bruijn graph model, which we can easily fit to the paths:
    """)
    return


@app.cell
def _(g2, pp):
    layout_style = {}
    layout_style['layout'] = 'Fruchterman-Reingold'
    layout_style['seed'] = 1
    layout_style['k'] = 0.5
    layout_style['iterations'] = 300
    _layout = pp.layout(g2, **layout_style)
    pp.plot(g2, backend='matplotlib', layout=_layout, edge_size=0.5, node_size=3, show_labels=False)
    return (layout_style,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this graph, every node is a link and links correspond to causal paths of length two, i.e. temporally ordered sequences consisting of two edges that overlap in the center node. In this graph, we clearly see a cluster pattern that is due to the way in which temporal edges are ordered in time. In particular, we see three clusters, where the edges in three of the clusters correspond to causal paths of length two that connect nodes within each of the three clusters. The edges in the fourth cluster (in the center of the visualization) represent causal paths that connect nodes in different clusters.

    ## Comparison to Temporal Graph with Shuffled Time Stamps

    You may wonder whether this pattern is really due to the temporal ordering of time-stamped edges. It is easy to check this. We can simply randomly shuffle the time stamps of all edges, which will break any correlations in the temporal ordering that lead to patterns in the causal topology.

    We repeat the path calculation for this shuffled temporal graph and construct the second-order De Bruijn Graph model again:
    """)
    return


@app.cell
def _(deepcopy, t):
    t_shuffled = deepcopy(t)
    t_shuffled.shuffle_time()
    return (t_shuffled,)


@app.cell
def _(pp, t_shuffled):
    g2_shuffled = pp.MultiOrderModel.from_temporal_graph(t_shuffled, max_order=2).layers[2]
    return (g2_shuffled,)


@app.cell
def _(g2_shuffled):
    print(g2_shuffled)
    return


@app.cell
def _(g2_shuffled, layout_style, pp):
    _layout = pp.layout(g2_shuffled, **layout_style)
    pp.plot(g2_shuffled, backend='matplotlib', layout=_layout, edge_size=0.5, node_size=3, show_labels=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now find that the cluster pattern in the second-order graph has vanished. In fact, there is no pattern whatsoever since the underlying (static) graph topology is random and the random shuffling of time stamps leads to random causal paths.

    ## Spectral clustering with second-order graph Laplacian

    To take a different perspective on cluster patterns, we can actually use `pathpyG` to apply a spectral analysis to the higher-order graph. We can simply calculate a generalization of the Laplacian matrix to the second-order graph both for the actual temporal graph and its shuffled counterpart:
    """)
    return


@app.cell
def _(g2, g2_shuffled):
    L = g2.laplacian(normalization='rw', edge_attr='edge_weight')
    L_shuffled= g2_shuffled.laplacian(normalization='rw',edge_attr='edge_weight')
    return L, L_shuffled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then calculate the eigenvalues and eigenvectors of the Laplacians, and compute the Fiedler vector, i.e. the eigenvector that corresponds to the second-smallest eigenvalue of the Laplacian.
    """)
    return


@app.cell
def _(L, L_shuffled, sp):
    w,v = sp.linalg.eig(L.todense(),left= False, right = True)
    w_shuffled, v_shuffled = sp.linalg.eig(L_shuffled.todense())
    return v, v_shuffled, w, w_shuffled


@app.cell
def _(np, v, v_shuffled, w, w_shuffled):
    fiedler = v[:,np.argsort(w)[1]]
    fiedler_shuffled = v_shuffled[:,np.argsort(w_shuffled)[1]]
    return fiedler, fiedler_shuffled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below, we show that the clusters in the causal topology of the temporal graph correspond to clusters in the distribution of entries in the Fiedler vector, while there is no such pattern for the Fiedler vector of the second-order graph constructed from the shuffled temporal graph:
    """)
    return


@app.function
def higher_order_class_assignment(ho_node_id):
    """Assign class labels based on the higher-order node ids.
    
    There are three classes that were assigned based on the original node ids:
        - Class 0: node ids 0-9
        - Class 1: node ids 10-19
        - Class 2: node ids 20-29

    The higher-order patterns were constructed such that the clusters are formed by second-order nodes whose first-order node ids belong to the same class.
    We therefore assign higher-order class labels based on the first-order node ids in the higher-order node id tuple.
    """
    if ho_node_id[0] < 10 and ho_node_id[1] < 10:
        return 0
    elif ho_node_id[0] < 20 and ho_node_id[0] >= 10 and ho_node_id[1] < 20 and ho_node_id[1] >= 10:
        return 1
    elif ho_node_id[0] < 30 and ho_node_id[0] >= 20 and ho_node_id[1] < 30 and ho_node_id[1] >= 20:
        return 2
    else:
        return 3


@app.cell
def _():
    colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'gray'}
    opacities = {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.1}
    return colors, opacities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the plots below, we have colored those entries of the Fiedler vectors that correspond to edges connecting nodes within one of the three clusters shown above. The Fiedler vector shows a clear pattern, which translates to the cluster pattern in the causal topology that we have planted into our synthetic temporal graph.
    """)
    return


@app.cell
def _(colors, fiedler, g2, np, opacities, plt):
    ho_class_ids = list(map(higher_order_class_assignment, g2.nodes))
    plt.scatter(
        range(g2.n), np.real(fiedler), c=[colors[i] for i in ho_class_ids], alpha=[opacities[i] for i in ho_class_ids]
    )
    plt.ylim(-0.25, 0.25)
    return (ho_class_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No such pattern exists in the Fiedler vector of the second-order graph corresponding to the shuffled `TemporalGraph`.
    """)
    return


@app.cell
def _(colors, fiedler_shuffled, g2_shuffled, np, opacities, plt):
    shuffled_ho_class_ids = list(map(higher_order_class_assignment, g2_shuffled.nodes))
    plt.scatter(
        range(g2_shuffled.n),
        np.real(fiedler_shuffled),
        c=[colors[i] for i in shuffled_ho_class_ids],
        alpha=[opacities[i] for i in shuffled_ho_class_ids],
    )
    plt.ylim(-0.25, 0.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node Classification with Causality-Aware Graph Neural Networks

    Let us now explore how we can develop a causality-aware deep graph learning architecture that utilizes this pattern in the causal topology. We will follow the architecture introduced [in this work](https://proceedings.mlr.press/v198/qarkaxhija22a.html). The architecture actually performs message passing in higher-order models with multiple orders at once. In a final message passing step, a bipartite graph is used to obtain vector-space representations of actual nodes in the temporal graph.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now set up a `pytorch_geometric.Data` object that contains all of the information needed to train the DBGNN model. For this, we can use a convenience function of the `MultiOrderModel` class in `pathpyG`. Combining a first- and a second-order model, this uses the edge indices and the weight tensors for a message passing scheme. it further constructs an `edge_index` of a bipartite graph that uses the last node in a second-order node to map messages back to first-order nodes.
    """)
    return


@app.cell
def _(device, m, t, torch):
    data = m.to_dbgnn_data(max_order=2, mapping="last")
    data.y = torch.tensor([int(i) // 10 for i in t.nodes], device=device)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the model

    We are now ready to train and evaluate our causality-aware graph neural network. We will frist create a random split of the nodes, set the optimizer and the hyperparameters of our model.
    """)
    return


@app.cell
def _(DBGNN, RandomNodeSplit, data, device, g, g2, torch):
    data_1 = RandomNodeSplit(num_val=0, num_test=0.3)(data)
    model = DBGNN(num_features=[g.n, g2.n], num_classes=len(data_1.y.unique()), hidden_dims=[16, 32, 8], p_dropout=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_function = torch.nn.CrossEntropyLoss()
    return data_1, loss_function, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following function evaluates the prediction of our model based on the balanced accuracy score for categorical predictions.
    """)
    return


@app.cell
def _(balanced_accuracy_score):
    def test(model, data):
        """Evaluate the model on training and test data."""
        model.eval()

        _, pred = model(data).max(dim=1)

        metrics_train = balanced_accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu().numpy())

        metrics_test = balanced_accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu().numpy())

        return metrics_train, metrics_test

    return (test,)


@app.cell
def _(data_1, loss_function, model, optimizer, test):
    losses = []
    for epoch in range(50):
        output = model(data_1)
        loss = loss_function(output[data_1.train_mask], data_1.y[data_1.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        if epoch % 10 == 0:
            train_ba, test_ba = test(model, data_1)
            print(f'Epoch: {epoch}, Loss: {loss}, Train balanced accuracy: {train_ba}, Test balanced accuracy: {test_ba}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Causality-aware latent space representation of nodes

    We can inspect the model by plotting a latent space representation of the edges generated by the second-order layer of our architecture.
    """)
    return


@app.cell
def _(TSNE, colors, data_1, g2, ho_class_ids, model, pp):
    model.eval()
    _latent = model.higher_order_layers[0].forward(data_1.x_h, data_1.edge_index_higher_order).detach()
    _latent = model.higher_order_layers[1].forward(_latent, data_1.edge_index_higher_order).detach()
    _node_embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(_latent.cpu())
    _embedding_layout = {v: _node_embedding[g2.mapping.to_idx(v)] for v in g2.nodes}
    pp.plot(g2, backend='matplotlib', layout=_embedding_layout, show_labels=False, edge_size=0.3, node_size=3, edge_opacity=0.1, node_color=[colors[i] for i in ho_class_ids])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can further generate latent space representations of the nodes generated by the last bipartite layer of our architecture:
    """)
    return


@app.cell
def _(TSNE, data_1, g, model, pp, style):
    model.eval()
    _latent = model.forward(data_1).detach()
    _node_embedding = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(_latent.cpu())
    _embedding_layout = {v: _node_embedding[g.mapping.to_idx(v)] for v in g.nodes}
    pp.plot(g, backend='matplotlib', layout=_embedding_layout, show_labels=False, edge_size=0.3, node_size=3, edge_opacity=0.1, **style)
    return


if __name__ == "__main__":
    app.run()
