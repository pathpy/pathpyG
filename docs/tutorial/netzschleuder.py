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
    # Learning in Graphs from the Netzschleuder Repository

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

    Access to a large number of graphs with different topological characteristics and from different domains is crucial for the development and evaluation of graph learning methods. Thousands of graph data sets are available scattered throughout the web, possibly using different data formats and with missing information on their actual origin. Addressing this issue the [Netschleuder Online Repository](https://networks.skewed.de/) by Tiago Peixoto provides a single repository of graphs in a single format, including descriptions, citations, and node-/edge- or graph-level meta-data. To facilitate the development of graph learning techniques, pathpyG provides a feature that allows to directly read networks from the netzschleuder repository via an API.

    In this brief unit, we will learn how we can retrieve network records and graph data from the netzschleuder repository. We will further demonstrate how we can conveniently apply a Graph Neural Network to predict node-level categories contained in the meta-data.

    We first need to import a few modules.
    """)
    return


@app.cell
def _():
    import torch
    import torch_geometric
    from matplotlib import pyplot as plt
    from sklearn import metrics
    from sklearn.decomposition import TruncatedSVD
    from torch.nn import ReLU, Sigmoid
    from torch_geometric.nn import GCNConv, Sequential

    import pathpyG as pp

    return (
        GCNConv,
        ReLU,
        Sequential,
        Sigmoid,
        TruncatedSVD,
        metrics,
        plt,
        pp,
        torch,
        torch_geometric,
    )


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading graphs from the netzschleuder repository

    In the `pathpy.io` module, there is a function that allows to read graph data from the API.

    We can read a given networks from the netzschleuder database using its record name. Just browse the [Netschleuder Online Repository](https://networks.skewed.de/) to find the record names. As an example, we use a graph capturing co-purchase relationships between political books.
    """)
    return


@app.cell
def _(pp):
    g = pp.io.read_netzschleuder_graph(name='polbooks')
    g.mapping = pp.IndexMap(g.data.node_label)
    print(g)
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot this temporal graph in an interactive way:
    """)
    return


@app.cell
def _(g, pp):
    pp.plot(g);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see how we can apply GNNs to attributed graphs, let us read the famous karate club network. The record `karate` actually contains two networks with labels `77` and `78`, which refer to two different versions of the data with different numbers of edges. If multiple graph data sets exist in the same record, we can specify the name of the network as second argument.
    """)
    return


@app.cell
def _(device, pp):
    g_1 = pp.io.read_netzschleuder_graph(name='karate', network='78').to(device)
    print(g_1)
    return (g_1,)


@app.cell
def _(g_1, pp):
    pp.plot(g_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that the nodes actually have a `node_groups` property, which maps the nodes to two groups. Those groups are often used as `ground truth` for communities in this simple illustrative graph. We will instead use it as ground truth categorical node label for a node classification experiment based on a Graph Neural Network.

    Conveniently, numerical node attributes (either scalar or vector values) are automatically converted to torch tensors, so we can directly use them for a GNN.
    """)
    return


@app.cell
def _(g_1):
    print(g_1.data.node_groups)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For convenience, let us shift the group labels to binary values 0 and 1:
    """)
    return


@app.cell
def _(g_1):
    g_1.data.node_groups = g_1.data.node_groups - 1
    print(g_1.data.node_groups)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot categorical labels by passing node colors in the plot function.
    """)
    return


@app.cell
def _(g_1, pp):
    pp.plot(g_1, node_color=[g_1['node_groups', v].item() for v in g_1.nodes])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can apply custom colors to the two binary groups of nodes:
    """)
    return


@app.cell
def _(g_1, pp):
    color_map = {0: 'red', 1: 'blue'}
    _colors = [color_map[g_1['node_groups', v].item()] for v in g_1.nodes]
    pp.plot(g_1, node_color=_colors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Applying Graph Neural Networks to Netzschleuder Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To simplify the application of deep learning models, we can retrieve a data object that contains the graph and its attributes:
    """)
    return


@app.cell
def _(g_1):
    print(g_1.data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's use a one-hot encoding of nodes as a simple additional node feature `x`, and let's use the node groups as target label `y`.
    """)
    return


@app.cell
def _(device, g_1, torch):
    data = g_1.data
    g_1['node_feature'] = torch.eye(g_1.n, device=device)
    data['x'] = data['node_feature']
    data['y'] = data['node_groups'].reshape(-1, 1).float()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is easy to define a Graph Convolutional Network that ues the one-hot-encodings of nodes and the topology to predict binary node labels:
    """)
    return


@app.cell
def _(GCNConv, ReLU, Sequential, Sigmoid, data, device):
    model = Sequential('node_ohe, edge_index', [
        (GCNConv(in_channels=data.num_node_features, out_channels=8), 'node_ohe, edge_index -> hidden'),
        ReLU(inplace=True),
        (GCNConv(in_channels=8, out_channels=1), 'hidden, edge_index -> output'),
        Sigmoid(),
    ])
    model.to(device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We next apply a `RandomNodeSplit` transformation to split the nodes in a training and test set.
    """)
    return


@app.cell
def _(data, torch_geometric):
    transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.5, num_test=0)
    data_1 = transform(data)
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then train our model for 200 epochs on the training set.
    """)
    return


@app.cell
def _(data_1, model, plt, torch):
    epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_1.x, data_1.edge_index)
        loss = torch.nn.functional.binary_cross_entropy(out[data_1.train_mask], data_1.y[data_1.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    plt.plot(range(epochs), losses)
    plt.grid()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We evaluate the model in the test set and calculate the adjusted mutual information for the ground truth.
    """)
    return


@app.cell
def _(data_1, metrics, model):
    model.eval()
    predicted_groups = model(data_1.x, data_1.edge_index).round().long()
    metrics.adjusted_mutual_info_score(data_1.y[data_1.test_mask].squeeze().cpu().numpy(), predicted_groups[data_1.test_mask].squeeze().cpu().numpy())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We visualize node representations learned by the model. The test nodes are colored, while training nodes are greyed out.
    """)
    return


@app.cell
def _(TruncatedSVD, data_1, g_1, model, plt):
    embedding = model[0].forward(data_1.x, data_1.edge_index)
    svd = TruncatedSVD()
    low_dim = svd.fit_transform(embedding.cpu().detach().numpy())
    _colors = {}
    for v in range(g_1.n):
        if not data_1.val_mask[v]:
            _colors[v] = 'grey'
        elif data_1.y[v].item() == 0.0:
            _colors[v] = 'blue'
        else:
            _colors[v] = 'orange'
    plt.scatter(low_dim[:, 0], low_dim[:, 1], c=_colors.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This simple code gives you thousands of networks with various meta information at your fingertips, to wich you can directly apply graph learning models provided in pyG, or deep graoh learning architectures defined by yourself.
    """)
    return


if __name__ == "__main__":
    app.run()
