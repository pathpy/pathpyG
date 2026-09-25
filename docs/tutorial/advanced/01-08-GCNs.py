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
    # Graph Convolutional Networks

    *August 4 2026*
    *Training Workshop: Introduction to Deep Graph Learning*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*

    Building on the Neural Message Passing layer implemented in the previous notebook, we now implement a full-fledged graph convolutional neural network and apply it to synthetic and empirical example networks.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import torch_geometric
    from matplotlib import pyplot as plt
    from sklearn.datasets import make_circles
    from sklearn.decomposition import TruncatedSVD

    import pathpyG as pp

    plt.style.use('default')
    sns.set_style("whitegrid")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device)
    return (
        TruncatedSVD,
        make_circles,
        np,
        pd,
        plt,
        pp,
        sns,
        torch,
        torch_geometric,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use a simple example for a network with two strong communities from the lecture, which define our ground truth (binary) node classes. In fact, these two node classes can be easily predicted based on the topology of the graph alone, i.e. here we do not need any node features.
    """)
    return


@app.cell
def _(np, pp, torch):
    n = 50
    random_1 = pp.algorithms.generative_models.watts_strogatz(n=n, s=4, p=0.01, mapping=pp.IndexMap([str(x) for x in range(n)]), undirected=True)
    random_1.data.y = torch.tensor([0] * n)
    random_2 = pp.algorithms.generative_models.watts_strogatz(n=n, s=4, p=0.01, mapping=pp.IndexMap([str(x + 1 * n) for x in range(n)]), undirected=True)
    random_2.data.y = torch.tensor([1] * n)
    colors = {}
    for _v in random_1.nodes:
        colors[_v] = 'orange'
    for _v in random_2.nodes:
        colors[_v] = 'blue'
    network = random_1 + random_2
    network = network.to_undirected()
    network.data.y = torch.cat([random_1.data.y, random_2.data.y])
    src_indices = torch.where(network.data.y < 1)[0]
    tgt_indices = torch.where(network.data.y >= 1)[0]
    num_edges_to_update = 2
    # to_undirected() only preserves edge_index/x/edge_attr, so we have to restore the node labels
    edge_indices_to_update = np.random.choice(network.data.edge_index.shape[1], num_edges_to_update, replace=False)
    for edge_index in edge_indices_to_update:
    # randomly update src and tgt in network.edge_index so that link source in random_1 to target in random_2 and vice-versa
        src, tgt = network.data.edge_index[:, edge_index]
        if src in src_indices and tgt in src_indices:
            new_tgt = np.random.choice(tgt_indices.cpu().numpy())
            network.data.edge_index[1, edge_index] = new_tgt
        elif src in tgt_indices and tgt in tgt_indices:
            new_src = np.random.choice(src_indices.cpu().numpy())
            network.data.edge_index[0, edge_index] = new_src
    pp.plot(network, edge_color='gray', node_color=colors)
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also add a one-hot encoding of the nodes as additional feature:
    """)
    return


@app.cell
def _(network, torch):
    network.data.x = torch.eye(network.n, dtype=torch.float32)
    network.data.y.unsqueeze_(-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now implement the graph convolutional network (GCN). We first reuse the implementation of the neural message passing layer from the previous notebook:
    """)
    return


@app.cell
def _(torch, torch_geometric):
    class GraphConvolution(torch_geometric.nn.MessagePassing):
        """A graph convolution layer following (Kipf, Welling 2017)."""

        def __init__(self, in_ch, out_ch):
            """Initialize the layer parameters."""
            super().__init__(aggr='add')

            # this linear function is used to transform node features 
            # into messages that are then "sent" to neighbors
            self.linear = torch.nn.Linear(in_ch, out_ch)
        
        def forward(self, x, edge_index):
            """Perform a graph convolution.

            This function uses the edges captured in edge_index, performs
            the graph convolution function according to (Kipf, Welling 2017)
            and propagates the transformed features along the edges of the graph.
            """
            # by adding self-loops, we ensure that aggregated messages from neighbors 
            # are combined with information from the node itself
            # this corresponds to matrix $\tilde{A}$ in (Kipf, Welling 2017)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

            # we linearly transform the features of *all* nodes stored in x
            x = self.linear(x)

            # extract source and target nodes of all edges
            source, target = edge_index
        
            # compute the (in-)degrees $d_i$ of source nodes
            deg = torch_geometric.utils.degree(target, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            # with this, the normalization to be applied in the propagation step can be expressed as
            # this corresponds to D^{-0.5} A * D^{-0.5} in (Kipf, Welling 2017)
            norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
        
            # the propagate function propagates messages along the edges of the graph
            # this function internally calls the functions: message(), aggregate() and update()
            # the normalization is applied in the message() function
            return self.propagate(edge_index, x=x, norm=norm)
    
        def message(self, x_j, norm):
            """Construct the message that is sent along each edge."""
            # x_j is a so-called **lifted** tensor which contains the source node features of each edge, 
            # i.e. it has a shape (m, out_ch) where m is the number of edges

            # a call to view(-1, 1) returns a reshaped tensor, where the second dimension 
            # is one and the first dimension is inferred automatically
            return norm.view(-1,1) * x_j

    return (GraphConvolution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As explained before, this class implements the convolution layer based on a single round of message passing. The parameter `in_ch` must correspond to the dimensionality of the node feature vector. By changing the parameter `out_ch` we can use a linear transformation that maps node features to a higher-dimensional vector, which - analogy to the neural graph embeddings introduced in Lecture 10 - we can view as hidden layer that contains a latent representation of nodes.

    We eventually map those latent node representations to the output classes for all nodes. We apply a non-linear transformation to node features and add a second graph convolutional layer, i.e. we run a second round of message passing, where the messages now contain latent representations of nodes that will be used for the classification. The second convolutional layer generates the output for our binary classification problem, i.e. the probability of class 1.
    """)
    return


@app.cell
def _(GraphConvolution, torch, torch_geometric):
    class GCN(torch.nn.Module):
        """A two-layer graph convolutional network."""

        def __init__(self, data: torch_geometric.data.Data, out_ch, hidden_dim=16):
            """Initialize the model parameters."""
            super().__init__()

            # first convolution layer 
            self.input_to_hidden = GraphConvolution(data.num_node_features, hidden_dim)

            # second convolution layer
            self.hidden_to_output =  GraphConvolution(hidden_dim, out_ch)
        
        def forward(self, x, edge_index):
            """Compute class probabilities for all nodes."""
            # first graph convolution -> map nodes to representations in hidden_dim dimensions
            x = self.input_to_hidden(x, edge_index)

            # non-linear activation function
            x = torch.sigmoid(x)

            # second graph convolution -> maps node representations to output classes
            x = self.hidden_to_output(x, edge_index)

            # output class probabilities
            return torch.sigmoid(x)

    return (GCN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are now ready to apply our model to our example. We use a hidden layer to learn a latent node representations in a 16-dimensional space. Due to the use of the one-hot-encoding as node features, each node has 100 features:
    """)
    return


@app.cell
def _(GCN, network):
    model = GCN(network.data, out_ch=1, hidden_dim=16)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's inspect the parameters of our Graph Convolutional Network with 16 hidden dimensions and two graph convolution layers, where the first layer maps the features of the 100 nodes to 16 hidden dimensions and the second layer maps the 16 hidden dimensions to one output per node, that is interpreted as a class probability.

    In the first graph convolution layer, a linear transformation is applied in the message passing and we map the 100-dimensional node features of nodes to 16 dimensions in the first hidden layer. In other words, each node feature is passed through the same linear layer with 100 inputs and 16 outputs. This translates to a $16 \times 100$ weight matrix with additional 16 bias terms.
    """)
    return


@app.cell
def _(model):
    _params = [x for x in model.input_to_hidden.parameters()]
    print(_params[0].shape)
    print(_params[1].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the second layer, we apply a linear layer that maps the 16 hidden dimensions for each node to a single output dimension, that we will interpret as probability of the positive class in our binary classification task. We thus have 16 weight parameters and one additional bias parameter.
    """)
    return


@app.cell
def _(model):
    _params = [x for x in model.hidden_to_output.parameters()]
    print(_params[0].shape)
    print(_params[1].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a network with 100 nodes, we thus have 100 * 16 + 16 + 16 + 1 = 1633 learanble parameters. Without training those parameters, we get similar probabilities
    """)
    return


@app.cell
def _(model, network):
    model.forward(network.data.x, network.data.edge_index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use torch_geometric to generate a training-test split. The `RandomNodeSplit` function creates a training and test mask, that is used by the functions below:
    """)
    return


@app.cell
def _(network, torch_geometric):
    _transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.3, num_test=0)
    data = _transform(network.data)
    print(data)
    return (data,)


@app.cell
def _(data, model, torch):
    out = model.forward(data.x, data.edge_index)
    print(out[80])
    print(data.y[80])

    torch.nn.functional.binary_cross_entropy(out[1], data.y.float()[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following function trains the network and plots the evolution of the accuracy in the training and test set. Since we have a binary classification task, we use binary cross entropy as a loss function.
    """)
    return


@app.cell
def _(GCN, data, torch):
    model_1 = GCN(data, out_ch=1, hidden_dim=4)
    epochs = 100
    _lrn_rate = 0.1
    optimizer = torch.optim.SGD(model_1.parameters(), lr=_lrn_rate)
    return epochs, model_1, optimizer


@app.cell
def _(data, epochs, model_1, np, optimizer, plt, torch):
    _indices = np.arange(100)
    _losses = []
    model_1.train()
    for _epoch in range(epochs):
        _error = 0
        np.random.shuffle(_indices)
        for _i in _indices:
            if data.train_mask[_i]:
                optimizer.zero_grad()
                _output = model_1(data.x, data.edge_index)
                _loss = torch.nn.functional.binary_cross_entropy(_output[_i], data.y.float()[_i])
                _loss.backward()
                optimizer.step()
                _error = _error + _loss.detach().numpy()
        _losses.append(_error)
    plt.plot(range(epochs), _losses)
    return


@app.cell
def _(data, model_1):
    model_1.forward(data.x, data.edge_index)
    return


@app.cell
def _(data, model_1, network, pp):
    _output = model_1.forward(data.x, data.edge_index)
    _prediction = _output.round().long()
    colors_1 = {}
    for _v in network.nodes:
        _index = network.mapping.to_idx(_v)
        if data.val_mask[_index]:
            if _prediction[_index] == 0:
                colors_1[_v] = 'blue'
            else:
                colors_1[_v] = 'orange'
        else:
            colors_1[_v] = 'gray'
    pp.plot(network, edge_color='gray', node_color=colors_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do We Really Need a One-Hot Encoding?

    As discussed above, using a one-hot encoding as node features effectively gives every node its own free, independently learnable embedding vector already in the first layer of the GCN, before any message passing even takes place. This raises a legitimate concern: does our classifier actually learn something about the *topology* of the network, or could it simply be memorizing individual node identities via this per-node lookup, which would not generalize to any node outside of the training graph?

    To test this, we repeat the exact same classification experiment, but replace the 100-dimensional one-hot encoding with a much lower-dimensional **random** feature vector for every node. Since a low-dimensional random vector does not act as a unique identifier for a node (many nodes may even have very similar values), the GCN can no longer rely on a per-node lookup and has to exploit the actual graph topology (via message passing) to succeed.
    """)
    return


@app.cell
def _(GCN, network, np, torch, torch_geometric):
    from sklearn.metrics import balanced_accuracy_score
    results = {}
    for feature_dim in [1, 2, 4, 8, network.n]:
        torch.manual_seed(0)
        if feature_dim == network.n:
            network.data.x = torch.eye(network.n, dtype=torch.float32)
            label = f'one-hot ({network.n}-dim)'
        else:
            network.data.x = torch.rand(network.n, feature_dim)
            label = f'{feature_dim}-dim random'
        _transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.3, num_test=0)
        data_1 = _transform(network.data)
        model_2 = GCN(data_1, out_ch=1, hidden_dim=4)
        optimizer_1 = torch.optim.SGD(model_2.parameters(), lr=0.1)
        _indices = np.arange(network.n)
        model_2.train()
        for _epoch in range(100):
            np.random.shuffle(_indices)
            for _i in _indices:
                if data_1.train_mask[_i]:
                    optimizer_1.zero_grad()
                    _output = model_2(data_1.x, data_1.edge_index)
                    _loss = torch.nn.functional.binary_cross_entropy(_output[_i], data_1.y.float()[_i])
                    _loss.backward()
                    optimizer_1.step()
        _output = model_2.forward(data_1.x, data_1.edge_index)
        _prediction = _output.round().long()
        val_acc = balanced_accuracy_score(data_1.y[data_1.val_mask].cpu().numpy(), _prediction[data_1.val_mask].cpu().numpy())
        results[label] = val_acc
        print(f'{label:>20}: validation balanced accuracy = {val_acc:.3f}')
    return data_1, model_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The exact numbers vary a bit between runs (this is a small, 100-node graph trained with plain SGD for a fixed, fairly small number of epochs, so individual results are noisy), but the overall pattern is clear: already with a handful of random input dimensions, dramatically fewer than the 100 dimensions required for a one-hot encoding, the GCN reaches an accuracy that is close to (and sometimes on par with) the one-hot baseline. A single random dimension is occasionally enough to beat chance level, but is not reliable; by around 8 random dimensions, performance consistently approaches the one-hot result.

    This confirms that the classification signal is coming from the *topology* of the graph, propagated through message passing, and not from memorized per-node identities: a low-dimensional random vector cannot possibly serve as a unique lookup key for all 100 nodes, so the only way for the model to succeed is by aggregating and comparing feature information along the edges of the graph. A one-hot encoding is a convenient choice exactly because it requires no real attribute data and trivially provides every node with a distinct input, but as discussed above this comes at the cost of not scaling to large graphs and not generalizing to nodes outside of the training graph. Low-dimensional random features avoid both of these problems, though in practice we of course prefer real, informative node attributes whenever they are available, as we will see in the next notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viusalizing feature maps

    We can visualize the latent representations generated by the first hidden layer of our model.
    """)
    return


@app.cell
def _(data_1, model_2):
    embedding = model_2.input_to_hidden.forward(data_1.x, data_1.edge_index)
    print(embedding)
    return (embedding,)


@app.cell
def _(TruncatedSVD, data_1, embedding, network, plt):
    _svd = TruncatedSVD()
    _low_dim = _svd.fit_transform(embedding.detach().numpy())
    colors_2 = {}
    for _v in network.nodes:
        _index = network.mapping.to_idx(_v)
        if data_1.val_mask[_index]:
            colors_2[_index] = 'grey'
        elif data_1.y[_index] == 0:
            colors_2[_index] = 'blue'
        else:
            colors_2[_index] = 'orange'
    plt.scatter(_low_dim[:, 0], _low_dim[:, 1], c=colors_2.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application to network with non-linear pattern in latent space

    We can now use this model to classify nodes in a synthetic example network generated based on a non-linear pattern in a two-dimensional latent space. We reuse the example from Lecture 09, i.e. we generate network based on the circles data set in sklearn.
    """)
    return


@app.cell
def _(make_circles, pd, plt, sns):
    x, c = make_circles(n_samples=200, noise=0.05, factor=0.5)
    data_2 = pd.DataFrame({'x0': x[:, 0], 'x1': x[:, 1], 'y_class': c})
    sns.scatterplot(x='x0', y='x1', data=data_2, hue='y_class')
    plt.xlabel('$X_0$', fontsize=16)
    plt.ylabel('$X_1$', fontsize=16)
    return c, data_2


@app.cell
def _(np):
    def euclidean_dist(x, y):
        """Return the Euclidean distance between two points."""
        return np.linalg.norm(x-y)

    def soft_rule(dist, **kwargs):
        """Randomly decide whether to connect two nodes at the given distance."""
        p = kwargs['beta'] * np.exp(-dist/kwargs['alpha'])
        if np.random.random() <= p:
            return True
        else:
            return False

    return euclidean_dist, soft_rule


@app.cell
def _(c, data_2, euclidean_dist, np, pp, soft_rule, torch):
    nodes = []
    pos = {}
    g = {}
    cluster = []
    for _i, row in data_2.iterrows():
        nodes.append(str(_i))
        pos[str(_i)] = np.array([row['x0'], row['x1']])
        g[str(_i)] = c[_i]
        cluster.append(torch.tensor([c[_i]]))
    edges = []
    for _v in nodes:
        for w in nodes:
            if soft_rule(euclidean_dist(pos[_v], pos[w]), alpha=0.05, beta=50) and _v != w:
                edges.append((_v, w))
    net = pp.Graph.from_edge_list(edges).to_undirected()
    net.data.y = torch.tensor(cluster).unsqueeze_(-1)
    net.data.x = torch.eye(net.n, dtype=torch.float32)
    print(net.data)
    return g, net, nodes


@app.cell
def _(g, net, nodes, pp):
    colors_3 = {}
    colors_3[0] = 'CornflowerBlue'
    colors_3[1] = 'orange'
    for _v in nodes:
        colors_3[_v] = colors_3[g[_v]]
    pp.plot(net, edge_color='grey', node_color=colors_3)
    return (colors_3,)


@app.cell
def _(net, torch_geometric):
    _transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.3, num_test=0)
    data_3 = _transform(net.data)
    return (data_3,)


@app.cell
def _(GCN, net, torch):
    model_3 = GCN(net.data, out_ch=1, hidden_dim=4)
    epochs_1 = 100
    _lrn_rate = 0.1
    optimizer_2 = torch.optim.SGD(model_3.parameters(), lr=_lrn_rate)
    return epochs_1, model_3, optimizer_2


@app.cell
def _(data_3, epochs_1, model_3, net, np, optimizer_2, plt, torch):
    _indices = np.arange(net.n)
    _losses = []
    model_3.train()
    for _epoch in range(epochs_1):
        _error = 0
        np.random.shuffle(_indices)
        for _i in _indices:
            if data_3.train_mask[_i]:
                optimizer_2.zero_grad()
                _output = model_3(net.data.x, net.data.edge_index)
                _loss = torch.nn.functional.binary_cross_entropy(_output[_i], data_3.y.float()[_i])
                _loss.backward()
                optimizer_2.step()
                _error = _error + _loss.detach().numpy()
        _losses.append(_error)
    plt.plot(range(epochs_1), _losses)
    return


@app.cell
def _(colors_3, data_3, model_3, net, pp):
    _output = model_3.forward(net.data.x, net.data.edge_index)
    _prediction = _output.round().long()
    _node_cols = {}
    for _v in net.nodes:
        _index = net.mapping.to_idx(_v)
        if data_3.val_mask[_index]:
            _node_cols[_v] = colors_3[_prediction[_index].item()]
        else:
            _node_cols[_v] = 'gray'
    pp.plot(net, edge_color='gray', node_color=_node_cols)
    return


@app.cell
def _(TruncatedSVD, colors_3, data_3, model_3, net, plt):
    embedding_1 = model_3.input_to_hidden.forward(data_3.x, data_3.edge_index)
    _svd = TruncatedSVD()
    _low_dim = _svd.fit_transform(embedding_1.detach().numpy())
    color_points = {}
    for _v in net.nodes:
        _index = net.mapping.to_idx(_v)
        if data_3.val_mask[_index]:
            color_points[_index] = 'grey'
        else:
            color_points[_index] = colors_3[net.data.y[_index].item()]
    plt.scatter(_low_dim[:, 0], _low_dim[:, 1], c=color_points.values())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Application to empirical network
    """)
    return


@app.cell
def _(pp, torch):
    karate = pp.io.read_netzschleuder_graph(name='karate', network='78')
    karate.data.y = torch.tensor(karate.data.node_groups-1, dtype=torch.float)
    karate.data.y.unsqueeze_(-1)
    karate.data.x = torch.eye(karate.n, dtype=torch.float32)
    print(karate)
    pp.plot(karate, edge_color='gray',  node_color=[karate["node_groups", v].item() for v in karate.nodes])
    return (karate,)


@app.cell
def _(karate, torch_geometric):
    _transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.3, num_test=0)
    data_4 = _transform(karate.data)
    return (data_4,)


@app.cell
def _(GCN, data_4, torch):
    model_4 = GCN(data_4, out_ch=1, hidden_dim=4)
    epochs_2 = 200
    _lrn_rate = 0.1
    optimizer_3 = torch.optim.SGD(model_4.parameters(), lr=_lrn_rate)
    return epochs_2, model_4, optimizer_3


@app.cell
def _(data_4, epochs_2, karate, model_4, np, optimizer_3, plt, torch):
    _indices = np.arange(karate.n)
    _losses = []
    model_4.train()
    for _epoch in range(epochs_2):
        _error = 0
        np.random.shuffle(_indices)
        for _i in _indices:
            if data_4.train_mask[_i]:
                optimizer_3.zero_grad()
                _output = model_4(data_4.x, data_4.edge_index)
                _loss = torch.nn.functional.binary_cross_entropy(_output[_i], data_4.y[_i])
                _loss.backward()
                optimizer_3.step()
                _error = _error + _loss.detach().numpy()
        _losses.append(_error)
    plt.plot(range(epochs_2), _losses)
    return


@app.cell
def _(data_4, karate, model_4, pp):
    _output = model_4.forward(data_4.x, data_4.edge_index)
    _prediction = _output.round().long()
    _node_cols = {}
    for _v in karate.nodes:
        _index = karate.mapping.to_idx(_v)
        if data_4.val_mask[_index]:
            if _prediction[_index] == 0:
                _node_cols[_v] = 'CornflowerBlue'
            else:
                _node_cols[_v] = 'orange'
        else:
            _node_cols[_v] = 'gray'
    pp.plot(karate, edge_color='gray', node_color=_node_cols)
    return


@app.cell
def _(TruncatedSVD, data_4, karate, model_4, plt):
    embedding_2 = model_4.input_to_hidden.forward(data_4.x, data_4.edge_index)
    _low_dim = TruncatedSVD().fit_transform(embedding_2.detach().numpy())
    colors_4 = {}
    for _v in karate.nodes:
        _index = karate.mapping.to_idx(_v)
        if data_4.y[_index].item() == 0:
            colors_4[_index] = 'CornflowerBlue'
        else:
            colors_4[_index] = 'orange'
    plt.scatter(_low_dim[:, 0], _low_dim[:, 1], c=colors_4.values())
    return


if __name__ == "__main__":
    app.run()
