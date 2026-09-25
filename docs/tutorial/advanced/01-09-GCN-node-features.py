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
    # Node Classification with Additional Node Features

    *August 4 2026*
    *Training Workshop: Introduction to Deep Graph Learning*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*

    In the previous notebooks, we have considered networks where nodes did not have additional features, i.e. we used a one-hot encoding of the nodes as input to our GCN. With this approach, the GCN can only learn based on patterns contained in the graph topology. But what if we have additional features that could help the GCN, e.g., to classify nodes? Here we demonstrate this important capability of graph convolutional networks.
    """)
    return


@app.cell
def _():
    import numpy as np
    import seaborn as sns
    import torch
    import torch_geometric
    from matplotlib import pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    from torch.nn.functional import one_hot as ohe

    import pathpyG as pp

    plt.style.use('default')
    sns.set_style("whitegrid")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device)
    return TruncatedSVD, np, ohe, plt, pp, torch, torch_geometric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first use `pathpyG` to generate the synthetic network shown on slide 29.
    """)
    return


@app.cell
def _(np, ohe, pp, torch):
    n = 50

    random_1 = pp.algorithms.generative_models.watts_strogatz(n=n, s=4, p=0.01, mapping=pp.IndexMap([str(x) for x in range(n)]), undirected=True)
    random_2 = pp.algorithms.generative_models.watts_strogatz(n=n, s=4, p=0.01, mapping=pp.IndexMap([str(x+1*n) for x in range(n)]), undirected=True)

    network = random_1 + random_2
    network = network.to_undirected()

    # node features, cluster labels, and targets are assigned after merging and undirecting the
    # graph, since to_undirected() does not preserve custom node attributes set beforehand
    network.data.x = torch.cat((torch.tensor([0.]*int(n/2)), torch.tensor([1.]*int(n/2)),
                                 torch.tensor([0.]*int(n/2)), torch.tensor([1.]*int(n/2))))
    network.data.cluster = torch.cat((torch.tensor([0]*int(n/2)), torch.tensor([1]*int(n/2)),
                                       torch.tensor([2]*int(n/2)), torch.tensor([3]*int(n/2))))
    network.data.y = torch.cat((torch.tensor([[0,0]]*int(n/2)), torch.tensor([[0,1]]*int(n/2)),
                                 torch.tensor([[1,0]]*int(n/2)), torch.tensor([[1,1]]*int(n/2)))).float()

    # randomly update src and tgt in network.edge_index so that link source in random_1 to target in random_2 and vice-versa
    src_indices = torch.where(network.data.cluster < 2)[0]
    tgt_indices = torch.where(network.data.cluster >= 2)[0]
    num_edges_to_update = 10
    edge_indices_to_update = np.random.choice(network.data.edge_index.shape[1], num_edges_to_update, replace=False)
    for edge_index in edge_indices_to_update:
        src, tgt = network.data.edge_index[:, edge_index]
        if src in src_indices and tgt in src_indices:
            new_tgt = np.random.choice(tgt_indices.cpu().numpy())
            network.data.edge_index[1, edge_index] = new_tgt
        elif src in tgt_indices and tgt in tgt_indices:
            new_src = np.random.choice(src_indices.cpu().numpy())
            network.data.edge_index[0, edge_index] = new_src

    network.data.x = torch.cat((network.data.x.unsqueeze(1), ohe(torch.arange(0,2*n))), dim=1)

    pp.plot(network, edge_color='gray');
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We reuse the implementation of the GCN from the previous notebook:
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

            # extract the source and target nodes of all edges
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


@app.cell
def _(GraphConvolution, torch, torch_geometric):
    class GCN(torch.nn.Module):
        """A two-layer graph convolutional network."""

        def __init__(self, data: torch_geometric.data.Data, out_ch, hidden_dim=16):
            """Initialize the model parameters."""
            super().__init__()

            self.input_to_hidden = GraphConvolution(data.num_node_features, hidden_dim)
            self.hidden_to_output =  GraphConvolution(hidden_dim, out_ch)
        
        def forward(self, x, edge_index):
            """Compute class probabilities for all nodes."""
            # first graph convolution -> map nodes to representations in hidden_dim dimensions
            x = self.input_to_hidden(x, edge_index)

            # non-linear activation function
            x = torch.sigmoid(x)

            # graph convolution -> maps node representations to output classes
            x = self.hidden_to_output(x, edge_index)

            return torch.sigmoid(x)

    return (GCN,)


@app.cell
def _(network, torch_geometric):
    transform = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.3, num_test=0)
    data = transform(network.data)

    print(data)
    return (data,)


@app.cell
def _(GCN, data, torch):
    model = GCN(data, out_ch=2, hidden_dim=4)

    epochs = 200
    lrn_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate)
    return epochs, model, optimizer


@app.cell
def _(data, epochs, model, network, np, optimizer, plt, torch):
    indices = np.arange(100)
    losses = []
    model.train()
    for epoch in range(epochs):
        error = 0
        print(epoch)
        np.random.shuffle(indices)
        for i in indices:
            if data.train_mask[i]:
                optimizer.zero_grad()
                _output = model(network.data.x, network.data.edge_index)
                loss = torch.nn.functional.binary_cross_entropy(_output[i], data.y[i])
                loss.backward()
                optimizer.step()
                error += loss.detach().numpy()
        losses.append(error)  # set gradients to zero
    # plot evolution of loss function
    plt.plot(range(epochs), losses)  # compute loss function for training sample and backpropagate  # update parameters
    return


@app.cell
def _(model, network):
    _output = model.forward(network.data.x, network.data.edge_index)
    prediction = _output.round().long()
    # we efficiently map probabilities to classes by rounding values to the 
    # nearest integer, i.e. we obtain class 0 for probabilities smaller than 0.5
    # and class 1 for probabilities larger than 0.5
    print(prediction)
    return (prediction,)


@app.cell
def _(data, network, pp, prediction, torch):
    _colors = {}
    for _v in network.nodes:
        _index = network.mapping.to_idx(_v)
        if data.val_mask[_index]:
            if torch.equal(prediction[_index], torch.tensor([0, 0])):
                _colors[_v] = 'blue'
            elif torch.equal(prediction[_index], torch.tensor([0, 1])):
                _colors[_v] = 'orange'
            elif torch.equal(prediction[_index], torch.tensor([1, 0])):
                _colors[_v] = 'magenta'
            else:
                _colors[_v] = 'cyan'
        else:
            _colors[_v] = 'gray'
    pp.plot(network, edge_color='gray', node_color=_colors)
    return


@app.cell
def _(TruncatedSVD, data, model, network, plt):
    embedding = model.input_to_hidden.forward(network.data.x, network.data.edge_index)
    svd = TruncatedSVD()
    low_dim = svd.fit_transform(embedding.detach().numpy())
    _colors = {}
    for _v in network.nodes:
        _index = network.mapping.to_idx(_v)
        if data.cluster[_index] == 0:
            _colors[_index] = 'blue'
        elif data.cluster[_index] == 1:
            _colors[_index] = 'orange'
        elif data.cluster[_index] == 2:
            _colors[_index] = 'magenta'
        else:
            _colors[_index] = 'cyan'
    plt.clf()
    plt.scatter(low_dim[:, 0], low_dim[:, 1], c=_colors.values())
    return


if __name__ == "__main__":
    app.run()
