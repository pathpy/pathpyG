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
    # Neural Message Passing in `pytorch-geometric`

    *August 4 2026*
    *Training Workshop: Introduction to Deep Graph Learning*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*

    We introduce neural message passing, a fundamental building block of Graph Neural Networks, and show how we can efficiently implement it based on the geometric deep learning package `pytorch-geometric`, which is commonly referred to as `pyg`.
    """)
    return


@app.cell
def _():
    import seaborn as sns
    import torch
    import torch_geometric
    from matplotlib import pyplot as plt

    import pathpyG as pp

    plt.style.use('default')
    sns.set_style("whitegrid")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device)
    return pp, torch, torch_geometric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We build our example networks with `pathpy`. In pathpy, we can store arbitrary attributes at the level of nodes, edges and graphs. We first implement a method that converts `pathpy` networks and their attributes to `torch` tensor representations that can be used with `torch-geometric`.

    For the following example network with three nodes, we add two attributes `x` and `y` to the nodes. Following the usual convention, the attribute `x` contains a **node feature tensor**. In our simple toy example from the lecutre, we use a one-dimensional tensor that contains a single number. The `y` attribute contains a tensor with a **node-level target variable**. For the purpose of illustration, we use two target values per node.
    """)
    return


@app.cell
def _(pp, torch):
    n = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'c')]).to_undirected()

    n.data.x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
    n.data.y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1]])

    pp.plot(n, edge_color='gray');
    return (n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As shown in notebook 01, in `pathpyG` networks are internally stored as torch tensors that represent the `edge_index` of the graph, which makes it easy to apply messsage passing with `pyG`
    """)
    return


@app.cell
def _(n):
    print(n.data.edge_index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, any additional features are stored as tensors:
    """)
    return


@app.cell
def _(n):
    n.data.x
    return


@app.cell
def _(n):
    n.data.y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The package `torch_geometric` (also called `pyG`, which is short for "pyTorch Geometric" provides special data types, functions, and classes that support deep learning on non-Euclidean, i.e. geometric data like graphs). To simplify the handling of graph data comprised of edge indices with associated node-, edge-, or graph-level attributes, `pyG` provides the class `torch_geometric.data.Data`. Each `pathpyG` network instance has a `Data` object that contains the edge index as well as any node features and/or attributes:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us try this for our example. Printing the data object will give the shape of features, edge index, and target labels. Edge index, node features and target labels can be accessed via the respective attributes of the `Data` instance.
    """)
    return


@app.cell
def _(n):
    print(n.data)

    print(n.data.x)
    print(n.data.edge_index)
    print(n.data.y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementing Message Passing in pyG

    We are now ready to generalize the idea of convolutional neural networks to graphs. As we have seen in notebook 07, image convolutions are implemented by adding each pixel of an image to the pixel values of the neighboring pixels and additionally applying an appropriate weighting scheme. This can be generalized to networks, where we can apply a similar scheme to nodes and their neighbors.

    This can be expressed as a message passing algorithm, where in each step of the convolution nodes exchange their current feature values with their neighbors. Each node then takes the incoming "messages" from its neighbors, aggregates them and computes a new local value. Depending on how much information from the neighborhood we wish to incorporate in the learning process, this convolution can be repeated multiple times (just as in convolutional neural networks).

    While we could implement message passing on our own, `torch_geometric` conveniently provides a base class that simplifies this process (and makes it fast by calculating it at the GPU). The process of implementing a graph convolutional layer based on the `MessagePassing` base class is explained in [detail here](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html). Note that there are many different ways to implement the message passing (using different aggregation rules, weighting schemes, or normalizations) and depending on the specific implementation we obtain different graph neural networks.

    Following the lecture, here we focus on the approach proposed by [Kipf and Welling in 2017](https://arxiv.org/abs/1609.02907), which is a more or less direct translation of image convolutions to graphs. Using the notation from the lecture, the message passing process can be mathematically described via the following update rule for the node features $h_i^{(k)}$ in step $k$ of the message passing:

    $$ h_i^{(k)}  := \sigma \left(\mathbf{W}^{(k)}\sum_{j \in N_i \cup \{i\}} \frac{\left(h_j^{(k-1)}\right)}{\sqrt{d_i\cdot d_j}}\right)$$

    Here $h_i^{(k)}$ is the feature of node $i$ in step $k$, $d_i$ is the degree of node $i$, $N_i$ is the set of neighbors of node $i$, $\mathbf{A}^{(k)}$ are learnable weight parameters and $\sigma$ is a non-linear activation function.

    This can be viewed as a localized, graph-based generalization of the convolution commonly applied to image data (which can be viewed as a grid graph).

    We will implement this step by step, starting with a simple aggregation of the neighbor features, i.e. we initially do not apply the symmetric degree-normalization as well as the multiplication with the weight matrix. We further do not include self-loops. We thus start with an update rule that simply adds the features of all neighbors.

    $$ h_i^{(k)}  :=\sum_{j \in N_i} h_j^{(k-1)}$$

    In `pyG` we can implement this via a message passing class that derived from the `torch_geometric.nn.MessagePassing` base class. In the constructor of the base class, we can set the aggregation function that should be used by the nodes receiving messages (i.e. features) from their neighbors.

    We basically have to implement two functions:

    - `forward` is analogous to the forward function in a `torch.nn.Module`, i.e. here we specify the function that is applied when a tensor `x` with associated edge index `edge_index` is passed as input. We call the `self.propagate` function, which will trigger a single round of message passing along all edges in the graph.
    - `message` constructs the message to be sent along all directed edges $(j,i)$ from node $j$ to node $i$. For a tensor x that is passed as an argument to the propagate function, we can use special variables "x_i" and "x_j" to extract the tensor associated with nodes i and j for all edges (j,i). Hence, x_j and x_i have shape [m, d] where m is the number of edges and d is the dimensionality of the node features x

    The simple update rule outlined above can be implemented as follows:
    """)
    return


@app.cell
def _(torch_geometric):
    class MP(torch_geometric.nn.MessagePassing):
        """A message passing layer that sums the features of neighboring nodes."""

        def __init__(self):
            """Initialize the layer parameters."""
            super().__init__(aggr='add')
    
        def forward(self, x, edge_index):
            """Perform a single round of message passing."""
            # this triggers a single round of message passing, where the message is passed along all edges in the network
            return self.propagate(edge_index, x=x)

        def message(self, x_j):
            """Construct the message that is sent along each edge."""
            # This method constructs the message to be sent along all directed edges (j,i) from j to i
            # For a tensor x that is passed as an argument to the propagate function, we can 
            # use special variables "x_i" and "x_j" to extract the tensor associated with nodes i 
            # and j for all edges (j,i). Hence, x_j and x_i have shape [m, d] where m is the number 
            # of edges and d is the dimensionality of the node features x

            # Here we simply return a message that contains the feature of the source node j
            print('Message = ', x_j.t())
            return x_j

    return (MP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now try this. Here is the feature vector that we use to initialize the node states in the message passing:
    """)
    return


@app.cell
def _(n):
    print(n.data.x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is the graph:
    """)
    return


@app.cell
def _(n, pp):
    pp.plot(n, edge_color='gray');
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now apply one round of message passing and print the result:
    """)
    return


@app.cell
def _(MP, n):
    mp = MP()
    _x = mp.forward(n.data.x, n.data.edge_index)
    print(_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also use a `mean` aggregation, which corresponds to the following update rule:

    $$ h_j^{(k)}  = \frac{1}{d_i} \sum_{j \in N_i} h_j^{(k-1)}$$
    """)
    return


@app.cell
def _(torch_geometric):
    class MP_1(torch_geometric.nn.MessagePassing):
        """A message passing layer that averages the features of neighboring nodes."""

        def __init__(self):
            """Initialize the layer parameters."""
            super().__init__(aggr='mean')

        def forward(self, x, edge_index):
            """Perform a single round of message passing."""
            return self.propagate(edge_index, x=x)  # this triggers a single round of message passing along all edges in the network

        def message(self, x_j):
            """Construct the message that is sent along each edge."""
            print('Message = ', x_j.t())
            return x_j  # This method constructs the message to be sent to nodes i for all edges (j,i)  # For a tensor x that is passed as an argument to the propagate function, we can  # use special variables "x_i" and "x_j" to extract the tensor associated with nodes i  # and j for all edges (j,i). Hence, x_j and x_i have shape [m, d] where m is the number  # of edges and d is the dimensionality of the node features x

    return (MP_1,)


@app.cell
def _(MP_1, n):
    mp_1 = MP_1()
    _x = mp_1.forward(n.data.x, n.data.edge_index)
    print(_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In both examples, we note that the nodes overwrite their own features, i.e. the values are replaced with the (aggregate) value received from the neighbors. To avoid this, we can add self-loops, which means that each node also sends a message to itself. This can be done by adding self-loops to the `edge_index` before running the message passing. We can use the convenience function `torch_geometric.utils.add_self_loops` as follows:
    """)
    return


@app.cell
def _(torch_geometric):
    class MP_2(torch_geometric.nn.MessagePassing):
        """A message passing layer that sums the features of neighbors and of the node itself."""

        def __init__(self):
            """Initialize the layer parameters."""
            super().__init__(aggr='add')

        def forward(self, x, edge_index):
            """Perform a single round of message passing."""
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))  # add self-loops so that nodes include their own features in the aggregation
            return self.propagate(edge_index, x=x)

        def message(self, x_j):  # this triggers a single round of message passing along all edges in the network
            """Construct the message that is sent along each edge."""
            print('Message = ', x_j.t())
            return x_j  # This method constructs the message to be sent to nodes i for all edges (j,i)  # For a tensor x that is passed as an argument to the propagate function, we can  # use special variables "x_i" and "x_j" to extract the tensor associated with nodes i  # and j for all edges (j,i). Hence, x_j and x_i have shape [m, d] where m is the number  # of edges and d is the dimensionality of the node features x

    return (MP_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now find that the features of neighbours are added to the local features of the nodes:
    """)
    return


@app.cell
def _(MP_2, n):
    mp_2 = MP_2()
    _x = mp_2.forward(n.data.x, n.data.edge_index)
    print(_x)
    return (mp_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We finally add the two missing ingredients for a graph convolutional layer, which is the normalization by the product of the square root of the node degrees as well as the linear transformation with learnable weights. For this, we first confirm our intuition that - so far - our message passing model actually has no parameters that we could fit to the data.
    """)
    return


@app.cell
def _(mp_2):
    print([x for x in mp_2.parameters()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To understand the implementation of the convolution class, it is helpful to consider the degree-based normalization step by step. Let's first split the `edge_index` into two tensors that take the indices of source and target nodes:
    """)
    return


@app.cell
def _(n):
    source, target = n.data.edge_index
    print(source)
    print(target)
    return source, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the `torch_geometric.utils.degree` to calculate node degrees based on a node index tensor. This method assumes that we pass a tensor that contains the source or target nodes of an edge index, which means that the in- or out-degree can simply be calculated by counting how often the indices occur.

    For an undirected network, this method will calculate the degree of all nodes, independent of whether we pass the source or target nodes (since all links exist in both directions). So the following code calculates the degree sequence of our network:
    """)
    return


@app.cell
def _(n, target, torch, torch_geometric):
    deg = torch_geometric.utils.degree(target, n.data.x.size(0), dtype=torch.float32)
    print(deg)
    return (deg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By indexing the degree tensor with the indices of source and target nodes, we can calculate the (in-)degrees of source and target nodes for all directed edges in the network.
    """)
    return


@app.cell
def _(deg, source, target):
    print(deg[source])
    print(deg[target])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Apart from an edge index and a tensor with the node features, the `propagate` function of the `MessagePassing` base class accepts an argument `norm`. It is a tensor of shape (m,1), whose entries capture the normalization factors that are applied in the exchange of all messages across all $m$ edges of the network. The symmetric degree-based normalization term

    $$ \frac{1}{\sqrt{d_i} \cdot \sqrt{d_j}} $$

    can be implemented as follows (additionally accounting for nodes with zero degree): The normalization factor for a directed edge (a,b) is

    $$ \frac{1}{\sqrt{d_a} \cdot \sqrt{d_b}} = \frac{1}{\sqrt{2 \cdot 2}} := \frac{1}{2} $$

    The normalization factor for directed edges (b,c) and (a,c) are

    $$ \frac{1}{\sqrt{d_b} \cdot \sqrt{d_c}} = \frac{1}{\sqrt{4 \cdot 2}} = \frac{1}{2 \cdot \sqrt{2}}  \approx 0.3536 $$
    """)
    return


@app.cell
def _(deg, source, target):
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
    print(norm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now put all pieces together: We include self-loops, the linear transformation of node features using an instance of `torch.nn.Linear`, as well as the symmetric degree-based normalization. Conveniently, we can pass a normalization tensor (with one entry per directed edge) to the `propagate` function. We then apply this during the actual message passing, implemented in the body of `message`. This is done by multiplying the transposed normalization tensor (which we compute as `view(-1,1)`) with the node features. Note that in the `message` function node features are automatically **lifted** to the level of edges, i.e. `x_j` has shape $(m,1)$ and includes the node features of all source nodes for all (directed) edges. This allows us to simply multiply the transposed `norm` tensor (shape $(1,m)$) with `x_j` (shape $(m,1$):
    """)
    return


@app.cell
def _(torch, torch_geometric):
    class GraphConvolution(torch_geometric.nn.MessagePassing):
        """A graph convolution layer following (Kipf, Welling 2017)."""

        def __init__(self, in_ch, out_ch):
            """Initialize the layer parameters."""
            super().__init__(aggr='add')

            self.linear = torch.nn.Linear(in_ch, out_ch)
    
        def forward(self, x, edge_index):
            """Perform a single round of message passing."""
            # we add self loops
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

            # we linearly transform node features before doing the message passing
            x = self.linear(x)

            # we normalize features based on the product of the square root of in-degrees
            source, target = edge_index
            deg = torch_geometric.utils.degree(target, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]        

            return self.propagate(edge_index, x=x, norm=norm)

        def message(self, x_j, norm):
            """Construct the message that is sent along each edge."""
            # in our example, x_j has shape [m, 1]
            return norm.view(-1, 1) * x_j

    return (GraphConvolution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This completes the implementation of the Graph Convolution according to Kipf and Welling. We can instantiate our model and calculate the transformed node features after one step of the message passing:
    """)
    return


@app.cell
def _(GraphConvolution, n):
    mp_3 = GraphConvolution(1, 1)
    _x = mp_3.forward(n.data.x, n.data.edge_index)
    print(_x)
    return (mp_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the model now includes a linear perceptron with two trainable parameters (the `weight` and `bias`) that is applied to the feature by each node before passing the (transformed) feature to its neighbors. We can output those parameters, which we will later learn based on Stochastic Gradient Descent:
    """)
    return


@app.cell
def _(mp_3):
    print([print(x) for x in mp_3.parameters()])
    return


if __name__ == "__main__":
    app.run()
