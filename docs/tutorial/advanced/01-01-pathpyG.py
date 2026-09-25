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
    # Introducing `pathpyG`

    *August 4 2026*
    *Training Workshop: Introduction to Deep Graph Learning*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We first introduce `pathpyG`, a graph learning and visualization package that is being developed at my chair at University of Wuerzburg.

    `pathpyG` has a couple of advantages that motivate the use across the two training workshops. First, it is easy to install since it is a pure `python` package that does not require compilation. Second, `pathpyG` has a user-friendly API that makes it easy to handle directed and undirected networks, networks where nodes or edges have attributes as well as temporal networks. Third, it provides interactive HTML visualizations that can be directly displayed inside `jupyter` notebooks, making it particularly suitable for educational settings. Moreover, it directly supports the analysis and visualization of time series data on networked systems, such as time-stamped edges or data on paths in networks.

    Finally and most importantly, `pathpyG` is based on `pyTorch` and `torch-geometric` and uses tensors as internal data representation. This facilitates the analysis of large data sets on the GPU and makes it easy to apply graph neural networks in later chapters of our tutorials.

    To get started, we first import `pathpyG` and assign the local alias `pp`:
    """)
    return


@app.cell
def _():
    import pathpyG as pp

    return (pp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating networks

    `pathpyG` provides the `Graph` class. The constructor takes a `pyG` Data object that can be used to pass an edge-index that captures the edges of a graph, as well as arbitrary node-, edge- or graph-level attributes. To simplify the creation of small example networks, you can use a static function that create a `Graph` object based on a list of edges represented as tuples of integers or strings.

    Printing the `Graph` object will give a short string summary which tells whether the network is directed or undirected, as well as the number of unique nodes and links.
    """)
    return


@app.cell
def _(pp):
    g1 = pp.Graph.from_edge_list([(0, 1), (1, 2), (2, 0)])
    print(g1)
    return (g1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A network is directed by default, but we can create an undirected network by calling the function `to_undirected` of the `Graph` instance. This will internally generate edges for all directions, i.e. for the example above it will additionally generate the edges that connect nodes in the opposite direction.
    """)
    return


@app.cell
def _(g1):
    g2 = g1.to_undirected()
    print(g2)
    return (g2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the example above, we have used integer numbers to refer to different nodes. Internally, this will create a `pyG` Data object with an edge index, where nodes are always represented by integer indices. Let us have a look at this internal data structure:
    """)
    return


@app.cell
def _(g2):
    print(g2.data.edge_index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that this edge index contains all edges for both directions, where the first tensor contains all source nodes (ordered by their index) while the second tensor contains all target nodes. This sorted tensor representation allows us to easy convert the edge index tensor to sparse matrix representations, that can be used e.g. to calculate matrix-based measures or Laplacian operators.

    While the approach to use integers is easy to understand, it is often more convenient to use string-based node labels. Different from `pyG`, this is supported in `pathpyG`, i.e. we can create a network as follows:
    """)
    return


@app.cell
def _(pp):
    n = pp.Graph.from_edge_list([('Tom', 'Bert'), ('Bert', 'Bill'), ('Bill', 'Tom')]).to_undirected()
    print(n)
    return (n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Checking the edge index, we find that we again have integer-based node indices:
    """)
    return


@app.cell
def _(n):
    n.data.edge_index
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, `pathpyG` automatically generates a ID to index mapping object that is automatically applied when we e.g. enumerate through nodes. We can manually check this mapping and use it to resolve IDs to integer indices and vice-versa:
    """)
    return


@app.cell
def _(n):
    print(n.mapping)
    return


@app.cell
def _(n):
    n.mapping.to_idx('Tom')
    return


@app.cell
def _(n):
    n.mapping.to_idx('Bert')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we want to check explicitly whether a node exists before creating and edge, we can test this with the `in` operator on the set of nodes available via `Graph.nodes`:
    """)
    return


@app.cell
def _(n):
    print('Tom' in n.nodes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To count the number of nodes and (directed) edges in a network we can use the `n` and `m` attributes:
    """)
    return


@app.cell
def _(n):
    print('Network has {0} nodes and {1} edges'.format(n.n, n.m))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Enumerating nodes and edges

    We can iterate through nodes via the nodes iterator as follows. If the Graph object includes an index-ID mapping, this will be applied automatically:
    """)
    return


@app.cell
def _(n):
    for v in n.nodes:
        print(v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similar to `nodes`, the `edges` iterator of the network contains all edges of a network. Each edge is returned as a tuple and the id-index mapping is applied automatically if such a mapping exists (otherwise we obtain a tuple of node indices):
    """)
    return


@app.cell
def _(n):
    for e in n.edges:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We often want to check whether an edge exists between a specific pair of nodes. We can do this by using the `is_edge` function:
    """)
    return


@app.cell
def _(n):
    print(n.is_edge('Tom', 'Bert'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can access the degrees of nodes, i.e. the number of other nodes to which a node is connected, via the `degrees()` function of the Network. For an undirected network, the degrees() function gives the undirected degrees (i.e. irrespective of the directionality of an edge). For directed networks we can use the mode parameter to calculate the in- or out-degree of of ndoes in a directed network (i.e. to how many other nodes the edges of a node point of from how many other nodes edges point to the given node).

    All of those functions return a dictionary that can be indexed via the node ids.
    """)
    return


@app.cell
def _(n):
    n.degrees(mode='in')['Tom']
    return


@app.cell
def _(n):
    n.degrees(mode='in')['Tom']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Networks, Nodes and Edges with attributes

    We often want to use networks to model relational data that contain additional information on nodes, edges, or networks. To support this, `pathpyG` stores data in terms of a pyG data frame, which allows to store arbitrary additional information at the level of nodes, edges or the graph in terms of torch.tensors.
    """)
    return


@app.cell
def _(pp):
    n_1 = pp.Graph.from_edge_list([('Tom', 'Bert'), ('Bert', 'Bill'), ('Bill', 'Tom')])
    print(n_1)
    return (n_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following example, we add an attribute to the modes of the graph. We can directly assing this to the underylying `pyG` data object. All node attributes must be prefixed with `node_`:
    """)
    return


@app.cell
def _(n_1):
    import torch
    n_1.data.node_age = torch.tensor([[44], [28], [125]])
    return (torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The assignment of these values to the nodes is based on the indices of nodes, which we can check via the mapping object. We can now use the following code to access the properties of individual nodes:
    """)
    return


@app.cell
def _(n_1):
    n_1['node_age', 'Bill']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To retrieve the tensor containing the ages of all nodes, we can do the following:
    """)
    return


@app.cell
def _(n_1):
    n_1['node_age']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Just like nodes, `Edge` objects can store arbitrary attributes that we can add as a tensor. The name of the attribute must be prefixed by `edge_`
    """)
    return


@app.cell
def _(n_1, torch):
    n_1.data.edge_type = torch.tensor([[1], [2], [1]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can access those as follows:
    """)
    return


@app.cell
def _(n_1):
    print(n_1['edge_type'])
    print(n_1['edge_type', 'Tom', 'Bert'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjacency and Laplacian matrices

    Adjacency matrices and Laplacian matrices are important mathematical representations of networks.

    ### Adjacency matrix

    The topology of a graph can be represented in the entries of a matrix $A$, where an entry $A[i,j]=1$ indicates that an edge exists from the i-th to the j-th node of the network. The absence of edges is encoded by zero entries. The size of an adjacency matrix representation of a network with n nodes is generally $n^2$, which is not suitable for networks with thousands or millions of nodes. `pathpyG` nevertheless supports efficient adjacency matrix calculation for *sparse* networks, i.e. networks where the majority of node pairs are not connected by an edge. Instead of a fully populated matrix, a call to `Graph.sparse_adj_matrix()` returns a *sparse matrix object*, which is an efficient adjacency-list representation capturing the indices and values of non-zero entries.

    ### Laplacian matrix

    The **Laplacian matrix** is a second, closely related representation that plays a central role in spectral graph theory. For an undirected graph with adjacency matrix $A$ and diagonal degree matrix $D$ (where $D_{ii}$ is the degree of the $i$-th node), the (unnormalized) Laplacian is defined as

    $$ L = D - A $$

    The eigenvalues and eigenvectors of $L$ reveal important structural properties of a graph. For instance, the multiplicity of the eigenvalue zero corresponds to the number of connected components of the graph, and the eigenvector belonging to the second-smallest eigenvalue (the so-called *Fiedler vector*) can be used to detect natural cluster structure in a graph. We will use exactly this idea to construct a simple, unsupervised node embedding technique called *Laplacian eigenmaps* in a later notebook of this tutorial. Just like for the adjacency matrix, `pathpyG` provides a convenience function `Graph.laplacian()` that directly returns a sparse matrix representation of the Laplacian.
    """)
    return


@app.cell
def _(n_1):
    print(n_1.sparse_adj_matrix())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This enables us to directly apply matrix algebra operations from the sparse linear algebra module that is contained in `scipy`. If we instead want a dense matrix that includes zero entries, we can write:
    """)
    return


@app.cell
def _(n_1):
    print(n_1.sparse_adj_matrix().todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fact that the matrix is assymetric tells us that this is a directed network. By default, a binary matrix representation is returned where entries store the presence or absence of edges as 0 or 1 entries. If we want to use numerical attributes of edges instead, we can pass the name of a numerical attribute that should be used:
    """)
    return


@app.cell
def _(n_1):
    print(n_1.sparse_adj_matrix(edge_attr='edge_type').todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How does `pathpyG` populate adjaecency matrices if the network contains multiple edges between the same pair of nodes? Let's try this by creating another edge between Tom and Bert, and let's further add a strength attribute:
    """)
    return


@app.cell
def _(pp, torch):
    n_2 = pp.Graph.from_edge_list([('Tom', 'Bert'), ('Bert', 'Bill'), ('Bill', 'Tom'), ('Tom', 'Bert')])
    n_2.data.edge_weight = torch.tensor([[2], [0.5], [1.2], [3.7]])
    print(n_2)
    return (n_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we now generate an adjacency matrix, the entries contain the *number of different edge objects* between pairs of nodes:
    """)
    return


@app.cell
def _(n_2):
    n_2.sparse_adj_matrix().todense()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we use a numerical attribute to calculate the matrix entries in such a network, the attributes of all edges between the same pair of nodes is automatically summed:
    """)
    return


@app.cell
def _(n_2):
    n_2.sparse_adj_matrix(edge_attr='edge_weight').todense()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's compute the Laplacian matrix for the undirected triangle graph `g2` that we created at the very beginning of this notebook:
    """)
    return


@app.cell
def _(g2):
    print(g2.laplacian().todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since `g2` is the undirected triangle graph on three nodes (each node has degree two), the diagonal entries of $L$ are all equal to two, while the off-diagonal entries are $-1$ for every pair of adjacent nodes (and would be zero for non-adjacent nodes). We can double check this by manually computing $D - A$ from the adjacency matrix and the node degrees:
    """)
    return


@app.cell
def _(g2):
    import scipy as sp

    A = g2.sparse_adj_matrix()
    D = sp.sparse.diags(g2.degrees(mode='in', return_tensor=True).numpy())
    L = D - A
    print(L.todense())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In many applications it is useful to work with a **normalized** Laplacian instead, since the unnormalized Laplacian is dominated by nodes with high degree. `pathpyG` supports two common normalization schemes, which can be selected via the `normalization` argument of `Graph.laplacian()`:

    - `normalization='sym'` gives the **symmetric normalized Laplacian** $L_{sym} = I - D^{-1/2} A D^{-1/2}$
    - `normalization='rw'` gives the **random-walk normalized Laplacian** $L_{rw} = I - D^{-1} A$, which is closely related to the transition matrix of a random walk on the graph

    Let's compute the symmetric normalized Laplacian of our example graph:
    """)
    return


@app.cell
def _(g2):
    print(g2.laplacian(normalization='sym').todense())
    return


if __name__ == "__main__":
    app.run()
