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
    # Networks, Adjacency and Laplacian Matrices

    In [Basic pathpyG Concepts](../basic_concepts/), we have seen how we can create (un)directed `Graph` objects with node-, edge- and graph-level attributes. In this tutorial, we build on this and show how we can represent networks as adjacency and Laplacian matrices, two important mathematical representations of networks that we will use throughout the following tutorials.

    We first import `pathpyG` and `torch`:
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
    We create two small example networks: an undirected triangle graph with integer node indices, and a directed network with string node IDs where edges have a numerical `edge_type` attribute.
    """)
    return


@app.cell
def _(pp):
    g1 = pp.Graph.from_edge_list([(0, 1), (1, 2), (2, 0)])
    g2 = g1.to_undirected()
    print(g2)
    return (g2,)


@app.cell
def _(pp, torch):
    n_1 = pp.Graph.from_edge_list([('Tom', 'Bert'), ('Bert', 'Bill'), ('Bill', 'Tom')])
    n_1.data.edge_type = torch.tensor([[1], [2], [1]])
    print(n_1)
    return (n_1,)


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
    Let's compute the Laplacian matrix for the undirected triangle graph `g2` that we created at the beginning of this notebook:
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
