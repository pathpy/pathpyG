"""Algorithms to generate random graphs

The functions in this module allow to generate graphs based on
different probabilistic generative models.

Example:
    ```py
    import pathpyG as pp
    
    g = pp.algorithms.generative_models.erdos_renyi_gnm(n=100, m=200)
    ```
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional
import warnings
import random 

import scipy.special

import numpy as _np
import torch
from torch_geometric.utils import degree

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap


def max_edges(n: int, directed: bool = False, multi_edges: bool = False, self_loops: bool = False) -> int | float:
    """Returns the maximum number of edges that a directed or undirected network with n nodes can
    possible have (with or without loops).

    Args:
        n: The number of nodes in the network
        directed: If True, return the maximum number of edges in a directed network.
        multi_edges: If True, multiple edges between each node pair are allowed. In this case np.inf is returned.
        self_loops: If True, include self-loops.

    Examples:
        Compute maximum number of edges in undirected network without self-loops and 100 nodes

        >>> import pathpyG as pp
        >>> print(pp.algorithms.generative_models.max_edges(100)
        4950

        Directed networks without self-loops

        >>> print(pp.algorithms.generative_models.max_edges(100, directed=True)
        9900

        Directed networks with self-loops 

        >>> print(pp.algorithms.generative_models.max_edges(100, directed=True, loops=True)
        10000
    """

    if multi_edges:
        return _np.inf
    elif self_loops and directed:
        return int(n**2)
    elif self_loops and not directed:
        return int(n * (n + 1) / 2)
    elif not self_loops and not directed:
        return int(n * (n - 1) / 2)
    else:  # not loops and directed:
        return int(n * (n - 1))


def erdos_renyi_gnm(n: int, m: int, mapping: IndexMap | None = None,
                    self_loops: bool = False, multi_edges: bool = False,
                    directed: bool = False) -> Graph:
    """Generate a random graph with n nodes and m edges based on the G(n,m) model by Pal Eröds and Alfred Renyi.

    Args:
        n: the number of nodes of the graph
        m: the number of random edges to be generated
        mapping: optional given mapping of n nodes to node IDs. If this is not given a mapping is created
        self_loops: whether or not to allow self-loops (v,v) to be generated
        multi_edges: whether or not multiple identical edges are allowed
        directed: whether or not to generate a directed graph
    
    Returns:
        Graph: graph object
    """
    assert m <= max_edges(n, directed=directed, self_loops=self_loops, multi_edges=multi_edges)

    edges = set()
    edges_added: int = 0

    if mapping is None:
        # make sure that we have indices for all n nodes even if not all
        # nodes have incident edges
        mapping = IndexMap([str(i) for i in range(n)])

    # Add m edges at random
    while edges_added < m:

        # Choose two random nodes (with replacement if self-loops are included)
        v, w = _np.random.choice(n, size=2, replace=self_loops)

        # avoid multi-edges
        if multi_edges or (mapping.to_id(v), mapping.to_id(w)) not in edges:
            edges.add((mapping.to_id(v), mapping.to_id(w)))
            if not directed and v != w:
                edges.add((mapping.to_id(w), mapping.to_id(v)))
            edges_added += 1

    return Graph.from_edge_list(list(edges), is_undirected=not directed, mapping=mapping, num_nodes=n)


def erdos_renyi_gnm_randomize(graph: Graph, self_loops: bool = False, multi_edges: bool = False) -> Graph:
    """Generate a random graph whose number of nodes, edges, edge directedness and node IDs
    match the corresponding values of a given network instance. Useful to generate a randomized
    version of a network.

    Args:
        graph: A given network used to determine number of nodes, edges, node uids, and edge directedness
        self_loops: Whether or not the generated network can contain loops.
        multi_edges: Whether or not multiple edges can be added to the same node pair

    Example:
    ```py
        # Generate undirected network
        import pathpyG as pp
        g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('d', 'e')])
        r = pp.algorithms.generative_models.G_nm_randomize(g)
    ```
    """
    if graph.is_undirected():
        m = int(graph.m / 2)
    else:
        m = graph.m
    return erdos_renyi_gnm(
        graph.n, m, directed=graph.is_directed(),
        self_loops=self_loops,
        multi_edges=multi_edges,
        mapping=graph.mapping
    )


def erdos_renyi_gnp(n: int, p: float, mapping: IndexMap | None = None,
                    self_loops: bool = False, directed: bool = False) -> Graph:
    """Generate an Erdös-Renyi random graph with n nodes and 
    link probability p, using the G(n,p) model by Edgar Nelson Gilbert.

    Args:
        n: the number of nodes of the graph
        p: the link probability
        self_loops: whether or not to allow self-loops (v,v) to be generated
        directed: whether or not to generate a directed graph
    """
    edges = set()

    if mapping is None:
        # make sure that we have indices for all n nodes even if not all
        # nodes have incident edges
        mapping = IndexMap([str(i) for i in range(n)])

    # fast handling of special case p = 0
    if p == 0.0:
        return Graph.from_edge_list([], is_undirected=not directed, num_nodes=0)

    # connect pairs of nodes with probability p
    for s in range(n):
        if directed:
            x = n
        else:
            x = s + 1
        for t in range(x):
            if not self_loops and t == s:
                continue
            if _np.random.random() <= p:
                edges.add((mapping.to_id(s), mapping.to_id(t)))
                if not directed and s != t:
                    edges.add((mapping.to_id(t), mapping.to_id(s)))

    return Graph.from_edge_list(list(edges), is_undirected=not directed, mapping=mapping, num_nodes=n)


def erdos_renyi_gnp_randomize(graph: Graph, self_loops: bool = False) -> Graph:
    """Randomize a given graph based on the Erdös-Renyi random graph G(n,p) model.

    The number of nodes, expected number of edges, edge directedness and node uids of the
    generated graph match the corresponding values of the graph given as parameter.
    """
    if graph.is_directed():
        m = graph.m
    else:
        m = int(graph.m / 2)
    M = max_edges(graph.n, directed=graph.is_directed(), self_loops=self_loops)
    p = m / M
    return erdos_renyi_gnp(n=graph.n, p=p, directed=graph.is_directed(), 
                           self_loops=self_loops, mapping=graph.mapping)


def erdos_renyi_gnp_likelihood(p: float, graph: Graph) -> float:
    """Calculate the likelihood of parameter p for a G(n,p) model and a given graph"""
    assert graph.is_directed is False
    return p**graph.n * (1 - p) ** (scipy.special.binom(graph.n, 2) - graph.m / 2)


def erdos_renyi_gnp_log_likelihood(p: float, graph: Graph) -> float:
    """Calculate the log-likelihood of parameter p for a G(n,p) model and a given graph"""
    return (graph.m / 2) * _np.log10(p) + (scipy.special.binom(graph.n, 2) - (graph.m / 2)) * _np.log10(1 - p)


def erdos_renyi_gnp_mle(graph: Graph) -> float:
    """Calculate the maximum likelihood estimate of parameter p for a G(n,p) model and a given undirected graph"""
    assert graph.is_directed() is False
    return (graph.m / 2) / scipy.special.binom(graph.n, 2)


def watts_strogatz(
    n: int,
    s: int,
    p: float = 0.0,
    undirected: bool = True,
    allow_duplicate_edges: bool = True,
    allow_self_loops: bool = True,
    mapping: IndexMap | None = None,
) -> Graph:
    """Generate a Watts-Strogatz small-world graph.

    Args:
        n: The number of nodes in the graph.
        s: The number of edges to attach from a new node to existing nodes.
        p: The probability of rewiring each edge.
        undirected: If True, the graph will be undirected.
        allow_duplicate_edges: If True, allow duplicate edges in the graph.
            This is faster but may result in fewer edges than requested in the undirected case
            or duplicates in the directed case.
        allow_self_loops: If True, allow self-loops in the graph.
            This is faster but may result in fewer edges than requested in the undirected case.
        mapping: A mapping from the node indices to node names.

    Returns:
        Graph: A Watts-Strogatz small-world graph.

    Examples:
        ```py
        g = Watts_Strogatz(100, 4, 0.1, mapping=pp.IndexMap([f"n_{i}" for i in range(100)])
        ```
    """

    nodes = torch.arange(n)

    # construct a ring lattice (dimension 1)
    edges = (
        torch.stack([torch.stack((nodes, torch.roll(nodes, shifts=-i, dims=0))) for i in range(1, s + 1)], dim=0)
        .permute(1, 0, 2)
        .reshape(2, -1)
    )

    if not allow_duplicate_edges:
        if n * (n - 1) < edges.shape[1]:
            raise ValueError(
                "The number of edges is greater than the number of possible edges in the graph. Set `allow_duplicate_edges=True` to allow this."
            )
        elif n * (n - 1) * 0.5 < edges.shape[1] and p > 0.3:
            warnings.warn(
                "Avoding duplicate in graphs with high connectivity and high rewiring probability may be slow. Consider setting `allow_duplicate_edges=True`."
            )

    # Rewire each link with probability p
    rand_vals = torch.rand(edges.shape[1])
    rewire_mask = rand_vals < p

    # Generate random nodes excluding the current node for each edge that needs to be rewired, also avoid duplicate edges
    edges[1, rewire_mask] = torch.randint(n, (rewire_mask.sum(),))

    # In the undirected case, make sure the edges all point in the same direction
    # to avoid duplicate edges pointing in opposite directions
    if undirected:
        edges = edges.sort(dim=0)[0]
    final_edges = edges

    if not allow_duplicate_edges:
        # Remove duplicate edges
        final_edges, counts = edges.unique(dim=1, return_counts=True)
        if final_edges.shape[0] < edges.shape[1]:
            for i, edge in enumerate(final_edges[:, counts > 1].T):
                for _ in range(counts[counts > 1][i] - 1):
                    while True:
                        new_edge = torch.tensor([edge[0], torch.randint(n, (1,))]).sort()[0].unsqueeze(1)
                        # Check if the new edge is already in the final edges
                        # and add it if not
                        if (new_edge != final_edges).any(dim=0).all():
                            final_edges = torch.cat((final_edges, new_edge), dim=1)
                            break

    if not allow_self_loops:
        self_loop_edges = final_edges[:, final_edges[0] == final_edges[1]]
        final_edges = final_edges[:, final_edges[0] != final_edges[1]]
        for self_loop_edge in self_loop_edges.T:
            while True:
                new_edge = torch.tensor([self_loop_edge[0], torch.randint(n, (1,))]).sort()[0].unsqueeze(1)
                # Check if the new edge is already in the final edges
                # and add it if not
                if (new_edge != final_edges).any(dim=0).all() and new_edge[0] != new_edge[1]:
                    final_edges = torch.cat((final_edges, new_edge), dim=1)
                    break

    g = Graph.from_edge_index(final_edges, mapping=mapping)
    if undirected:
        g = g.to_undirected()
    return g


def is_graphic_erdos_gallai(degrees: list[int] | _np.ndarray) -> bool:
    """Check Erdös and Gallai condition.

    Checks whether the condition by Erdös and Gallai (1967) for a graphic degree
    sequence is fulfilled.

    Args:
        degrees: List of integer node degrees to be tested.
    """
    degree_sequence = sorted(degrees, reverse=True)
    S = sum(degree_sequence)
    n = len(degree_sequence)
    if S % 2 != 0:
        return False
    for r in range(1, n):
        M = 0
        S = 0
        for i in range(1, r + 1):
            S += degree_sequence[i - 1]
        for i in range(r + 1, n + 1):
            M += min(r, degree_sequence[i - 1])
        if S > r * (r - 1) + M:
            return False
    return True


def generate_degree_sequence(
    n: int,
    distribution: Dict[float, float] | scipy.stats.rv_continuous | scipy.stats.rv_discrete,
    **distribution_args: Any,
) -> _np.ndarray:
    """Generates a random graphic degree sequence drawn from a given degree distribution"""
    s = _np.array([1])
    # create rv_discrete object with custom distribution and generate degree sequence
    if isinstance(distribution, dict):
        degrees = [k for k in distribution]
        probs = [distribution[k] for k in degrees]

        dist = scipy.stats.rv_discrete(name="custom", values=(degrees, probs))

        while not is_graphic_erdos_gallai(s):
            s = dist.rvs(size=n, **distribution_args)
        return s
    # use scipy rv objects to generate graphic degree sequence
    elif hasattr(distribution, "rvs"):
        while not is_graphic_erdos_gallai(s):
            s = distribution.rvs(size=n, **distribution_args)
            # Check if the distribution is discrete
            if s.dtype != int:
                s = _np.rint(s)
        return s
    else:
        raise NotImplementedError()


def stochastic_block_model(M: _np.matrix, z: _np.array, mapping: Optional[IndexMap] = None) -> Graph:
    """Generate a random undirected graph based on the stochastic block model

    Args:
        M: n x n stochastic block matrix, where entry M[i,j] gives probability of edge to be generated
            between nodes in blocks i and j
        z: n-dimensional block assignment vector, where z[i] gives block assignment of i-th node
        mapping: optional mapping of node IDs to indices. If not given, a standard
            mapping based on integer IDs will be created
    """
    # the number of nodes is implicitly given by the length of block assignment vector z
    n = len(z)

    # we can use pre-defined node names, if not given, we use contiguous numbers
    if mapping is None:
        mapping = IndexMap([str(i) for i in range(n)])

    edges = []

    # randomly generate links with probabilities given by entries of the stochastic block matrix M
    for u in range(n):
        for v in range(u):
            if _np.random.random() <= M[z[u], z[v]]:
                edges.append((mapping.to_id(u), mapping.to_id(v)))
                edges.append((mapping.to_id(v), mapping.to_id(u)))

    g = Graph.from_edge_list(edges, mapping=mapping, num_nodes=n).to_undirected()
    return g



def molloy_reed(degree_sequence: _np.array | Dict[int, float],
                multiedge: bool = False,
                relax: bool = False,
                node_ids: Optional[list] = None) -> Graph:
    """Generate Molloy-Reed graph.

    Generates a random undirected network without self-loops, with given degree sequence based on
    the Molloy-Reed algorithm. The condition proposed by Erdös and Gallai (1967)
    is used to test whether the degree sequence is graphic, i.e. whether a network
    with the given degree sequence exists.

    Args:
        degrees: List of integer node degrees. The number of nodes of the generated
        network corresponds to len(degrees).

        relax: If True, we conceptually allow self-loops and multi-edges, but do not
        add them to the network. This implies that the generated graph may not
        have exactly sum(degrees)/2 edges, but it ensures that the algorithm
        always finishes.

        node_ids : Optional list of node IDs that will be used for Indexmapping.

    Examples:

    Generate random undirected network with given degree sequence

    >>> import pathpyG as pp
    >>> random_network = pp.algorithms.generative_models.molloy_reed([1,0])
    >>> print(random_network)
    ...

    Network generation fails for non-graphic degree sequence

    >>> import pathpyG as pp
    >>> random_network = pp.algorithms.generative_models.molloy_reed([1,0])
    raises AttributeError

    """

    # assume that we are given a graphical degree sequence
    if not is_graphic_erdos_gallai(degree_sequence):
        raise AttributeError('degree sequence is not graphic')

    # create empty network with n nodes
    n = len(degree_sequence)
    edges: list = []

    if node_ids is None or len(node_ids) != n:
        node_ids: list = []
        for i in range(n):
            node_ids.append(i)

    # generate edge stubs based on degree sequence
    stubs: list = []
    for i in range(n):
        for _ in range(int(degree_sequence[i])):
            stubs.append(node_ids[i])

    # connect randomly chosen pairs of stubs
    while len(stubs) > 0:
        # find candidate node pair to connect
        v, w = _np.random.choice(stubs, 2, replace=False)

        # we encountered candidate edge that we cannot add
        if v == w or (((v, w) in edges or (w, v) in edges) and not multiedge and not relax):
            # break up random edge and add back stubs to avoid
            # infinite loop
            if len(edges) > 0:
                e = random.choice(edges)
                edges.remove(e)
                stubs.append(e[0])
                stubs.append(e[1])
        elif v != w:
            edges.append((v, w))
            stubs.remove(v)
            stubs.remove(w)

    return Graph.from_edge_list(edges).to_undirected()


def molloy_reed_randomize(g: Graph) -> Optional[Graph]:
    """Generates a random realization of a given network based on the observed degree sequence.
    """
    if g.is_directed():
        raise NotImplementedError('molloy_reed_randomize is only implemented for undirected graphs')
    # degrees are listed in order of node indices
    degrees = degree(g.data.edge_index[1], num_nodes=g.n, dtype=torch.int).tolist()

    return molloy_reed(degrees, node_ids=g.nodes).to_undirected()


def k_regular_random(k: int, n: Optional[int] = None, node_ids: Optional[list] = None) -> Optional[Graph]:
    """Generate a random graph in which all nodes have exactly degree k

    Args:
        k: degree of all nodes in the generated network.
        node_ids: Optional list of node uids that will be used.

    Examples:
        
        Generate random undirected network with given degree sequence

        >>> import pathpy as pp
        >>> random_network = pp.algorithms.random_graphs.Molloy_Reed([1,0])
        >>> print(random_network.summary())
        ...

        Network generation fails for non-graphic sequences

        >>> import pathpy as pp
        >>> random_network = pp.algorithms.random_graphs.Molloy_Reed([1,0])
        >>> print(random_network)
        None
    """
    if k < 0:
        msg = 'Degree parameter k must be non-negative'
        raise ValueError(msg)
    if n is None and node_ids is None:
        msg = 'You must either pass a list of node ids or a number of nodes to generate'
        raise ValueError(msg)

    if n is None:
        n = len(node_ids)
    
    return molloy_reed([k]*n, multiedge=False, relax=False, node_ids=node_ids)
