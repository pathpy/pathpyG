"""Algorithms to generate random graphs

The functions in this module allow to generate graphs based on 
probabilistic generative models.

Example:
    ```py
    import pathpyG as pp

    
    ```
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional
)

import scipy.special
from networkx import centrality
from tqdm import tqdm

from collections import defaultdict, Counter, deque
from pathpyG.algorithms.temporal import temporal_shortest_paths, lift_order_temporal
import numpy as _np
import torch
from torch import tensor

from torch_geometric.utils import to_networkx, degree

from pathpyG.core.Graph import Graph
from pathpyG.core.IndexMap import IndexMap


def max_edges(n: int, directed: bool = False, multi_edges: bool = False, self_loops: bool = False) -> int | float:
    """Returns the maximum number of edges that a directed or undirected network with n nodes can
    possible have (with or without loops).

    Args:
        n: The number of nodes in the network  
        directed: If True, return the maximum number of edges in a directed network.
        multi_edges: If True, multiple edges between each node pair are allowed. In this case np.inf is returned.
        self_loops: If True, include self-loops.

    Example:
    ```py
        # Compute maximum number of edges in directed/undirected network with/without self-loops and 100 nodes
        import pathpyG as pp
        print(pp.algorithms.generative_models.max_edges(100)
        # 4950

        print(pp.algorithms.generative_models.max_edges(100, directed=True)
        9900

        print(pp.algorithms.generative_models.max_edges(100, directed=True, loops=True)
        # 10000
    ```
    """

    if multi_edges:
        return _np.inf
    elif self_loops and directed:
        return int(n**2)
    elif self_loops and not directed:
        return int(n*(n+1)/2)
    elif not self_loops and not directed:
        return int(n*(n-1)/2)
    else:  # not loops and directed:
        return int(n*(n-1))


def G_nm(n: int, m: int, mapping: IndexMap | None = None, self_loops: bool = False, multi_edges: bool = False, directed: bool = False) -> Graph:
    """Generate a random graph with n nodes and m edges based on the G(n,m) model by Pal Eröds and Alfred Renyi.
    
    Args:
        n: the number of nodes of the graph
        m: the number of random edges to be generated
        mapping: optional given mapping of n nodes to node IDs. If this is not given a mapping is created
        self_loops: whether or not to allow self-loops (v,v) to be generated
        multi_edges: whether or not multiple identical edges are allowed
        directed: whether or not to generate a directed graph
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


def G_nm_randomize(graph: Graph, self_loops: bool = False, multi_edges: bool = False) -> Graph:
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
    """
    if graph.is_undirected():
        m = int(graph.M/2)
    else:
        m = graph.M
    return G_nm(graph.N, m, directed=graph.is_directed(), self_loops=self_loops, multi_edges=multi_edges,
                mapping=graph.mapping)


def G_np(n: int, p: float, mapping: IndexMap | None = None, self_loops: bool = False, directed: bool = False) -> Graph:
    """Generate a random graph with n nodes link probability p on the G(n,p) model by Edgar Nelson Gilbert.
    
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


def G_np_randomize(graph: Graph, self_loops: bool = False) -> Graph:
    """Generate a random microstate based on the G(n,p) model. 
    
    The number of nodes,
    the expected number of edges, the edge directedness and the node uids of the 
    generated network match the corresponding values of a given network instance.
    """
    if graph.is_directed():
        m = graph.M
    else:
        m = int(graph.M/2)
    M = max_edges(graph.N, directed=graph.is_directed(), self_loops=self_loops)
    p = m/M
    return G_np(n=graph.N, p=p, directed=graph.is_directed(), self_loops=self_loops, mapping=graph.mapping)


def G_np_likelihood(p: float, graph: Graph) -> float:
    """Calculate the likelihood of parameter p for a G(n,p) model and a given graph
    """
    assert graph.is_directed is False
    return p**graph.N * (1-p)**(scipy.special.binom(graph.N, 2)-graph.M/2)


def Gnp_log_likelihood(p: float, graph: Graph) -> float:
    """Calculate the log-likelihood of parameter p for a G(n,p) model and a given graph
    """
    return (graph.M/2)*_np.log10(p) + (scipy.special.binom(graph.N, 2)-(graph.M/2)) * _np.log10(1-p)


def G_np_MLE(graph: Graph) -> float:
    """Calculate the maximum likelihood estimate of parameter p for a G(n,p) model and a given undirected graph
    """
    assert graph.is_directed() is False
    return (graph.M/2) / scipy.special.binom(graph.N, 2)


def is_graphic_Erdos_Gallai(degrees: list[int]) -> bool:
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
        for i in range(1, r+1):
            S += degree_sequence[i-1]
        for i in range(r+1, n+1):
            M += min(r, degree_sequence[i-1])
        if S > r * (r-1) + M:
            return False
    return True


def generate_degree_sequence(n, distribution: Dict[float, float] | scipy.stats.rv_continuous | scipy.stats.rv_discrete, 
                             **distribution_args) -> _np.array:
    """Generates a random graphic degree sequence drawn from a given degree distribution"""
    # create rv_discrete object with custom distribution and generate degree sequence
    if isinstance(distribution, dict):
        degrees = [k for k in distribution]
        probs = [distribution[k] for k in degrees]

        dist = scipy.stats.rv_discrete(name='custom', values=(degrees, probs))
        s = [1]
        while not is_graphic_Erdos_Gallai(s):
            s = dist.rvs(size=n, **distribution_args)
        return s
    # use scipy rv objects to generate graphic degree sequence
    elif isinstance(distribution, scipy.stats.rv_discrete):
        s = [1]
        while not is_graphic_Erdos_Gallai(s):
            s = distribution.rvs(size=n, **distribution_args)
        return s

    elif isinstance(distribution, scipy.stats.rv_continuous):
        s = [1]
        while not is_graphic_Erdos_Gallai(s):
            s = _np.rint(distribution.rvs(size=n, **distribution_args))
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

    g = Graph.from_edge_list(edges, mapping=mapping, num_nodes=n)
    g.data.node_label = torch.tensor(z)
    return g
