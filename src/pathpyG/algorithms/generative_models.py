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
)

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
