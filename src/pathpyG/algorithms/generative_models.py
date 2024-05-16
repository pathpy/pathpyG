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

