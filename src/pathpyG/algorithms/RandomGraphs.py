"""Random Graph Generation Algorithms"""

from typing import Optional, Union, Dict

import torch

import pathpyG as pp




def Watts_Strogatz(n: int, s: int, p: float = 0.0, loops: bool = False,
                mapping: pp.IndexMap = None) -> pp.Graph:
    """Generate a Watts-Strogatz small-world graph.
    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    s : int
        The number of edges to attach from a new node to existing nodes.
    p : float
        The probability of rewiring each edge.
    loops : bool
        If True, allow the graph to contain self-loops.
    mapping : IndexMap
        A mapping from the node indices to node names.
    Returns
    -------
    Graph
        A Watts-Strogatz small-world graph.

    Example
    -------
    >>> g = Watts_Strogatz(100, 4, 0.1, mapping=pp.IndexMap([f"n_{i}" for i in range(100)])
    >>> pp.visualisations.plot(g, node_label=g.mapping.node_ids)
    """

    nodes = torch.arange(n)

    # construct a ring lattice (dimension 1)
    edges_right = torch.stack([torch.stack((nodes, torch.roll(nodes, shifts=-i, dims=0))) for i in range(1, s+1)], dim=0).permute(1, 0, 2).reshape(2, -1)
    edges_left = torch.stack([torch.stack((nodes, torch.roll(nodes, shifts=+i, dims=0))) for i in range(1, s+1)], dim=0).permute(1, 0, 2).reshape(2, -1)
    edges = torch.cat((edges_right, edges_left), dim=1)

    if p == 0:
        return pp.Graph()

    # Rewire each link with probability p
    rand_vals = torch.rand(edges.shape[0])
    rewire_mask = rand_vals < p

    new_edges = edges.clone()
    new_edges[rewire_mask, 1] = torch.randint(n, (torch.sum(rewire_mask),)).squeeze()

    g = pp.Graph.from_edge_index(new_edges.tolist(), mapping=mapping)

    g = g.to_undirected()
    return g