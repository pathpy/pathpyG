"""Random Graph Generation Algorithms"""

from typing import Optional, Union, Dict

import numpy as np
from tqdm.notebook import tqdm

import pathpyG as pp
from pathpyG import Graph




def Watts_Strogatz(n: int, s: int, p: float = 0.0, loops: bool = False,
                mapping: list = None) -> pp.Graph:
    """Undirected Watts-Strogatz lattice network

    Generates an undirected Watts-Strogatz lattice network with lattice
    dimensionality one.

    Parameters
    ----------
    n : int

        The number of nodes in the generated network

    s : float

        The number of nearest neighbors that will be connected
        in the ring lattice

    p : float

        The rewiring probability

    mapping : list, optional

        Optional mapping of node indices to string IDs

    Examples
    --------
    Generate random undirected network with 10 nodes

    >>> import pathpy as pp
    >>> random_graph = pp.algorithms.random_graphs.Watts_Strogatz(n=50, s=4, p=0.01, mapping=[str(x) for x in range(50)])
    >>> print(random_graph.summary())
    ...

    """
    
    nodes = []

    if mapping is None or len(mapping) != n:
        mapping = {}
        for i in range(n):
            nodes.append(str(i))
            mapping[str(i)] = i
        reversed_mapping = {
            j: i for i,j in mapping.items()
        }
    else:
        mapping = {
        j: i for i,j in enumerate(mapping)
        }
        reversed_mapping = {
            i: j for i,j in enumerate(mapping)
        }
        for i in range(n):
            nodes.append(reversed_mapping[i])

    # construct a ring lattice (dimension 1)
    nodes = torch.arange(s)
    edges = IntTensor([nodes[:-1], nodes[1:]])

    if p == 0:
        # nothing to do here
        return pp.Graph()

    # Rewire each link with probability p
    new_edges = []
    rnd_edge_mask = (torch.rand(edges.size(1) < p)
    rnd_edges = edges[rnd_edge_mask]
    rnd_edges[1] = torch.randint(len(nodes), (rnd_edges.size(1),)
    edges[rnd_edge_mask] = rnd_edges
    while edges.size(1) != edges.unique(dim=0).size(1):
       unique_edges = edges.unique(return_inverse=True, dim=0)[1]
       mask = torch.ones(edges.size(1), dtype=bool)
       mask[unique_edges] = False
       doubled_edges = edges[mask]
       doubled_edges[1] =  torch.randint(len(nodes), (doubled_edges.size(1),)
       edges[mask] = doubled_edges

    g = pp.Graph.from_edge_list(new_edges)
    g = g.to_undirected()
    return g