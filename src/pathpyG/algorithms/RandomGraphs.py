"""Random Graph Generation Algorithms"""

import pathpyG as pp
from pathpyG import Graph
from tqdm.notebook import tqdm
import numpy as np
from typing import Optional, Union, Dict




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
    edges = []
    for i in range(n):
        if loops:
            x = 0
            y = s
        else:
            x = 1
            y = s+1
        for j in range(x, y):
            v = nodes[i]
            w = nodes[(i+j) % n]
            if (v, w) not in edges:
                edges.append([v, w])

    if p == 0:
        # nothing to do here
        return pp.Graph()

    # Rewire each link with probability p
    new_edges = []
    for edge in tqdm(edges, 'generating WS network'):
        if np.random.rand() < p:
            # Delete original link and remember source node
            v = edge[0]
            edges.remove(edge)

            # Find new random tgt, which is not yet connected to src
            new_target = None

            # This loop repeatedly chooses a random target until we find
            # a target not yet connected to src. Note that this could potentially
            # result in an infinite loop depending on parameters.
            while new_target is None:
                x = reversed_mapping[np.random.randint(n)]
                if (x != v or loops) and (v, x) not in edges:
                    new_target = x
            new_edges.append([v, new_target])
        else:
            new_edges.append(edge)

    g = pp.Graph.from_edge_list(new_edges)
    g = g.to_undirected()
    return g