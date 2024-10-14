from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Set
)

from collections import defaultdict

from pathpyG.core.Graph import Graph
import numpy as _np

def local_clustering_coefficient(g: Graph, u: str) -> float:

    k_u = 0

    # Compute number of directly connected neighbour pairs
    for (v,w) in g.edges:
        if v in g.successors(u) and w in g.successors(u):

            # In this case we have three edges (u,v), (u,w) and (v,w), i.e. a closed triad
            k_u += 1
    
    # Normalise fraction based on number of possible edges
    if g.is_directed:
        if g.out_degrees[u]>1:
            return k_u/(g.out_degrees[u]*(g.out_degrees[u]-1))
        else: 
            return 0.
    else:    
        if g.degrees()[u]>1:
            return 2*k_u/(g.degrees()[u]*(g.degrees()[u]-1))
        else:
            return 0.


def avg_clustering_coefficient(g: Graph) -> float:
    return _np.mean([ local_clustering_coefficient(g, v) for v in g.nodes ])


def closed_triads(g: Graph, v: str) -> Set:
    """Calculates the set of edges that represent a closed triad
    around a given node v.

    Parameters
    ----------

    network : Network

        The network in which to calculate the list of closed triads

    """
    ct: set = set()
    edges = set()

    for w in g.successors(v):
        for x in g.predecessors(w):
            edges.add((x, w))
    
    for (v, w) in edges:
        if (v in g.successors(v) and
               w in g.successors(v)):
            ct.add((v,w))
    return ct