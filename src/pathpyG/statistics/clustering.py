from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Set
)

from pathpyG.core.Graph import Graph
import numpy as _np


def local_clustering_coefficient(g: Graph, u: str) -> float:

    # Compute number of directly connected neighbour pairs
    k_u = len(closed_triads(g, u))

    # Normalise fraction based on number of possible edges
    if g.is_directed():
        if g.out_degrees[u] > 1:
            return k_u/(g.out_degrees[u]*(g.out_degrees[u]-1))
        return 0.
    else:
        k_u /= 2
        if g.degrees()[u] > 1:
            return 2*k_u/(g.degrees()[u]*(g.degrees()[u]-1))
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
    c_triads: set = set()
    edges = set()

    # Collect all edges of successors
    for x in g.successors(v):
        for y in g.successors(x):
            edges.add((x, y))

    for (x, y) in edges:
        if y in g.successors(v):
            c_triads.add((x, y))
    return c_triads
