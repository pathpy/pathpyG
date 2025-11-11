"""Module for clustering coefficient calculations."""

from typing import Set

import torch

from pathpyG.core.graph import Graph


def local_clustering_coefficient(g: Graph, u: str) -> float:
    """Calculates the local clustering coefficient $C_u$ for a given node $u$ from a [graph][pathpyG.core.graph.Graph] $G=(V, E)$.

    The local clustering coefficient is defined as the fraction of closed
    triads around node $u$ over the number of possible triads.

    Args:
        g (Graph): The graph in which to calculate the clustering coefficient.
        u (str): The node for which to calculate the clustering coefficient.

    Returns:
        float: The local clustering coefficient of node u.
    """
    # Compute number of directly connected neighbour pairs
    k_u = float(len(closed_triads(g, u)))

    # Normalise fraction based on number of possible edges
    if g.is_directed():
        if g.out_degrees[u] > 1:
            return k_u / (g.out_degrees[u] * (g.out_degrees[u] - 1))
        else:
            return 0.0
    else:
        k_u /= 2.0
        if g.degrees()[u] > 1:
            return 2.0 * k_u / (g.degrees()[u] * (g.degrees()[u] - 1))
        else:
            return 0.0


def avg_clustering_coefficient(g: Graph) -> float:
    r"""Calculates the average clustering coefficient $C$ of the [graph][pathpyG.core.graph.Graph] $G=(V, E)$.

    Given the local clustering coefficients $C_u$ for all nodes $u \in V$,
    the average clustering coefficient is defined as their mean:

    $$
    C = \frac{1}{n} \sum_{u \in V} C_u
    $$

    Warning:
        This measurement of globale clustering should not be confused with the global clustering coefficient
        defined as the fraction of closed paths of length two over all paths of length two in the graph.

    ??? reference
        Proposed by Watts and Strogatz in their seminal paper on "Collective dynamics of 'small-world' networks"[^1].
        Further details can be found in in Chapter 7.3 in *Networks*[^2] by Mark Newman.

    [^1] *Watts, D. J. & Strogatz, S. H. Collective dynamics of 'small-world' networks. Nature 393, 440-442 (1998).*
    [^2] *Newman, M. E. J. Networks. (Oxford University Press, 2018). doi:10.1093/oso/9780198805090.001.0001.*

    Args:
        g (Graph): The graph in which to calculate the average clustering coefficient.

    Returns:
        float: The average clustering coefficient of the graph.
    """
    return torch.mean(torch.tensor([local_clustering_coefficient(g, v) for v in g.nodes], dtype=torch.float32)).item()


def closed_triads(g: Graph, v: str) -> Set:
    """Calculates the set of edges that represent a closed triad around a given node $v$.

    Args:
        g (Graph): The graph in which to calculate the list of closed triads.
        v (str): The node around which to calculate the closed triads.
    """
    c_triads: set = set()
    edges = set()

    # Collect all edges of successors
    for x in g.successors(v):
        for y in g.successors(x):
            edges.add((x, y))

    for x, y in edges:
        if y in g.successors(v):
            c_triads.add((x, y))
    return c_triads
