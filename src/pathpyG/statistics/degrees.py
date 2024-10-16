from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict
)

from collections import defaultdict

from pathpyG.core.Graph import Graph
import numpy as _np


def degree_sequence(g: Graph, mode: str = 'total') -> _np.array:
    """Calculates the degree sequence of an undirected network.

    Args:
        graph: The `Graph` object for which degrees are calculated
    """
    d = g.degrees()
    if g.is_directed() and mode == 'in':
        d = g.in_degrees
    elif g.is_directed() and mode == 'out':
        d = g.out_degrees
    elif g.is_directed() and mode == 'total':
        d = g.degrees()

    _degrees = _np.zeros(g.N, dtype=float)
    for v in g.nodes:
        _degrees[g.mapping.to_idx(v)] = d[v]
    return _degrees


def degree_distribution(g: Graph, mode: str = 'total') -> Dict[int, float]:
    """Calculates the degree distribution of a graph
    """
    d = g.degrees()
    if g.is_directed() and mode == 'in':
        d = g.in_degrees
    elif g.is_directed() and mode == 'out':
        d = g.out_degrees
    elif g.is_directed() and mode == 'total':
        d = g.degrees()

    cnt: defaultdict = defaultdict(float)
    for v in g.nodes:
        cnt[d[v]] += 1.0 / g.N
    return cnt


def degree_raw_moment(graph: Graph, k: int = 1, mode: str = 'total') -> float:
    """Calculates the k-th raw moment of the degree distribution of a network

    Args:
        graph:  The graph in which to calculate the k-th raw moment

    """
    p_k = degree_distribution(graph, mode=mode)
    mom = 0.0
    for x in p_k:
        mom += x**k * p_k[x]
    return mom


def mean_degree(graph: Graph, mode: str = 'total') -> float:
    return _np.mean(degree_sequence(graph, mode=mode))


def degree_central_moment(graph: Graph, k: int = 1, mode: str = 'total') -> float:
    """Calculates the k-th central moment of the degree distribution.

    Args:
        graph: The graph for which to calculate the k-th central moment

    """
    p_k = degree_distribution(graph, mode=mode)
    mean = _np.mean(degree_sequence(graph, mode=mode))
    m = 0.
    for x in p_k:
        m += (x - mean)**k * p_k[x]
    return m


def degree_assortativity(g: Graph, mode: str = 'total') -> float:
    """Calculate the degree assortativity"""

    A = g.get_sparse_adj_matrix().todense()
    m = _np.sum(A)

    d = g.degrees()
    if g.is_directed() and mode == 'in':
        d = g.in_degrees
    elif g.is_directed() and mode == 'out':
        d = g.out_degrees
    elif g.is_directed() and mode == 'total':
        d = g.degrees()
    elif not g.is_directed():
        m = m/2.

    cov = 0.
    var = 0.
    for i in g.nodes:
        for j in g.nodes:
            cov += (A[g.mapping.to_idx(i), g.mapping.to_idx(j)] - (d[i]*d[j])/(2*m)) * d[i] * d[j]
            if i != j:
                var -= (d[i]*d[j])/(2*m) * d[i] * d[j]
            else:
                var += (d[i] - (d[i]*d[j])/(2*m)) * d[i] * d[j]
    return cov/var


def degree_generating_function(graph: Graph, x: float | list[float] | _np.ndarray, mode: str = 'total') -> float | _np.ndarray:
    """Returns the generating function of the degree distribution of a network,
        calculated for either a single argument x or a list or numpy array of arguments x


    Returns f(x) where f is the probability generating function for the degree
    distribution P(k) for a graph. The function is defined in the interval
    [0,1].  The value returned is from the range [0,1]. The following properties
    hold:

    [1/k! d^k/dx f]_{x=0} = P(k)
    with d^k/dx f being the k-th derivative of f by x

    f'(1) = <k>
    with f' being the first derivative and <k> the mean degree

    [(x d/dx)^m f]_{x=1} = <k^m>
    with <k^m> being the m-th raw moment of P

    Args:
        graph: The graph for which the generating function shall be computed

    x:  float, list, numpy.ndarray
        The argument(s) for which value(s) f(x) shall be computed.

    Example:
    ```py
        # Generate simple network
        import pathpyG as pp
        import numpy as np
        import matplotlib.pyplot as plt
    
        g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'),
                                    ('d', 'e'), ('d', 'f'), ('e', 'f')]).to_undirected()

        # Return single function value
        val = pp.statistics.degreee_generating_func(n, 0.3)
        print(val)
        0.069

        # Plot generating function of degree distribution

        x = np.linspace(0, 1, 20)
        y = pp.statistics.degree_generating_func(n, x)
        x = plt.plot(x, y)
        # [Function plot]

        # Plot generating function based on degree sequence

        x = np.linspace(0, 1, 20)
        y = pp.statistics.degree_generating_func([1,2,1,2], x)
        x = plt.plot(x, y)
        # [Function plot]
    ```
    """

    p_k = degree_distribution(graph, mode=mode)

    if isinstance(x, float):
        x_range = [x]
    else:
        x_range = x

    values: defaultdict = defaultdict(float)
    for k in p_k:
        for v in x_range:
            values[v] += p_k[k] * v**k

    _values: float | _np.ndarray
    if len(x_range) > 1:
        _values = _np.fromiter(values.values(), dtype=float)
    else:
        _values = values[x]
    return _values
