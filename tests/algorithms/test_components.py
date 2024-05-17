from __future__ import annotations

import numpy as _np

from pathpyG.core.Graph import Graph
from pathpyG.algorithms import connected_components

def test_connected_components_undirected_1():
    # undirected graph with two connectecd components
    n = Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('d', 'e'), ('e', 'f'), ('f', 'g'), ('g', 'd'), ('d', 'f')]).to_undirected()
    n, labels = connected_components(n)
    assert n == 2
    assert (labels == _np.array([0, 0, 0, 1, 1, 1, 1])).all()

def test_connected_components_undirected_2():
    # undirected graph with single connected component
    n = Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('d', 'e'), ('e', 'f'),
                              ('f', 'g'), ('g', 'd'), ('d', 'f'), ('c', 'd')]).to_undirected()
    n, labels = connected_components(n)
    assert n == 1
    assert (labels == _np.array([0, 0, 0, 0, 0, 0, 0])).all()

def test_connected_components_directed_1():
    # directed graph with single weak and two strongly connected components
    g = Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('d', 'e'), ('e', 'f'),
                              ('f', 'g'), ('g', 'd'), ('d', 'f'), ('c', 'd')])
    n, labels = connected_components(g, connection='weak')
    assert n == 1
    assert (labels == _np.array([0, 0, 0, 0, 0, 0, 0])).all()

    n, labels = connected_components(g, connection='strong')
    assert n == 2
    assert (labels == _np.array([1, 1, 1, 0, 0, 0, 0])).all()

def test_connected_components_directed_2():
    # directed graph with two weak and two strongly connected components
    g = Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('d', 'e'), ('e', 'f'),
                              ('f', 'g'), ('g', 'd'), ('d', 'f')])
    n, labels = connected_components(g, connection='weak')
    assert n == 2
    assert (labels == _np.array([0, 0, 0, 1, 1, 1, 1])).all()

    n, labels = connected_components(g, connection='strong')
    assert n == 2
    assert (labels == _np.array([0, 0, 0, 1, 1, 1, 1])).all()
