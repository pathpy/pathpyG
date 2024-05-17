from __future__ import annotations

from pathpyG.core.Graph import Graph
from pathpyG.algorithms import WeisfeilerLeman_test

def test_WeisfeilerLeman_test_1():
    g1 = Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    g2 = Graph.from_edge_list([('y', 'z'), ('x', 'y')])
    test, c1, c2 = WeisfeilerLeman_test(g1, g2)
    assert c1 == c2
    assert test is True

def test_WeisfeilerLeman_test_2():
    g1 = Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    g2 = Graph.from_edge_list([('y', 'z'), ('x', 'z')])
    test, c1, c2 = WeisfeilerLeman_test(g1, g2)
    assert c1 != c2
    assert test is False
