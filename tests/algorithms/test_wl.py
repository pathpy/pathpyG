from __future__ import annotations

from pathpyG.core.Graph import Graph
from pathpyG.algorithms import WeisfeilerLeman_test

def test_WeisfeilerLeman_test_1():
    # isomorphic graphs
    g1 = Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    g2 = Graph.from_edge_list([('y', 'z'), ('x', 'y')])
    test, c1, c2 = WeisfeilerLeman_test(g1, g2)
    assert c1 == c2
    assert test is True

def test_WeisfeilerLeman_test_2():
    # non-isomorphic graphs
    g1 = Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    g2 = Graph.from_edge_list([('y', 'z'), ('x', 'z')])
    test, c1, c2 = WeisfeilerLeman_test(g1, g2)
    assert c1 != c2
    assert test is False

def test_WeisfeilerLeman_test_3():
    # isomorphic graphs
    g1 = Graph.from_edge_list([('a', 'g'), ('a', 'h'), ('a', 'i'),
                               ('b', 'g'), ('b', 'h'), ('b', 'j'),
                               ('c', 'g'), ('c', 'i'), ('c', 'j'),
                               ('d', 'h'), ('d', 'i'), ('d', 'j')]).to_undirected()
    g2 = Graph.from_edge_list([('1', '2'), ('1', '5'), ('1', '4'),
                               ('2', '6'), ('2', '3'),
                               ('3', '7'), ('3', '4'),
                               ('4', '8'),
                               ('5', '6'), ('6', '7'), ('7', '8'), ('8', '5')]).to_undirected()
    test, c1, c2 = WeisfeilerLeman_test(g1, g2)
    print(c1, c2)
    assert c1 == c2
    assert test is True

