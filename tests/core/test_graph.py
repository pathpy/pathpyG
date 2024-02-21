from __future__ import annotations

import torch

def test_N(simple_graph):
    assert simple_graph.N == 3

def test_weighted_graph(simple_graph_multi_edges):
    assert simple_graph_multi_edges.M == 4
    weighted_graph = simple_graph_multi_edges.to_weighted_graph()
    assert weighted_graph.M == 3
    assert weighted_graph.N == 3
    assert weighted_graph['edge_weight', 'a', 'b'] == 2

def test_undirected_graph(simple_graph):
    g_u = simple_graph.to_undirected()
    assert g_u.is_edge('a', 'b')
    assert g_u.is_edge('b', 'a')
    assert g_u.is_edge('b', 'c')
    assert g_u.is_edge('c', 'b')
    assert g_u.is_edge('a', 'c')
    assert g_u.is_edge('c', 'a')

def test_M(simple_graph):
    assert simple_graph.M == 3


def test_node_attr(simple_graph):
    simple_graph.data["node_class"] = torch.tensor([[1], [2], [3]])
    assert simple_graph["node_class", "a"].item() == 1


def test_edge_attr(simple_graph):
    simple_graph.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    assert simple_graph["edge_weight", "a", "b"].item() == 1


def test_graph_attr(simple_graph):
    simple_graph.data["graph_feature"] = torch.tensor([42])
    assert simple_graph["graph_feature"].item() == 42


def test_successors(simple_graph):
    s = [v for v in simple_graph.successors("a")]
    assert s == ["b", "c"]


def test_predecessors(simple_graph):
    s = [v for v in simple_graph.predecessors("b")]
    assert s == ["a"]


def test_is_edge(simple_graph):
    assert simple_graph.is_edge("a", "b")
