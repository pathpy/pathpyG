from __future__ import annotations

import torch

from pathpyG.core.Graph import Graph


def test_N(simple_graph):
    assert simple_graph.N == 3


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
