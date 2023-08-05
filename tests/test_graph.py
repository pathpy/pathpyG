
from __future__ import annotations

from pathpyG.core.Graph import Graph

def test_N(simple_graph):
    assert simple_graph.N == 3

def test_M(simple_graph):
    assert simple_graph.M == 3