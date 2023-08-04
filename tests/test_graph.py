
from __future__ import annotations

import pytest

import torch
from pathpyG.core.Graph import Graph


@pytest.fixture
def test_graph_constructor():

    g = Graph.from_edge_list([['a','b'], ['b','c'], ['a','c']])
    assert g.N==3
    assert g.M==3