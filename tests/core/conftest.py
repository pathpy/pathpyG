from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.Paths import PathData

@pytest.fixture
def simple_graph():
    return Graph.from_edge_list([['a','b'], ['b','c'], ['a','c']])

@pytest.fixture
def simple_paths():
    paths = PathData()
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E
    return paths

@pytest.fixture
def simple_temporal_graph():
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9)]
    return TemporalGraph.from_edge_list(tedges)