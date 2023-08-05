from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.PathData import GlobalPathStorage

@pytest.fixture
def simple_graph():
    return Graph.from_edge_list([['a','b'], ['b','c'], ['a','c']])

@pytest.fixture
def simple_paths():
    paths = GlobalPathStorage()
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E
    return paths