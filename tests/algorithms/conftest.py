from __future__ import annotations

import pytest
import torch

from pathpyG import Graph
from pathpyG import TemporalGraph
from pathpyG import PathData


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([['a', 'b'], ['b', 'c'], ['a', 'c']])


@pytest.fixture
def simple_temporal_graph() -> TemporalGraph:
    """Return a simple temporal graph."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9)]
    return TemporalGraph.from_edge_list(tedges)


@pytest.fixture
def simple_paths_centralities() -> PathData:
    paths = PathData()
    paths.add_walk(torch.tensor([[2, 1, 3], [1, 3, 5]]))
    paths.add_walk(torch.tensor([[0, 1], [1, 3]]))
    paths.add_walk(torch.tensor([[3], [4]]))
    return paths
