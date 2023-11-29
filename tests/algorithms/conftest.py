from __future__ import annotations

import pytest

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([['a', 'b'], ['b', 'c'], ['a', 'c']])


@pytest.fixture
def simple_temporal_graph() -> TemporalGraph:
    """Return a simple temporal graph."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9)]
    return TemporalGraph.from_edge_list(tedges)
