"""This module contains fixtures for testing the io module of pathpyG."""

import pytest
from pathpyG import Graph, TemporalGraph


@pytest.fixture
def backward_idx() -> list[int]:
    """Return a backward index."""
    return [2, 1, 0]


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c")])


@pytest.fixture
def simple_temporal_graph() -> TemporalGraph:
    """Return a simple temporal graph."""
    tedges = [("a", "b", 1), ("b", "c", 5), ("c", "d", 9), ("c", "e", 9)]
    return TemporalGraph.from_edge_list(tedges)
