import pytest
import torch

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph


@pytest.fixture
def gpu() -> torch.device:
    assert torch.cuda.is_available()
    return torch.device("cuda:0")


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c")])


@pytest.fixture
def long_temporal_graph() -> TemporalGraph:
    """Return a temporal graph with 20 time-stamped edges."""
    tedges = [
        ("a", "b", 1),
        ("b", "c", 5),
        ("c", "d", 9),
        ("c", "e", 9),
        ("c", "f", 11),
        ("f", "a", 13),
        ("a", "g", 18),
        ("b", "f", 21),
        ("a", "g", 26),
        ("c", "f", 27),
        ("h", "f", 27),
        ("g", "h", 28),
        ("a", "c", 30),
        ("a", "b", 31),
        ("c", "h", 32),
        ("f", "h", 33),
        ("b", "i", 42),
        ("i", "b", 42),
        ("c", "i", 47),
        ("h", "i", 50),
    ]
    return TemporalGraph.from_edge_list(tedges)
