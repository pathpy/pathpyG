import pytest

import pytest
import torch

from pathpyG.core.Graph import Graph

@pytest.fixture
def gpu() -> torch.device:
    assert torch.cuda.is_available()
    return torch.device('cuda:0')


@pytest.fixture
def cpu() -> torch.device:
    return torch.device('cpu')


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c')])