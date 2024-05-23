from __future__ import annotations

import pytest
import torch

from pathpyG.core.graph import Graph


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple undirected graph."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('b', 'd'),
                                 ('d', 'b'), ('d', 'e'), ('e', 'd'), ('c', 'e'), ('e', 'c')], is_undirected=True)