from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.DAGData import DAGData


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple directed graph."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c')])


@pytest.fixture
def simple_graph_multi_edges() -> Graph:
    """Return a directed graph with multiple edges."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'b')])


@pytest.fixture
def simple_walks() -> DAGData:
    """Return a simple example for path data."""
    paths = DAGData()
    paths.append(torch.tensor([[0, 2], [2, 3]]))  # A -> C -> D
    paths.append(torch.tensor([[0, 2], [2, 3]]))  # A -> C -> D
    paths.append(torch.tensor([[1, 2], [2, 4]]))  # B -> C -> E
    paths.append(torch.tensor([[1, 2], [2, 4]]))  # B -> C -> E
    return paths


@pytest.fixture
def simple_dags() -> DAGData:
    """Return a simple example for path data."""
    paths = DAGData()
    paths.append(torch.tensor([[0, 2], [2, 3]]))  # A -> C, C -> D
    paths.append(torch.tensor([[0, 1], [1, 4]]))  # B -> C, C -> E
    paths.append(torch.tensor([[1, 2, 2], [2, 3, 4]]))  # B -> C, C -> D, C -> E
    paths.append(torch.tensor([[0, 2, 2], [2, 3, 4]]))  # A -> C, C -> D, C -> E
    return paths


@pytest.fixture
def simple_temporal_graph() -> TemporalGraph:
    """Return a simple temporal graph."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9)]
    return TemporalGraph.from_edge_list(tedges)


@pytest.fixture
def long_temporal_graph() -> TemporalGraph:
    """Return a temporal graph with 20 time-stamped edges."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9),
              ('c', 'f', 11), ('f', 'a', 13), ('a', 'g', 18), ('b', 'f', 21),
              ('a', 'g', 26), ('c', 'f', 27), ('h', 'f', 27), ('g', 'h', 28),
              ('a', 'c', 30), ('a', 'b', 31), ('c', 'h', 32), ('f', 'h', 33),
              ('b', 'i', 42), ('i', 'b', 42), ('c', 'i', 47), ('h', 'i', 50)]
    return TemporalGraph.from_edge_list(tedges)
