from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.path_data import PathData


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple undirected graph."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('a', 'c'), ('c', 'a')], is_undirected=True)


@pytest.fixture
def simple_graph_sp() -> Graph:
    """Return a undirected graph."""
    return Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'e'), ('b', 'd'), ('d', 'e')]).to_undirected()


@pytest.fixture
def toy_example_graph() -> Graph:
    return Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a'), ('d', 'e'), ('e', 'f'), ('f', 'g'), ('g', 'd'), ('d', 'f'), ('b', 'd')]).to_undirected()

@pytest.fixture
def simple_temporal_graph() -> TemporalGraph:
    """Return a simple temporal graph."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9)]
    return TemporalGraph.from_edge_list(tedges)


@pytest.fixture
def simple_walks() -> PathData:
    paths = PathData(mapping=IndexMap(['A', 'B', 'C', 'D', 'E', 'F']))
    paths.append_walk(('C', 'B', 'D', 'F'), weight=1.0)
    paths.append_walk(('A', 'B', 'D'), weight=1.0)
    paths.append_walk(('D', 'E'), weight=1.0)
    return paths


@pytest.fixture
def long_temporal_graph() -> TemporalGraph:
    """Return a temporal graph with 20 time-stamped edges."""
    tedges = [('a', 'b', 1), ('b', 'c', 5), ('c', 'd', 9), ('c', 'e', 9),
              ('c', 'f', 11), ('f', 'a', 13), ('a', 'g', 18), ('b', 'f', 21),
              ('a', 'g', 26), ('c', 'f', 27), ('h', 'f', 27), ('g', 'h', 28),
              ('a', 'c', 30), ('a', 'b', 31), ('c', 'h', 32), ('f', 'h', 33),
              ('b', 'i', 42), ('i', 'b', 42), ('c', 'i', 47), ('h', 'i', 50)]
    return TemporalGraph.from_edge_list(tedges)
