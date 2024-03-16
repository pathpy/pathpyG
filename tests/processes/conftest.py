from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.WalkData import WalkData
from pathpyG.core.HigherOrderGraph import HigherOrderGraph

@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple example for a graph with a ring topology."""
    return Graph.from_edge_list([
        ['a', 'b'],
        ['b', 'c'],
        ['c', 'd'],
        ['d', 'e'],
        ['e', 'f'],
        ['f', 'g'],
        ['g', 'h'],
        ['h', 'i'],
        ['i', 'j'],
        ['j', 'k'],
        ['k', 'l'],
        ['l', 'm'],
        ['m', 'n'],
        ['n', 'o'],
        ['o', 'a']
    ])

@pytest.fixture
def simple_second_order_graph() -> Tuple[Graph, HigherOrderGraph]:
    """Return a simple second-order graph."""
    g = Graph.from_edge_list([
        ['a','b'],
        ['b','c'],
        ['c','a'],
        ['c','d'],
        ['d','a']
        ])

    g.data['edge_weight'] = torch.tensor([[1], [1], [2], [1], [1]])

    paths = WalkData(g.mapping)
    paths.add_walk_seq(['a', 'b', 'c'], freq=1)
    paths.add_walk_seq(['b', 'c', 'a'], freq=1)
    paths.add_walk_seq(['b', 'c', 'd'], freq=0.2)
    paths.add_walk_seq(['c', 'a', 'b'], freq=1)
    paths.add_walk_seq(['c', 'd', 'a'], freq=0.2)
    paths.add_walk_seq(['d', 'a', 'b'], freq=1)

    return (g, HigherOrderGraph(paths, order = 2))