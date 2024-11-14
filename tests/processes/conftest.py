from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import pytest
import torch

from pathpyG.core.graph import Graph
from pathpyG.core.path_data import PathData
from pathpyG.core.multi_order_model import MultiOrderModel


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple example for a graph with a ring topology."""
    return Graph.from_edge_list(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("f", "g"),
            ("g", "h"),
            ("h", "i"),
            ("i", "j"),
            ("j", "k"),
            ("k", "l"),
            ("l", "m"),
            ("m", "n"),
            ("n", "o"),
            ("o", "a"),
        ]
    )


@pytest.fixture
def simple_second_order_graph() -> Tuple[Graph, Graph]:
    """Return a simple second-order graph."""
    g = Graph.from_edge_list([["a", "b"], ["b", "c"], ["c", "a"], ["c", "d"], ["d", "a"]])

    g.data["edge_weight"] = torch.tensor([[1], [1], [2], [1], [1]])

    paths = PathData(g.mapping)
    paths.append_walk(["a", "b", "c"], weight=1)
    paths.append_walk(["b", "c", "a"], weight=1)
    paths.append_walk(["b", "c", "d"], weight=0.2)
    paths.append_walk(["c", "a", "b"], weight=1)
    paths.append_walk(["c", "d", "a"], weight=0.2)
    paths.append_walk(["d", "a", "b"], weight=1)

    m = MultiOrderModel.from_PathData(paths, max_order=2)
    return (g, m.layers[2])
