from __future__ import annotations

import pytest

from pathpyG.core.Graph import Graph


@pytest.fixture
def simple_graph() -> Graph:
    """Return a simple example for a graph with a ring topology."""
    return Graph.from_edge_list([
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd'),
        ('d', 'e'),
        ('e', 'f'),
        ('f', 'g'),
        ('g', 'h'),
        ('h', 'i'),
        ('i', 'j'),
        ('j', 'k'),
        ('k', 'l'),
        ('l', 'm'),
        ('m', 'n'),
        ('n', 'o'),
        ('o', 'a')
    ])
