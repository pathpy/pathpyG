from __future__ import annotations

from pathpyG.core.PathData import GlobalPathStorage

def test_constructor():
    p = GlobalPathStorage()
    assert p.num_paths == 0

def test_num_paths(simple_paths):
    assert simple_paths.num_paths == 4

def test_num_nodes(simple_paths):
    assert simple_paths.num_nodes == 5

def test_num_edges(simple_paths):
    assert simple_paths.num_edges == 8
