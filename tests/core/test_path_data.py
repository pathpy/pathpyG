from __future__ import annotations

from torch import equal, tensor

from pathpyG import config

from pathpyG.core.path_data import PathData
from pathpyG.core.IndexMap import IndexMap


def test_constructor():
    p = PathData()
    assert p.num_paths == 0


def test_num_walks(simple_walks):
    assert simple_walks.num_paths == 4


def test_add_walk_seq():

    paths = PathData(IndexMap(['a', 'c', 'b', 'd', 'e']))

    paths.append_walk(('a', 'c', 'd'), weight=1.0)
    paths.append_walk(('a', 'c'), weight=1.0)
    paths.append_walk(('b', 'c', 'd'), weight=1.5)
    paths.append_walk(('b', 'c', 'e'), weight=1.0)

    assert paths.num_paths == 4
    assert paths.get_walk(0) == ('a', 'c', 'd')
    assert paths.get_walk(1) == ('a', 'c')
    assert paths.get_walk(2) == ('b', 'c', 'd')
    assert paths.get_walk(3) == ('b', 'c', 'e')

    assert equal(paths.data.dag_weight, tensor([1.0, 1.0, 1.5, 1.0]))
    assert paths.data.dag_weight.shape[0] == 4
    assert equal(paths.data.dag_num_nodes, tensor([3, 2, 3, 3]))
    assert equal(paths.data.dag_num_edges, tensor([2, 1, 2, 2]))


def test_add_walk_seqs():
    paths = PathData(IndexMap(['a', 'c', 'b', 'd', 'e']))
    paths.append_walks([('a', 'c', 'd'), ('a', 'c'), ('b', 'c', 'd'), ('b', 'c', 'e')], weights=[1.0]*4)

    assert paths.num_paths == 4
    assert paths.get_walk(0) == ('a', 'c', 'd')
    assert paths.get_walk(1) == ('a', 'c')
    assert paths.get_walk(2) == ('b', 'c', 'd')
    assert paths.get_walk(3) == ('b', 'c', 'e')

    assert equal(paths.data.dag_weight, tensor([1.0]*4))
    assert paths.data.dag_weight.shape[0] == 4
    assert equal(paths.data.dag_num_nodes, tensor([3, 2, 3, 3]))
    assert equal(paths.data.dag_num_edges, tensor([2, 1, 2, 2]))
