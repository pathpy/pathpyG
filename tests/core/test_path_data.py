from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config

from pathpyG.core.path_data import PathData
from pathpyG.core.index_map import IndexMap


def test_constructor():
    p = PathData()
    assert p.num_paths == 0


def test_num_walks(simple_walks):
    assert simple_walks.num_paths == 4


def test_add_walk_seq():

    paths = PathData(IndexMap(['a', 'c', 'b', 'd', 'e']))

    paths.append_walk(('a', 'c', 'd'), weight=1.0)
    paths.append_walk(('a', 'c', 'e'), weight=1.0)
    paths.append_walk(('b', 'c', 'd'), weight=1.0)
    paths.append_walk(('b', 'c', 'e'), weight=1.0)

    assert paths.get_walk(0) == ('a', 'c', 'd')
    assert paths.get_walk(1) == ('a', 'c', 'e')
    assert paths.get_walk(2) == ('b', 'c', 'd')
    assert paths.get_walk(3) == ('b', 'c', 'e')

    assert equal(paths.paths[0].edge_weight.max(), tensor(1.0))
    assert equal(paths.paths[1].edge_weight.max(), tensor(1.0))
    assert equal(paths.paths[2].edge_weight.max(), tensor(1.0))
    assert equal(paths.paths[3].edge_weight.max(), tensor(1.0))


def test_add_walk_seqs():
    paths = PathData(IndexMap(['a', 'c', 'b', 'd', 'e']))
    paths.append_walks([('a', 'c', 'd'), ('a', 'c', 'e'), ('b', 'c', 'd'), ('b', 'c', 'e')], weights=[1.0]*4)

    assert paths.get_walk(0) == ('a', 'c', 'd')
    assert paths.get_walk(1) == ('a', 'c', 'e')
    assert paths.get_walk(2) == ('b', 'c', 'd')
    assert paths.get_walk(3) == ('b', 'c', 'e')

    assert equal(paths.paths[0].edge_weight.max(), tensor(1.0))
