from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config
from torch_geometric.utils import coalesce

from pathpyG.core.DAGData import DAGData
from pathpyG.core.IndexMap import IndexMap


def test_constructor():
    p = DAGData()
    assert p.num_dags == 0


def test_num_walks(simple_walks):
    assert simple_walks.num_dags == 4


def test_num_dags(simple_dags):
    assert simple_dags.num_dags == 4


def test_add_walk_seq():

    paths = DAGData(IndexMap(['a', 'c', 'b', 'd', 'e']))

    paths.append_walk(('a', 'c', 'd'), weight=1.0)
    paths.append_walk(('a', 'c', 'e'), weight=1.0)
    paths.append_walk(('b', 'c', 'd'), weight=1.0)
    paths.append_walk(('b', 'c', 'e'), weight=1.0)

    assert paths.get_walk(0) == ('a', 'c', 'd')
    assert paths.get_walk(1) == ('a', 'c', 'e')
    assert paths.get_walk(2) == ('b', 'c', 'd')
    assert paths.get_walk(3) == ('b', 'c', 'e')

    assert equal(paths.dags[0].weight, tensor(1.0))
    assert equal(paths.dags[1].weight, tensor(1.0))
    assert equal(paths.dags[2].weight, tensor(1.0))
    assert equal(paths.dags[3].weight, tensor(1.0))
