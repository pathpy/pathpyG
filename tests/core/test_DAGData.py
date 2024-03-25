from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config

from pathpyG.core.DAGData import DAGData
from pathpyG.core.IndexMap import IndexMap


def test_constructor():
    p = DAGData()
    assert p.num_paths == 0


def test_num_walks(simple_walks):
    assert simple_walks.num_paths == 4


def test_num_dags(simple_dags):
    assert simple_dags.num_paths == 4


def test_num_nodes_walks(simple_walks):
    assert simple_walks.num_nodes == 5


def test_num_nodes_dags(simple_dags):
    assert simple_dags.num_nodes == 5    


def test_num_edges_walks(simple_walks):
    assert simple_walks.num_edges == 4


def test_num_edges_dags(simple_dags):
    assert simple_dags.num_edges == 6


def test_add_walk_seq():

    paths = DAGData(IndexMap(['a', 'c', 'b', 'd', 'e']))

    paths.add_walk_seq(('a', 'c', 'd'), freq=1)
    paths.add_walk_seq(('a', 'c', 'e'), freq=1)
    paths.add_walk_seq(('b', 'c', 'd'), freq=1)
    paths.add_walk_seq(('b', 'c', 'e'), freq=1)

    assert equal(paths.paths[0], tensor([[0, 1], [1, 3]]))
    assert equal(paths.paths[1], tensor([[0, 1], [1, 4]]))
    assert equal(paths.paths[2], tensor([[2, 1], [1, 3]]))
    assert equal(paths.paths[3], tensor([[2, 1], [1, 4]]))
    assert paths.path_freq[0] == 1
    assert paths.path_freq[1] == 1
    assert paths.path_freq[2] == 1
    assert paths.path_freq[3] == 1


def test_path_mapping():
    mapping = {0: 1, 1: 1, 2: 0, 3: 0, 4: 2, 5: 2}

    e1 = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]).to(config["torch"]["device"])
    m_e1 = DAGData.map_nodes(e1, mapping)

    assert equal(
        m_e1,
        IntTensor([[1, 1, 0, 0, 2], [1, 0, 0, 2, 2]]).to(config["torch"]["device"]),
    )

    e2 = IntTensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]]).to(config["torch"]["device"])
    m_e2 = DAGData.map_nodes(e2, mapping)

    assert equal(
        m_e2,
        IntTensor([[[1, 1], [0, 0]], [[1, 0], [0, 2]]]).to(config["torch"]["device"]),
    )
