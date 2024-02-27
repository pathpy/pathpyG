# pylint: disable=missing-function-docstring,missing-module-docstring

from __future__ import annotations

import pytest
from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.core.PathData import PathData
from pathpyG.core.WalkData import WalkData
from pathpyG.core.IndexMap import IndexMap


def test_constructor():
    p = WalkData()
    assert p.num_paths == 0


def test_num_walks(simple_walks):
    assert simple_walks.num_paths == 4


def test_num_nodes_walks(simple_walks):
    assert simple_walks.num_nodes == 5


def test_num_edges_walks(simple_walks):
    assert simple_walks.num_edges == 4


def test_edge_index(simple_walks):
    assert equal(
        simple_walks.edge_index,
        IntTensor([[0, 1, 2, 2], [2, 2, 3, 4]])
    )


def test_edge_index_weighted(simple_walks):
    simple_walks.add(IntTensor([[0], [2]]))  # A -> C
    assert equal(
        simple_walks.edge_index_weighted[0],
        IntTensor([[0, 1, 2, 2], [2, 2, 3, 4]])
    )
    assert equal(
        simple_walks.edge_index_weighted[1],
        IntTensor([3, 2, 2, 2])
    )


def test_path_mapping():
    mapping = {0: 1, 1: 1, 2: 0, 3: 0, 4: 2, 5: 2}

    e1 = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]).to(config["torch"]["device"])
    m_e1 = PathData.map_nodes(e1, mapping)

    assert equal(
        m_e1,
        IntTensor([[1, 1, 0, 0, 2], [1, 0, 0, 2, 2]]).to(config["torch"]["device"]),
    )

    e2 = IntTensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]]).to(config["torch"]["device"])
    m_e2 = PathData.map_nodes(e2, mapping)

    assert equal(
        m_e2,
        IntTensor([[[1, 1], [0, 0]], [[1, 0], [0, 2]]]).to(config["torch"]["device"]),
    )


def test_add():
    p = WalkData()
    p.add(IntTensor([[0, 1], [1, 2]]), freq=2)
    p.add(IntTensor([[0, 1], [1, 2]]), freq=2)
    p.add(IntTensor([[1, 2], [2, 3]]), freq=1)
    p.add(IntTensor([[1, 2], [2, 3]]), freq=1)

    assert equal(p.paths[0], tensor([[0, 1], [1, 2]]))
    assert equal(p.paths[1], tensor([[0, 1], [1, 2]]))
    assert equal(p.paths[2], tensor([[1, 2], [2, 3]]))
    assert equal(p.paths[3], tensor([[1, 2], [2, 3]]))
    assert p.path_freq[0] == 2
    assert p.path_freq[1] == 2
    assert p.path_freq[2] == 1
    assert p.path_freq[3] == 1


def test_add_walk_seq():

    paths = WalkData(IndexMap(["a", "c", "b", "d", "e"]))

    paths.add_walk_seq(("a", "c", "d"), freq=1)
    paths.add_walk_seq(("a", "c", "e"), freq=1)
    paths.add_walk_seq(("b", "c", "d"), freq=1)
    paths.add_walk_seq(("b", "c", "e"), freq=1)

    assert equal(paths.paths[0], tensor([[0, 1], [1, 3]]))
    assert equal(paths.paths[1], tensor([[0, 1], [1, 4]]))
    assert equal(paths.paths[2], tensor([[2, 1], [1, 3]]))
    assert equal(paths.paths[3], tensor([[2, 1], [1, 4]]))
    assert paths.path_freq[0] == 1
    assert paths.path_freq[1] == 1
    assert paths.path_freq[2] == 1
    assert paths.path_freq[3] == 1


def test_str(simple_walks):
    assert str(simple_walks) == "WalkData with 4 walks and total weight 4"


def test_walk_to_node_seq():
    s = WalkData.walk_to_node_seq(IntTensor([[0, 2], [2, 3]]))
    assert equal(s, IntTensor([0, 2, 3]))


def test_edge_index_kth_order_walk():
    edge_index = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    e1 = WalkData.edge_index_kth_order(edge_index, k=1)
    assert equal(e1, IntTensor([[[0], [1], [2], [3], [4]], [[1], [2], [3], [4], [5]]]))

    e2 = WalkData.edge_index_kth_order(edge_index, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 1], [1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4], [4, 5]]]),
    )

    e3 = WalkData.edge_index_kth_order(edge_index, k=3)
    assert equal(
        e3,
        IntTensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]),
    )

    e4 = WalkData.edge_index_kth_order(edge_index, k=4)
    assert equal(e4, IntTensor([[[0, 1, 2, 3], [1, 2, 3, 4]], [[1, 2, 3, 4], [2, 3, 4, 5]]]))

    e5 = WalkData.edge_index_kth_order(edge_index, k=5)
    assert equal(e5, IntTensor([[[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 5]]]))


def test_edge_index_k_weighted(simple_walks):
    e2, w2 = WalkData.edge_index_k_weighted(simple_walks, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 2], [1, 2]], [[2, 3], [2, 4]]])
    )
    assert equal(w2, tensor([2.0, 2.0]))

    with pytest.raises(ValueError):
        # This should raise an error because the longest path is of length 2
        _, _ = WalkData.edge_index_k_weighted(simple_walks, k=3)
