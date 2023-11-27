from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.algorithms.temporal import (
    temporal_graph_to_event_dag,
)
from pathpyG.core.PathData import PathData


def test_constructor():
    p = PathData()
    assert p.num_paths == 0


def test_num_paths(simple_paths):
    assert simple_paths.num_paths == 4


def test_num_nodes(simple_paths):
    assert simple_paths.num_nodes == 5


def test_num_edges(simple_paths):
    assert simple_paths.num_edges == 4


def test_temporal_dag(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=False)
    assert dag.N == 21
    assert dag.M == 20


def test_temporal_dag_sparse(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    assert dag.N == 5
    assert dag.M == 4


def test_path_from_temporal_graph(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = PathData.from_temporal_dag(dag, detect_walks=False)
    assert paths.num_nodes == 5
    assert paths.num_edges == 4
    assert paths.num_paths == 1


def test_edge_index_k_weighted(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = PathData.from_temporal_dag(dag)

    e1, w1 = PathData.edge_index_k_weighted(paths, k=1)

    assert equal(
        e1, IntTensor([[0, 1, 2, 2], [1, 2, 3, 4]]).to(config["torch"]["device"])
    )  # a -> b | b -> c | c -> d | c -> e

    assert equal(w1, tensor([1.0, 1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    e2, w2 = PathData.edge_index_k_weighted(paths, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [2, 4]]]).to(
            config["torch"]["device"]
        ),
    )  # a-b -> b-c | b-c -> c-d | b-c -> c-e

    assert equal(w2, tensor([1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    e3, w3 = PathData.edge_index_k_weighted(paths, k=3)
    assert equal(
        e3,
        IntTensor([[[0, 1, 2], [0, 1, 2]], [[1, 2, 3], [1, 2, 4]]]).to(
            config["torch"]["device"]
        ),
    )

    assert equal(
        w3, tensor([1.0, 1.0]).to(config["torch"]["device"])
    )  # a-b-c -> b-c-d | a-b-c -> b-c-e


def test_edge_index_kth_order_walk():
    edge_index = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    e1 = PathData.edge_index_kth_order_walk(edge_index, k=1)
    assert equal(e1, IntTensor([[[0], [1], [2], [3], [4]], [[1], [2], [3], [4], [5]]]))

    e2 = PathData.edge_index_kth_order_walk(edge_index, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 1], [1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4], [4, 5]]]),
    )

    e3 = PathData.edge_index_kth_order_walk(edge_index, k=3)
    assert equal(
        e3,
        IntTensor(
            [[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]
        ),
    )

    e4 = PathData.edge_index_kth_order_walk(edge_index, k=4)
    assert equal(
        e4, IntTensor([[[0, 1, 2, 3], [1, 2, 3, 4]], [[1, 2, 3, 4], [2, 3, 4, 5]]])
    )

    e5 = PathData.edge_index_kth_order_walk(edge_index, k=5)
    assert equal(e5, IntTensor([[[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 5]]]))


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
