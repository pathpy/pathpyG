# pylint: disable=missing-function-docstring,missing-module-docstring

from __future__ import annotations

import pytest
from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.algorithms.temporal import (
    temporal_graph_to_event_dag,
)
from pathpyG.core.DAGData import DAGData


def test_constructor():
    p = DAGData()
    assert p.num_paths == 0


def test_num_dags(simple_dags):
    assert simple_dags.num_paths == 4


def test_num_nodes_dags(simple_dags):
    assert simple_dags.num_nodes == 5


def test_num_edges_dags(simple_dags):
    assert simple_dags.num_edges == 6


def test_edge_index(simple_dags):
    assert equal(
        simple_dags.edge_index,
        IntTensor([[0, 0, 1, 1, 2, 2], [1, 2, 2, 4, 3, 4]])
    )


def test_edge_index_weighted(simple_dags):
    assert equal(
        simple_dags.edge_index_weighted[0],
        IntTensor([[0, 0, 1, 1, 2, 2], [1, 2, 2, 4, 3, 4]])
    )
    assert equal(
        simple_dags.edge_index_weighted[1],
        IntTensor([1, 2, 1, 1, 3, 2])
    )


def test_dag_mapping():
    mapping = {0: 1, 1: 1, 2: 0, 3: 0, 4: 2, 5: 2}

    e1 = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]).to(config["torch"]["device"])
    m_e1 = DAGData.map_nodes(e1, mapping)

    assert equal(
        m_e1,
        IntTensor([[1, 1, 0, 0, 2], [1, 0, 0, 2, 2]]).to(config["torch"]["device"]),
    )


def test_path_from_temporal_graph(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = DAGData.from_temporal_dag(dag)
    assert paths.num_nodes == 5
    assert paths.num_edges == 4
    assert paths.num_paths == 1


def test_edge_index_k_weighted(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = DAGData.from_temporal_dag(dag)

    e1, w1 = DAGData.edge_index_k_weighted(paths, k=1)

    assert equal(
        e1, IntTensor([[0, 1, 2, 2], [1, 2, 3, 4]]).to(config["torch"]["device"])
    )  # a -> b | b -> c | c -> d | c -> e

    assert equal(w1, tensor([1.0, 1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    e2, w2 = DAGData.edge_index_k_weighted(paths, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [2, 4]]]).to(config["torch"]["device"]),
    )  # a-b -> b-c | b-c -> c-d | b-c -> c-e

    assert equal(w2, tensor([1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    e3, w3 = DAGData.edge_index_k_weighted(paths, k=3)
    assert equal(
        e3,
        IntTensor([[[0, 1, 2], [0, 1, 2]], [[1, 2, 3], [1, 2, 4]]]).to(config["torch"]["device"]),
    )

    assert equal(w3, tensor([1.0, 1.0]).to(config["torch"]["device"]))  # a-b-c -> b-c-d | a-b-c -> b-c-e


def test_edge_index_kth_order_dag():

    edge_index = IntTensor([[0, 1, 1], [1, 2, 3]])
    e1 = DAGData.edge_index_kth_order(edge_index, k=1)
    assert equal(e1, IntTensor([[[0], [1], [1]], [[1], [2], [3]]]))

    e2 = DAGData.edge_index_kth_order(edge_index, k=2)
    assert equal(e2, IntTensor([[[0, 1], [0, 1]], [[1, 2], [1, 3]]]))

    edge_index = IntTensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    e1 = DAGData.edge_index_kth_order(edge_index, k=1)
    assert equal(e1, IntTensor([[[0], [1], [2], [3], [4]], [[1], [2], [3], [4], [5]]]))

    e2 = DAGData.edge_index_kth_order(edge_index, k=2)
    assert equal(
        e2,
        IntTensor([[[0, 1], [1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4], [4, 5]]]),
    )

    e3 = DAGData.edge_index_kth_order(edge_index, k=3)
    assert equal(
        e3,
        IntTensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]),
    )

    e4 = DAGData.edge_index_kth_order(edge_index, k=4)
    assert equal(e4, IntTensor([[[0, 1, 2, 3], [1, 2, 3, 4]], [[1, 2, 3, 4], [2, 3, 4, 5]]]))

    e5 = DAGData.edge_index_kth_order(edge_index, k=5)
    assert equal(e5, IntTensor([[[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 5]]]))


def test_lift_order_dag():
    e1 = tensor([[[0], [1], [1], [3]], [[1], [2], [3], [4]]])
    x = DAGData.lift_order_dag(e1)
    assert equal(x, IntTensor([[[0, 1], [0, 1], [1, 3]], [[1, 2], [1, 3], [3, 4]]]))

    e2 = tensor([[[0, 1], [0, 1], [1, 3]], [[1, 2], [1, 3], [3, 4]]])
    x = DAGData.lift_order_dag(e2)
    assert equal(x, IntTensor([[[0, 1, 3]], [[1, 3, 4]]]))

    e3 = tensor([[[0, 1, 3]], [[1, 3, 4]]])
    x = DAGData.lift_order_dag(e3)
    assert equal(x, IntTensor([]))
