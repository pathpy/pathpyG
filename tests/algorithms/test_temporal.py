# pylint: disable=missing-function-docstring,missing-module-docstring

from pathpyG.algorithms.temporal import (
    temporal_graph_to_event_dag,
)


def test_temporal_dag(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=False)
    assert dag.N == 21
    assert dag.M == 20


def test_temporal_dag_sparse(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    assert dag.N == 5
    assert dag.M == 4
