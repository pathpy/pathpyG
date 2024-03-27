# pylint: disable=missing-function-docstring, missing-module-docstring

from __future__ import annotations

from pathpyG.algorithms.temporal import temporal_graph_to_event_dag
from pathpyG.core.HigherOrderGraph import HigherOrderGraph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.DAGData import DAGData
from pathpyG.core.WalkData import WalkData


def test_higher_order_graph(simple_walks: WalkData):
    g2 = HigherOrderGraph(simple_walks, order=2)
    assert set(g2.nodes) == set([("A", "C"), ("C", "D"), ("B", "C"), ("C", "E")])
    assert set(g2.edges) == set([(("A", "C"), ("C", "D")), (("B", "C"), ("C", "E"))])
    assert g2.N == 4
    assert g2.M == 2


def test_str(simple_walks: WalkData):
    g2 = HigherOrderGraph(simple_walks, order=2)
    assert isinstance(str(g2), str)


def test_higher_order_from_temporal_graph(simple_temporal_graph: TemporalGraph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = DAGData.from_temporal_dag(dag)

    g1 = HigherOrderGraph(paths, order=1)

    assert g1.N == 5
    assert g1.M == 4
    assert g1.data.edge_weight.sum() == 4.0

    g2 = HigherOrderGraph(paths, order=2)

    assert g2.N == 4
    assert g2.M == 3
    assert g2.data.edge_weight.sum() == 3.0

    g3 = HigherOrderGraph(paths, order=3)

    assert g3.N == 3
    assert g3.M == 2
    assert g3.data.edge_weight.sum() == 2.0


def test_successors(simple_walks: WalkData):
    g2 = HigherOrderGraph(simple_walks, order=2)

    assert set([("C", "D")]) == set(g2.successors(("A", "C")))
    assert set([("C", "E")]) == set(g2.successors(("B", "C")))


def test_predecessors(simple_walks: WalkData):
    g2 = HigherOrderGraph(simple_walks, order=2)

    assert set([("A", "C")]) == set(g2.predecessors(("C", "D")))
    assert set([("B", "C")]) == set(g2.predecessors(("C", "E")))


def test_higher_order_index_mapping(simple_walks):
    g2 = HigherOrderGraph(simple_walks, order=2)
    assert g2.mapping.to_idx(("A", "C")) == 0
    assert g2.mapping.to_id(1) == ("B", "C")
