from __future__ import annotations

from pathpyG.algorithms.temporal import temporal_graph_to_event_dag
from pathpyG.core.HigherOrderGraph import HigherOrderGraph
from pathpyG.core.PathData import PathData


def test_higher_order_from_temporal_graph(simple_temporal_graph):
    dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    paths = PathData.from_temporal_dag(dag, detect_walks=False)

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
