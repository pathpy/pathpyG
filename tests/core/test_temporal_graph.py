# pylint: disable=missing-function-docstring,missing-module-docstring

from __future__ import annotations

from torch import IntTensor, equal, tensor, Size
from torch_geometric.data import TemporalData

from pathpyG import Graph
from pathpyG.core.TemporalGraph import TemporalGraph


def test_init():
    tdata = TemporalData(
        src=IntTensor([1, 3, 2, 4]),
        dst=IntTensor([2, 4, 3, 5]),
        t=IntTensor([1000, 1100, 1010, 2000]),
    )
    tgraph = TemporalGraph(tdata)
    assert equal(tgraph.data.src, tensor([1, 2, 3, 4]))
    assert equal(tgraph.data.dst, tensor([2, 3, 4, 5]))
    assert equal(tgraph.data.t, tensor([1000, 1010, 1100, 2000]))


def test_from_edge_list():
    tedges = [("a", "b", 1), ("b", "c", 5), ("c", "d", 9), ("c", "e", 9)]
    tgraph = TemporalGraph.from_edge_list(tedges)


def test_N(long_temporal_graph):
    assert long_temporal_graph.N == 9


def test_M(long_temporal_graph):
    assert long_temporal_graph.M == 20


def test_temporal_edges(simple_temporal_graph):
    for src, dst, t in simple_temporal_graph.temporal_edges:
        assert src in simple_temporal_graph.mapping.node_ids
        assert dst in simple_temporal_graph.mapping.node_ids
        assert t in simple_temporal_graph.data.t


def test_shuffle_time(long_temporal_graph):
    g_1 = long_temporal_graph.to_static_graph()
    ordered_t = long_temporal_graph.data.t
    long_temporal_graph.shuffle_time()
    shuffled_t = long_temporal_graph.data.t
    assert not equal(ordered_t, shuffled_t)
    assert long_temporal_graph.N == 9
    assert long_temporal_graph.M == 20

    g_2 = long_temporal_graph.to_static_graph()
    assert g_1.N == g_2.N
    assert g_2.M == g_2.M


def test_to_static_graph(long_temporal_graph):
    # TODO: change to weighted simple graph
    g = long_temporal_graph.to_static_graph()
    assert g.N == long_temporal_graph.N
    assert g.M == long_temporal_graph.M


def test_get_window(long_temporal_graph):
    t_1 = long_temporal_graph.get_window(1, 9)
    assert t_1.N == 7
    assert t_1.M == 8
    t_2 = long_temporal_graph.get_window(9, 13)
    # N defined by largest node index!
    assert t_2.N == 8
    assert t_2.M == 4


def test_get_snapshot(long_temporal_graph):
    # TODO: test more elaborately
    snapshot = long_temporal_graph.get_snapshot(1, 9)
    assert isinstance(snapshot, Graph)
    assert not isinstance(snapshot, TemporalGraph)


def test_str(long_temporal_graph):
    assert isinstance(str(long_temporal_graph), str)
