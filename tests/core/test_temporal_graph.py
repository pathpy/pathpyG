from __future__ import annotations

import numpy as np
import torch
from torch import equal
from torch_geometric.data import Data

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils import to_numpy


def test_init():
    tdata = Data(edge_index=torch.IntTensor([[1, 3, 2, 4], [2, 4, 3, 5]]), time=torch.Tensor([1000, 1100, 1010, 2000]))
    tgraph = TemporalGraph(tdata)
    print(to_numpy(tgraph.data.edge_index))
    # After ordering the edges by time
    assert (to_numpy(tgraph.data.edge_index) == np.array([[1, 2, 3, 4], [2, 3, 4, 5]])).all()
    assert equal(tgraph.data.time, torch.tensor([1000, 1010, 1100, 2000]))


def test_from_edge_list():
    tedges = [("a", "b", 1), ("b", "c", 5), ("c", "d", 9), ("c", "e", 9)]
    tgraph = TemporalGraph.from_edge_list(tedges)
    print(tgraph.data.time)
    assert tgraph.N == 5
    assert tgraph.M == 4
    assert tgraph.start_time == 1
    assert tgraph.end_time == 9
    assert tgraph.data.edge_index.shape == (2, 4)


def test_csv(tmp_path, long_temporal_graph):
    path = tmp_path / "test.csv"
    long_temporal_graph.to_csv(path)
    tgraph = TemporalGraph.from_csv(path)
    assert tgraph.N == 9
    assert tgraph.M == 20


def test_N(long_temporal_graph):
    assert long_temporal_graph.N == 9


def test_M(long_temporal_graph):
    assert long_temporal_graph.M == 20


def test_temporal_edges(long_temporal_graph):
    for i, (u, v, t) in enumerate(long_temporal_graph.temporal_edges):
        assert u == long_temporal_graph.mapping.to_id(long_temporal_graph.data.edge_index[0, i])
        assert v == long_temporal_graph.mapping.to_id(long_temporal_graph.data.edge_index[1, i])
        assert t == long_temporal_graph.data.time[i]


def test_shuffle_time(long_temporal_graph):
    g_1 = long_temporal_graph.to_static_graph()
    long_temporal_graph.shuffle_time()
    assert long_temporal_graph.N == 9
    assert long_temporal_graph.M == 20

    g_2 = long_temporal_graph.to_static_graph()
    assert g_1.N == g_2.N
    assert g_2.M == g_2.M


def test_to_static_graph(long_temporal_graph):
    g = long_temporal_graph.to_static_graph()
    assert g.N == long_temporal_graph.N
    assert g.M == long_temporal_graph.M

    g = long_temporal_graph.to_static_graph(weighted=True)
    assert g.N == long_temporal_graph.N
    assert g.data.edge_weight[0].item() == 2.0  # A -> B is two times in the temporal graph
    assert g.data.edge_weight[1].item() == 1.0  # B -> C is one time in the temporal graph


def test_to_undirected(long_temporal_graph):
    g = long_temporal_graph.to_undirected()
    assert g.N == long_temporal_graph.N
    assert g.M == long_temporal_graph.M * 2


def test_get_window(long_temporal_graph):
    t_1 = long_temporal_graph.get_window(1, 9)
    # N stays the same
    assert t_1.N == 9
    assert t_1.M == 8
    t_2 = long_temporal_graph.get_window(9, 13)
    assert t_2.N == 9
    assert t_2.M == 4


def test_get_snapshot(long_temporal_graph):
    t_1 = long_temporal_graph.get_snapshot(1, 9)
    assert t_1.M == 4
    t_2 = long_temporal_graph.get_snapshot(9, 13)
    assert t_2.M == 4


def test_str(simple_temporal_graph):
    assert str(simple_temporal_graph)
