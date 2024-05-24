from __future__ import annotations

import torch
from torch import IntTensor, equal, tensor, Size

from pathpyG import config
from pathpyG.core.TemporalGraph import TemporalGraph

def test_N(long_temporal_graph):
    assert long_temporal_graph.N == 9

def test_M(long_temporal_graph):
    assert long_temporal_graph.M == 20

def test_shuffle_time(long_temporal_graph):
    g_1 = long_temporal_graph.to_static_graph()
    long_temporal_graph.shuffle_time()
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
