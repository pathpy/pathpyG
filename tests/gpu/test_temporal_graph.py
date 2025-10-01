from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_to_static_graph(long_temporal_graph, gpu):
    long_temporal_graph.to(gpu)

    g = long_temporal_graph.to_static_graph(True)
    assert g.n == long_temporal_graph.n
    assert g.m == 17
    assert g.data.edge_index.device == gpu


@pytest.mark.gpu
def test_get_window(long_temporal_graph, gpu):
    long_temporal_graph.to(gpu)

    t_1 = long_temporal_graph.get_window(1, 9)
    assert t_1.n == 9
    assert t_1.m == 4
    t_2 = long_temporal_graph.get_window(9, 13)
    # N defined by largest node index!
    assert t_2.n == 9
    assert t_2.m == 4

    assert t_1.data.edge_index.device == gpu
    assert t_2.data.edge_index.device == gpu
