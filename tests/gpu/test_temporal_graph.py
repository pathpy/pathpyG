from __future__ import annotations

import pytest
from pathpyG.core.TemporalGraph import TemporalGraph


@pytest.mark.gpu
def test_to_static_graph(long_temporal_graph, gpu):
    long_temporal_graph.to(gpu)

    # TODO: change to weighted simple graph
    g = long_temporal_graph.to_static_graph()
    assert g.N == long_temporal_graph.N
    assert g.M == long_temporal_graph.M
    assert g.data.edge_index.device == gpu


@pytest.mark.gpu
def test_get_window(long_temporal_graph, gpu):
    long_temporal_graph.to(gpu)

    t_1 = long_temporal_graph.get_window(1, 9)
    assert t_1.N == 7
    assert t_1.M == 8
    t_2 = long_temporal_graph.get_window(9, 13)
    # N defined by largest node index!
    assert t_2.N == 8
    assert t_2.M == 4

    assert t_1.data.edge_index.device == gpu
    assert t_2.data.edge_index.device == gpu
