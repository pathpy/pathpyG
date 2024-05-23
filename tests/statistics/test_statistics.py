from __future__ import annotations

import numpy  as _np

from pathpyG.core.graph import Graph
from pathpyG.statistics import degree_distribution, degree_raw_moment, degree_sequence, degree_central_moment, degree_generating_function


def test_degree_distribution(simple_graph):
    dist = degree_distribution(simple_graph)
    assert _np.isclose(dist[1], 1/5)
    assert _np.isclose(dist[2], 3/5)
    assert _np.isclose(dist[3], 1/5)

def test_degree_sequence(simple_graph):
    seq = degree_sequence(simple_graph)
    assert (seq == _np.array([1.0, 3.0, 2.0, 2.0, 2.0])).all()

def test_degree_raw_moment(simple_graph):
    k_1 = degree_raw_moment(simple_graph, k=1)
    assert k_1 == 2.0
    k_2 = degree_raw_moment(simple_graph, k=2)
    assert k_2 == 4.4
    k_3 = degree_raw_moment(simple_graph, k=3)
    assert _np.isclose(k_3, 10.4)

def test_degree_central_moment(simple_graph):
    k_1 = degree_central_moment(simple_graph, k=1)
    assert k_1 == 0.0
    k_2 = degree_central_moment(simple_graph, k=2)
    assert k_2 == 0.4
    k_3 = degree_central_moment(simple_graph, k=3)
    assert _np.isclose(k_3, 0.0)

def test_degree_generating_function(simple_graph):
    y = degree_generating_function(simple_graph, x=0.5)
    assert y == 0.275
    y = degree_generating_function(simple_graph, x=_np.array([0, 0.5, 1.0]))
    assert (y == _np.array([0, 0.275, 1.0])).all()
