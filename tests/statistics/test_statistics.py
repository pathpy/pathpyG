from __future__ import annotations

import numpy  as _np

from pathpyG.statistics import degree_distribution, degree_raw_moment, degree_sequence, degree_central_moment, degree_generating_function, mean_degree
from pathpyG.statistics.clustering import avg_clustering_coefficient, local_clustering_coefficient, closed_triads


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


def test_mean_degree(toy_example_graph):
    assert _np.isclose(degree_raw_moment(toy_example_graph, k=1), mean_degree(toy_example_graph), atol=1e-6)


def test_clustering_coefficient(toy_example_graph):
    assert local_clustering_coefficient(toy_example_graph, 'a') == 1.
    assert local_clustering_coefficient(toy_example_graph, 'b') == 1/3
    assert local_clustering_coefficient(toy_example_graph, 'f') == 2/3
    assert _np.isclose(avg_clustering_coefficient(toy_example_graph), 0.7619, atol=0.0001)


def test_closed_triads_undirected(toy_example_graph):
    assert closed_triads(toy_example_graph, 'a') == set([('b', 'c'), ('c', 'b')])
    assert closed_triads(toy_example_graph, 'd') == set([('e', 'f'), ('f', 'e'), ('f', 'g'), ('g', 'f')])


def test_closed_triads_directed(toy_example_graph_directed):
    assert closed_triads(toy_example_graph_directed, 'a') == set()
    assert closed_triads(toy_example_graph_directed, 'd') == set([('e', 'f')])
