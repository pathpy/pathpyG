import numpy as _np

from pathpyG.statistics.clustering import avg_clustering_coefficient, closed_triads, local_clustering_coefficient


def test_clustering_coefficient(toy_example_graph):
    assert local_clustering_coefficient(toy_example_graph, "a") == 1.0
    assert local_clustering_coefficient(toy_example_graph, "b") == 1 / 3
    assert local_clustering_coefficient(toy_example_graph, "f") == 2 / 3
    assert _np.isclose(avg_clustering_coefficient(toy_example_graph), 0.7619, atol=0.0001)


def test_closed_triads_undirected(toy_example_graph):
    assert closed_triads(toy_example_graph, "a") == set([("b", "c"), ("c", "b")])
    assert closed_triads(toy_example_graph, "d") == set([("e", "f"), ("f", "e"), ("f", "g"), ("g", "f")])


def test_closed_triads_directed(toy_example_graph_directed):
    assert closed_triads(toy_example_graph_directed, "a") == set()
    assert closed_triads(toy_example_graph_directed, "d") == set([("e", "f")])
