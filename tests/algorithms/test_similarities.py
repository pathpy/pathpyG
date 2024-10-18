from __future__ import annotations

import numpy as _np

from pathpyG.core.graph import Graph
from pathpyG.statistics.node_similarities import overlap_coefficient, LeichtHolmeNewman_index, jaccard_similarity, common_neighbors, adamic_adar_index, cosine_similarity, katz_index


def test_common_neighbors(toy_example_graph):
    assert common_neighbors(toy_example_graph, 'c', 'a') == 1
    assert common_neighbors(toy_example_graph, 'a', 'g') == 0
    assert common_neighbors(toy_example_graph, 'd', 'd') == 4
    assert common_neighbors(toy_example_graph, 'f', 'd') == 2


def test_overlap_coefficient(toy_example_graph):
    assert overlap_coefficient(toy_example_graph, 'a', 'b') == 1/2
    assert overlap_coefficient(toy_example_graph, 'd', 'f') == 2/3
    assert overlap_coefficient(toy_example_graph, 'a', 'a') == 1


def test_jaccard_similarity(toy_example_graph):
    assert jaccard_similarity(toy_example_graph, 'a', 'b') == 1/4
    assert jaccard_similarity(toy_example_graph, 'a', 'c') == 1/3
    assert jaccard_similarity(toy_example_graph, 'd', 'e') == 1/5


def test_adamic_adar_index(toy_example_graph):
    assert adamic_adar_index(toy_example_graph, 'e', 'g') == 1.0/_np.log(3) + 1.0/_np.log(4)


def test_cosine_similarity(toy_example_graph):
    assert _np.isclose(cosine_similarity(toy_example_graph, 'c', 'a'), 0.5)
    assert _np.isclose(cosine_similarity(toy_example_graph, 'a', 'g'), 0.0)


def test_LeichtHolmeNewman_index(toy_example_graph):
    assert _np.isclose(LeichtHolmeNewman_index(toy_example_graph, 'e', 'g', alpha=0.02), 0.0013079553726262417)
    assert _np.isclose(LeichtHolmeNewman_index(toy_example_graph, 'e', 'g', alpha=0.2), 0.14353902083713282)


def test_katz_index(toy_example_graph):
    assert _np.isclose(katz_index(toy_example_graph, 'e', 'g', beta=0.02), 0.0008178287973506426)
    assert _np.isclose(katz_index(toy_example_graph, 'e', 'g', beta=0.2), 0.12958435772871946)

