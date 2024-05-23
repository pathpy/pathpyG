from __future__ import annotations

import torch

from pathpyG.core.graph import Graph
from pathpyG.visualisations.hist_plots import hist
from pathpyG.algorithms import centrality
from pathpyG.algorithms.centrality import (
    path_node_traversals,
    path_visitation_probabilities,
    temporal_betweenness_centrality,
    temporal_closeness_centrality,
)


def test_closeness_centrality(simple_graph):
    r = centrality.closeness_centrality(simple_graph)
    assert r == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_betweenness_centrality(simple_graph):
    r = centrality.betweenness_centrality(simple_graph)
    assert r == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_node_traversals(simple_walks):
    print(simple_walks)
    traversals_dict = path_node_traversals(simple_walks)
    assert set(traversals_dict.keys()) == {'A', 'B', 'C', 'D', 'E', 'F'}
    assert traversals_dict['A'] == 1
    assert traversals_dict['B'] == 2
    assert traversals_dict['C'] == 1
    assert traversals_dict['D'] == 3
    assert traversals_dict['E'] == 1
    assert traversals_dict['F'] == 1


def test_visitation_probabilities(simple_walks):
    visitations_dict = path_visitation_probabilities(simple_walks)
    assert set(visitations_dict.keys()) == {'A', 'B', 'C', 'D', 'E', 'F'}
    assert visitations_dict['A'] == 1 / 9
    assert visitations_dict['B'] == 2 / 9
    assert visitations_dict['C'] == 1 / 9
    assert visitations_dict['D'] == 3 / 9
    assert visitations_dict['E'] == 1 / 9
    assert visitations_dict['F'] == 1 / 9


def test_temporal_betweenness(long_temporal_graph):
    bw = temporal_betweenness_centrality(long_temporal_graph, delta=5)
    assert bw["a"] == 2.0
    assert bw["b"] == 2.0
    assert bw["c"] == 4.5
    assert bw["d"] == 0
    assert bw["e"] == 0
    assert bw["f"] == 2.0
    assert bw["g"] == 0.5
    assert bw["h"] == 0
    assert bw["i"] == 0


def test_temporal_closeness(long_temporal_graph):
    c = temporal_closeness_centrality(long_temporal_graph, delta=5)
    assert c == {
        "a": 12.0,
        "b": 16.0,
        "c": 16.0,
        "d": 14.666666666666666,
        "e": 14.666666666666666,
        "f": 24.0,
        "g": 14.666666666666666,
        "h": 28.0,
        "i": 24.0,
    }
