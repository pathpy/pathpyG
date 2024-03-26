from __future__ import annotations

import torch

from pathpyG.core.Graph import Graph
from pathpyG.visualisations.hist_plots import hist
from pathpyG.algorithms import centrality
from pathpyG.algorithms.centrality import path_node_traversals, \
    path_visitation_probabilities, temporal_betweenness_centrality, temporal_closeness_centrality

def test_closeness_centrality(simple_graph):
    r = centrality.closeness_centrality(simple_graph)
    assert r == {'a': 1.0, 'b': 1.0, 'c': 1.0}

def test_betweenness_centrality(simple_graph):
    r = centrality.betweenness_centrality(simple_graph)
    assert r == {'a': 0.0, 'b': 0.0, 'c': 0.0}

def test_node_traversals(simple_dags):
    traversals_dict = path_node_traversals(simple_dags)
    assert set(traversals_dict.keys()) == {0, 1, 2, 3, 4, 5}
    assert traversals_dict[0] == 1
    assert traversals_dict[1] == 2
    assert traversals_dict[2] == 1
    assert traversals_dict[3] == 3
    assert traversals_dict[4] == 1
    assert traversals_dict[5] == 1

def test_visitation_probabilities(simple_dags):
    traversals_dict = path_visitation_probabilities(simple_dags)
    assert set(traversals_dict.keys()) == {0, 1, 2, 3, 4, 5}
    assert traversals_dict[0] == 1/9
    assert traversals_dict[1] == 2/9
    assert traversals_dict[2] == 1/9
    assert traversals_dict[3] == 3/9
    assert traversals_dict[4] == 1/9
    assert traversals_dict[5] == 1/9

def test_temporal_betweenness(long_temporal_graph):
    bw = temporal_betweenness_centrality(long_temporal_graph, delta=5, normalized=False)
    assert bw['a'] == 1.0
    assert bw['c'] == 2.5
    assert bw['g'] == 0.5
    assert bw['d'] == 0
    assert bw['e'] == 0
    assert bw['h'] == 0
    assert bw['i'] == 0
    assert bw['f'] == 1.0


def test_temporal_closeness(long_temporal_graph):
    c = temporal_closeness_centrality(long_temporal_graph, delta=5, normalized=False)
    assert c['b'] == 2.0
    assert c['d'] == 1/3
    assert c['e'] == 1/3
    assert c['g'] == 1/3
    assert c['h'] == 3/2
    assert c['i'] == 3.0
    assert c['a'] == 0.0
    assert c['c'] == 0.0
    assert c['f'] == 3.0

