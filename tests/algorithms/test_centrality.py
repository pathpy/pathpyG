from __future__ import annotations

import torch

from pathpyG.core.Graph import Graph
from pathpyG.visualisations.hist_plots import hist
from pathpyG.algorithms import centrality
from pathpyG.algorithms.centrality import path_node_traversals, \
    path_visitation_probabilities, shortest_paths, path_betweenness_centrality, \
    path_distance_matrix, path_closeness_centrality

def test_centrality(simple_graph):
    r = centrality.closeness_centrality(simple_graph)
    assert r == {'a': 0, 'b': 0.5, 'c': 1}

def test_node_traversals(simple_paths_centralities):
    traversals_dict = path_node_traversals(simple_paths_centralities)
    assert set(traversals_dict.keys()) == {0, 1, 2, 3, 4, 5}
    assert traversals_dict[0] == 1
    assert traversals_dict[1] == 2
    assert traversals_dict[2] == 1
    assert traversals_dict[3] == 3
    assert traversals_dict[4] == 1
    assert traversals_dict[5] == 1

def test_visitation_probabilities(simple_paths_centralities):
    traversals_dict = path_visitation_probabilities(simple_paths_centralities)
    assert set(traversals_dict.keys()) == {0, 1, 2, 3, 4, 5}
    assert traversals_dict[0] == 1/9
    assert traversals_dict[1] == 2/9
    assert traversals_dict[2] == 1/9
    assert traversals_dict[3] == 3/9
    assert traversals_dict[4] == 1/9
    assert traversals_dict[5] == 1/9

def test_shortest_paths(simple_paths_centralities):
    s_p = shortest_paths(simple_paths_centralities)
    # need to check equality of set of tensors
    # checking for paths longer than 1
    assert all(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(s_p[0][3], {torch.tensor([0, 1, 3])}))
    assert all(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(s_p[1][5], {torch.tensor([1, 3, 5])}))
    assert all(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(s_p[2][3], {torch.tensor([2, 1, 3])}))
    assert all(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(s_p[2][5], {torch.tensor([2, 1, 3, 5])}))

def test_betweenness_paths(simple_paths_centralities):
    bw = path_betweenness_centrality(simple_paths_centralities, normalized=False)
    # 1 is in the shortest path between 0-5,2-3,2-5
    assert bw[1] == 3.0
    # 1 is in the shortest path between 2-5,1-5
    assert bw[3] == 2.0

def test_distance_matrix_paths(simple_paths_centralities):
    dm = path_distance_matrix(simple_paths_centralities)
    assert dm[0] == {0: 0, 1: 1, 3: 2}
    assert dm[1] == {1: 0, 3: 1, 5: 2}
    assert dm[2] == {2: 0, 1: 1, 3: 2, 5: 3}
    assert dm[3] == {3: 0, 4: 1, 5: 1}
    assert dm[4] == {4: 0}
    assert dm[5] == {5: 0}

def test_closeness_paths(simple_paths_centralities):
    c = path_closeness_centrality(simple_paths_centralities, normalized=False)
    assert c[0] == 0.0
    # 1 reachable from 0 and 2 in one step
    assert c[1] == 1/1 + 1/1
    assert c[2] == 0
    # 3 reachable from 1 in one step, from 0 and 3 in two steps
    assert c[3] == 1 + 1/2 + 1/2
    assert c[4] == 1
    # 5 reachable from 3 in one step, from 1 in two steps, from 2 in three steps
    assert c[5] == 1 + 1/2 + 1/3
