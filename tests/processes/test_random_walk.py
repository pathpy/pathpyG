from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.processes.random_walk import RandomWalk, HigherOrderRandomWalk
from pathpyG.core.path_data import PathData
from pathpyG.core.graph import Graph
from pathpyG.core.multi_order_model import MultiOrderModel

def check_transitions(g, paths: PathData):
    for i in range(paths.num_paths):
        w = paths.get_walk(i)
        for j in range(len(w)-1):
            assert g.is_edge(w[j], w[j+1])

def test_random_walk(simple_graph):
    rw = RandomWalk(simple_graph)

    steps = 20
    data = rw.run_experiment(steps=steps, runs=[v for v in simple_graph.nodes])

    assert len(data) == simple_graph.n * steps * 2 + simple_graph.n * simple_graph.n

    # make sure that all transitions correspond to edges
    paths = rw.get_paths(data)
    check_transitions(simple_graph, paths)

def test_transition_matrix(simple_graph):
    rw = RandomWalk(simple_graph)

    assert (rw.transition_matrix.data == 1.).all()
    assert rw.transition_probabilities("a")[1] == 1.0

def test_higher_order_random_walk(simple_second_order_graph: Tuple[Graph, Graph]):
    g = simple_second_order_graph[0]
    g2 = simple_second_order_graph[1]
    print(g2.mapping)
    rw = HigherOrderRandomWalk(g2, g, weight=True)
    steps = 100
    data = rw.run_experiment(steps=steps, runs=g2.nodes)

    assert len(data) == g2.n * steps * 2 + g2.n * g2.n
    paths = rw.get_paths(data)
    check_transitions(g, paths)

    # rw.first_order_stationary_state()
