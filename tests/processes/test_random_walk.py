from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.processes.random_walk import RandomWalk, HigherOrderRandomWalk
from pathpyG.core.WalkData import WalkData
from pathpyG.core.Graph import Graph
from pathpyG.core.HigherOrderGraph import HigherOrderGraph

def check_transitions(g, paths):
    for p in paths.paths:
        w = paths.paths[p]
        for i in range(w.size(1)):
            src = g.mapping.idx_to_id[w[0][i].item()]
            dst = g.mapping.idx_to_id[w[1][i].item()]
            assert g.is_edge(src, dst)

def test_random_walk(simple_graph):
    rw = RandomWalk(simple_graph)

    steps = 20
    data = rw.run_experiment(steps = steps, runs=[v for v in simple_graph.nodes])

    assert len(data) == simple_graph.N * steps * 2 + simple_graph.N * simple_graph.N

    # make sure that all transitions correspond to edges
    paths = rw.get_paths(data)
    check_transitions(simple_graph, paths)

def test_transition_matrix(simple_graph):
    rw = RandomWalk(simple_graph)

    assert (rw.transition_matrix.data == 1.).all()

def test_higher_order_random_walk(simple_second_order_graph: Tuple[Graph, HigherOrderGraph]):
    g = simple_second_order_graph[0]
    g2 = simple_second_order_graph[1]
    rw = HigherOrderRandomWalk(g2, g, weight=True)
    steps = 100
    data = rw.run_experiment(steps=steps, runs=list(g2.nodes))

    assert len(data) == g2.N * steps * 2 + g2.N * g2.N
    paths = rw.get_paths(data)
    check_transitions(g, paths)
