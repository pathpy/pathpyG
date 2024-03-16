from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.processes.random_walk import RandomWalk
from pathpyG.core.WalkData import WalkData
from pathpyG.core.HigherOrderGraph import HigherOrderGraph


def test_random_walk(simple_graph):
    rw = RandomWalk(simple_graph)

    steps = 20
    data = rw.run_experiment(steps = steps, runs=[v for v in simple_graph.nodes])

    assert len(data) == simple_graph.N * steps * 2 + simple_graph.N * simple_graph.N

    # make sure that all transitions correspond to edges
    paths = rw.get_paths(data)
    for p in paths.paths:
        w = paths.paths[p]
        for i in range(w.size(1)):
            src = simple_graph.mapping.idx_to_id[w[0][i].item()]
            dst = simple_graph.mapping.idx_to_id[w[1][i].item()]
            assert simple_graph.is_edge(src, dst)
