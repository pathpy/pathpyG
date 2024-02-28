# pylint: disable=missing-module-docstring,missing-function-docstring

import torch
from pathpyG.algorithms.RandomGraphs import Watts_Strogatz


def test_watts_strogatz_simple():
    g = Watts_Strogatz(5, 1, 0.0)
    assert g.M == 10
    assert g.data.edge_index.tolist() == [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 4, 0, 2, 1, 3, 2, 4, 0, 3]]

    torch.manual_seed(1)
    g = Watts_Strogatz(5, 1, 0.5)
    print(g.data.edge_index)
    assert g.M == 10
    assert g.N == 5


def test_watts_strogatz():
    g = Watts_Strogatz(1000, 400, 0.0)
    m = g.M

    g = Watts_Strogatz(1000, 400, 0.8)
    assert g.N == 1000
    assert g.M == m
    assert torch.unique(g.data.edge_index, dim=0).size(1) == g.data.edge_index.size(1)
