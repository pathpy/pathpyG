# pylint: disable=missing-module-docstring,missing-function-docstring

import pytest
import torch
import numpy as np
from torch_geometric.utils import to_undirected, sort_edge_index

import pathpyG as pp
from pathpyG.algorithms.random_graphs import Watts_Strogatz
from pathpyG.utils import to_numpy


def test_watts_strogatz_simple():
    g = Watts_Strogatz(5, 1, 0.0)
    assert g.m == 10
    assert (
        to_numpy(g.data.edge_index) == np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 4, 0, 2, 1, 3, 2, 4, 0, 3]])
    ).all()

    torch.manual_seed(1)
    g = Watts_Strogatz(5, 1, 0.5, allow_duplicate_edges=False, allow_self_loops=False)
    print(g.data.edge_index)
    assert g.m == 10
    assert g.n == 5
    assert g.has_self_loops() == False
    assert g.is_directed() == False

    g = Watts_Strogatz(5, 1, 0.5, allow_duplicate_edges=False, allow_self_loops=False, undirected=False)
    assert g.m == 5
    assert g.n == 5
    assert g.has_self_loops() == False
    assert g.is_directed() == True


def test_watts_strogatz():
    g = Watts_Strogatz(1000, 5, 0.0)
    m = g.m

    for _ in range(3):
        g = Watts_Strogatz(1000, 5, 0.5, allow_duplicate_edges=False, allow_self_loops=False)
        assert g.n == 1000
        assert g.m == m
        assert torch.unique(g.data.edge_index, dim=0).size(1) == g.data.edge_index.size(1)
        assert g.has_self_loops() == False


def test_watts_strogatz_rewiring():
    torch.manual_seed(1)
    n = 1000
    s = 1
    p = 0.5

    nodes = torch.arange(n)
    edges = (
        torch.stack([torch.stack((nodes, torch.roll(nodes, shifts=-i, dims=0))) for i in range(1, s + 1)], dim=0)
        .permute(1, 0, 2)
        .reshape(2, -1)
    )
    # edges = to_undirected(edges)

    g = Watts_Strogatz(n, s, p, allow_duplicate_edges=True, allow_self_loops=True, undirected=False)

    sorted_edges_g = sort_edge_index(g.data.edge_index.as_tensor())
    sorted_edges = sort_edge_index(edges)

    ratio = (sorted_edges_g[1] == sorted_edges[1]).sum() / edges.size(1)
    assert ratio > 0.4
    assert ratio < 0.6


def test_watts_strogatz_get_warning():
    with pytest.raises(ValueError):
        print(Watts_Strogatz(5, 10, 0.5, allow_duplicate_edges=False))

    with pytest.warns(Warning):
        print(Watts_Strogatz(10, 5, 0.31, allow_duplicate_edges=False))
