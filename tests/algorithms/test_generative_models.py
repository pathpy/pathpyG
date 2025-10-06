from collections import Counter

import numpy as _np
import scipy

import pytest
import torch
import numpy as np
from torch_geometric.utils import sort_edge_index

from pathpyG.algorithms.components import connected_components
from pathpyG.utils import to_numpy

from pathpyG.statistics.degrees import degree_sequence

from pathpyG.core.index_map import IndexMap
from pathpyG.algorithms.generative_models import (
    erdos_renyi_gnm,
    erdos_renyi_gnp,
    is_graphic_erdos_gallai,
    max_edges,
    erdos_renyi_gnp_randomize,
    erdos_renyi_gnm_randomize,
    stochastic_block_model,
    erdos_renyi_gnp_mle,
    molloy_reed,
    watts_strogatz,
    generate_degree_sequence,
)


def test_erdos_renyi_gnm():

    # test undirected graph w/o multi-edges, w/o self-loops and without IndexMapping
    n = 100
    m = 200
    m_1 = erdos_renyi_gnm(n=n, m=m, directed=False, self_loops=False)
    assert m_1.n == n
    assert m_1.m == m
    # no multiple edges
    assert len(set([(v, w) for v, w in m_1.edges])) == len([(v, w) for v, w in m_1.edges])
    assert m_1.is_directed() is False

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = erdos_renyi_gnm(n=n, m=m, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.n == n
    assert m_2.m == m
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() is False

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = erdos_renyi_gnm(n=n, m=m, directed=True)
    assert m_3.n == n
    assert m_3.m == m
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() is True

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = erdos_renyi_gnm(n=n, m=m, self_loops=True)
    assert m_4.n == n
    assert m_4.m == m
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])
    assert m_4.is_directed() is False


def test_erdos_renyi_gnp():

    # test undirected graph w/o multi-edges, w/o self-loops and without IndexMapping
    n = 100
    p = 0.01
    m_1 = erdos_renyi_gnp(n=n, p=p)
    assert m_1.n == n
    # no multiple edges
    assert len(set([(v, w) for v, w in m_1.edges])) == len([(v, w) for v, w in m_1.edges])
    assert m_1.is_directed() is False
    assert m_1.m / 2 <= max_edges(n)

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = erdos_renyi_gnp(n=n, p=p, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.n == n
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() is False
    assert m_2.m / 2 <= max_edges(n)

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = erdos_renyi_gnp(n=n, p=p, directed=True)
    assert m_3.n == n
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() is True
    assert m_3.m <= max_edges(n, directed=True)

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = erdos_renyi_gnp(n=n, p=p, self_loops=True)
    assert m_4.n == n
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])
    assert m_4.is_directed() is False
    assert m_4.m <= max_edges(n, directed=False, self_loops=True)


def test_empty_graphs():
    g = erdos_renyi_gnp(n=100, p=0)
    assert g.n == 0

    g = erdos_renyi_gnm(n=100, m=0)
    assert g.n == 0

    g = molloy_reed([])
    assert g.n == 0


def test_molloy_Reed():
    g = molloy_reed([2, 4, 3, 4, 3])
    assert (degree_sequence(g) == np.array([2., 4., 3., 4., 3.])).all()


def test_erdos_renyi_gnm_randomize():
    n = 100
    m = 200

    g = erdos_renyi_gnm(n, m, directed=False, self_loops=False)
    g_r = erdos_renyi_gnm_randomize(g, self_loops=False)
    assert g_r.n == g.n
    assert g_r.is_directed() == g.is_directed()
    assert g_r.mapping == g.mapping
    assert g_r.m == g.m


def test_erdos_renyi_gnp_randomize():
    n = 100
    p = 0.01

    g = erdos_renyi_gnp(n, p)
    g_r = erdos_renyi_gnp_randomize(g)
    assert g_r.n == g.n
    assert g_r.mapping == g.mapping


def test_max_edges():
    assert max_edges(100) == 4950
    assert max_edges(100, directed=True) == 9900
    assert max_edges(100, directed=True, self_loops=True) == 10000
    assert max_edges(100, directed=True, self_loops=True, multi_edges=True) == _np.inf


def test_graphic_sequence():
    assert is_graphic_erdos_gallai([1, 0]) is False
    assert is_graphic_erdos_gallai([1, 3]) is False
    assert is_graphic_erdos_gallai([1, 1]) is True
    assert is_graphic_erdos_gallai([1, 3, 1, 1]) is True
    assert is_graphic_erdos_gallai([1, 3, 0, 2]) is False
    assert is_graphic_erdos_gallai([3, 2, 2, 1]) is True


def test_generate_degree_sequence():
    degree_prob_dict = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    degree_sequence = generate_degree_sequence(100, degree_prob_dict)
    assert len(degree_sequence) == 100

    xk = _np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)

    custm = scipy.stats.rv_discrete(name="custm", values=(xk, pk))
    degree_sequence = generate_degree_sequence(100, custm)
    assert len(degree_sequence) == 100

    binom_dist = scipy.stats.binom(n=10, p=0.5)
    degree_sequence = generate_degree_sequence(100, binom_dist)
    assert len(degree_sequence) == 100

    norm_dist = scipy.stats.norm(loc=3, scale=1)
    degree_sequence = generate_degree_sequence(100, norm_dist)
    assert len(degree_sequence) == 100


def test_watts_strogatz_simple():
    g = watts_strogatz(5, 1, 0.0)
    assert g.m == 5
    assert (
        to_numpy(g.data.edge_index) == np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 4, 0, 2, 1, 3, 2, 4, 0, 3]])
    ).all()

    torch.manual_seed(1)
    g = watts_strogatz(5, 1, 0.5, allow_duplicate_edges=False, allow_self_loops=False)
    print(g.data.edge_index)
    assert g.m == 5
    assert g.n == 5
    assert g.has_self_loops() is False
    assert g.is_directed() is False

    g = watts_strogatz(5, 1, 0.5, allow_duplicate_edges=False, allow_self_loops=False, undirected=False)
    assert g.m == 5
    assert g.n == 5
    assert g.has_self_loops() is False
    assert g.is_directed() is True


def test_watts_strogatz():
    g = watts_strogatz(1000, 5, 0.0)
    m = g.m

    for _ in range(3):
        g = watts_strogatz(1000, 5, 0.5, allow_duplicate_edges=False, allow_self_loops=False)
        assert g.n == 1000
        assert g.m == m
        assert torch.unique(g.data.edge_index, dim=0).size(1) == g.data.edge_index.size(1)
        assert g.has_self_loops() is False


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

    g = watts_strogatz(n, s, p, allow_duplicate_edges=True, allow_self_loops=True, undirected=False)

    sorted_edges_g = sort_edge_index(g.data.edge_index.as_tensor())
    sorted_edges = sort_edge_index(edges)

    ratio = (sorted_edges_g[1] == sorted_edges[1]).sum() / edges.size(1)
    assert ratio > 0.4
    assert ratio < 0.6


def test_watts_strogatz_get_error():
    with pytest.raises(ValueError):
        print(watts_strogatz(5, 10, 0.5, allow_duplicate_edges=False))


def test_stochastic_block_model():
    M = np.matrix('0.95 0.15; 0.15 0.85')
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g = stochastic_block_model(M, z, IndexMap(list('abcdefgh')))

    assert g.n == 8
    assert g.is_undirected()

    M = np.matrix('1 0 0; 0 1 0;0 0 1')
    z = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    g = stochastic_block_model(M, z, IndexMap(list('abcdefghi')))

    assert g.n == 9
    assert g.m == 9
    assert g.is_undirected()
    _, labels = connected_components(g)
    assert Counter(labels) == {0: 3, 1: 3, 2: 3}
