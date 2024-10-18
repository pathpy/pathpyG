import numpy as _np
import scipy

from pathpyG.core.IndexMap import IndexMap
from pathpyG.algorithms.generative_models import G_nm, G_np, is_graphic_Erdos_Gallai, max_edges, G_np_randomize, G_nm_randomize, G_np_MLE, generate_degree_sequence


def test_G_nm():

    # test undirected graph w/o multi-edges, w/o self-loops and without IndexMapping
    n = 100
    m = 200
    m_1 = G_nm(n=n, m=m)
    assert m_1.N == n
    # 400 directed edges for undirected graph
    assert m_1.M == 2 * m
    # no multiple edges
    assert len(set([(v, w) for v, w in m_1.edges])) == len([(v, w) for v, w in m_1.edges])
    assert m_1.is_directed() is False

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = G_nm(n=n, m=m, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.N == n
    # 400 directed edges for undirected graph
    assert m_2.M == 2 * m
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() is False

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = G_nm(n=n, m=m, directed=True)
    assert m_3.N == n
    # 200 directed edges
    assert m_3.M == m
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() is True

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = G_nm(n=n, m=m, self_loops=True)
    assert m_4.N == n
    # since self-loops only exist in one direction we have 2 * m - n <= M <= 2 * m
    assert m_4.M >= 2*m - n and m_4.M <= 2 * m
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])
    assert m_4.is_directed() is False


def test_G_np():

    # test undirected graph w/o multi-edges, w/o self-loops and without IndexMapping
    n = 100
    p = 0.001
    m_1 = G_np(n=n, p=p)
    assert m_1.N == n
    # no multiple edges
    assert len(set([(v, w) for v, w in m_1.edges])) == len([(v, w) for v, w in m_1.edges])
    assert m_1.is_directed() is False
    assert m_1.M/2 <= max_edges(n)

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = G_np(n=n, p=p, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.N == n
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() is False
    assert m_2.M/2 <= max_edges(n)

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = G_np(n=n, p=p, directed=True)
    assert m_3.N == n    
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() is True
    assert m_3.M <= max_edges(n, directed=True)

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = G_np(n=n, p=p, self_loops=True)
    assert m_4.N == n
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])    
    assert m_4.is_directed() is False
    assert m_4.M <= max_edges(n, directed=False, self_loops=True)


def test_G_nm_randomize():
    n = 100
    m = 200

    g = G_np(n, m)
    g_r = G_nm_randomize(g)
    assert g_r.N == g.N
    assert g_r.mapping == g.mapping
    assert g_r.M == g_r.M
 

def test_G_np_randomize():
    n = 100
    p = 0.01

    g = G_np(n, p)
    g_r = G_np_randomize(g)
    assert g_r.N == g.N
    assert g_r.mapping == g.mapping


def test_max_edges():
    assert max_edges(100) == 4950
    assert max_edges(100, directed=True) == 9900
    assert max_edges(100, directed=True, self_loops=True) == 10000
    assert max_edges(100, directed=True, self_loops=True, multi_edges=True) == _np.inf


def test_graphic_sequence():
    assert is_graphic_Erdos_Gallai([1, 0]) is False
    assert is_graphic_Erdos_Gallai([1, 3]) is False
    assert is_graphic_Erdos_Gallai([1, 1]) is True
    assert is_graphic_Erdos_Gallai([1, 3, 1, 1]) is True
    assert is_graphic_Erdos_Gallai([1, 3, 0, 2]) is False
    assert is_graphic_Erdos_Gallai([3, 2, 2, 1]) is True


def test_generate_degree_sequence():
    degree_prob_dict = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    degree_sequence = generate_degree_sequence(100, degree_prob_dict)
    assert len(degree_sequence) == 100

    xk = _np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)

    custm = scipy.stats.rv_discrete(name='custm', values=(xk, pk))
    degree_sequence = generate_degree_sequence(100, custm)
    assert len(degree_sequence) == 100

    binom_dist = scipy.stats.binom(n=10, p=0.5)
    degree_sequence = generate_degree_sequence(100, binom_dist)
    assert len(degree_sequence) == 100

    norm_dist = scipy.stats.norm(loc=3, scale=1)
    degree_sequence = generate_degree_sequence(100, norm_dist)
    assert len(degree_sequence) == 100
