from pathpyG.core.Graph import Graph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.algorithms.generative_models import G_nm, G_np


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
    assert m_1.is_directed() == False

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = G_nm(n=n, m=m, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.N == n
    # 400 directed edges for undirected graph
    assert m_2.M == 2 * m
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() == False

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = G_nm(n=n, m=m, directed=True)
    assert m_3.N == n
    # 200 directed edges
    assert m_3.M == m
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() == True

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = G_nm(n=n, m=m, self_loops=True)
    assert m_4.N == n
    # since self-loops only exist in one direction we have 2 * m - n <= M <= 2 * m
    assert m_4.M >= 2*m - n and m_4.M <= 2 * m
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])
    assert m_4.is_directed() == False


def test_G_np():

    # test undirected graph w/o multi-edges, w/o self-loops and without IndexMapping
    n = 100
    p = 0.001
    m_1 = G_np(n=n, p=p)
    assert m_1.N == n    
    # no multiple edges
    assert len(set([(v, w) for v, w in m_1.edges])) == len([(v, w) for v, w in m_1.edges])
    assert m_1.is_directed() == False

    # test undirected graph w/o multi-edges, w/o self-loops and with custom IDs
    m_2 = G_np(n=n, p=p, mapping=IndexMap([str(i) for i in range(n)]))
    assert m_2.N == n
    # no multiple edges
    assert len(set([(v, w) for v, w in m_2.edges])) == len([(v, w) for v, w in m_2.edges])
    assert m_2.is_directed() == False

    # test directed graph w/o multi-edges, w/o self-loops
    m_3 = G_np(n=n, p=p, directed=True)
    assert m_3.N == n    
    assert len(set([(v, w) for v, w in m_3.edges])) == len([(v, w) for v, w in m_3.edges])
    assert m_3.is_directed() == True

    # test undirected graph w/o multi-edges and with self-loops
    m_4 = G_np(n=n, p=p, self_loops=True)
    assert m_4.N == n
    assert len(set([(v, w) for v, w in m_4.edges])) == len([(v, w) for v, w in m_4.edges])    
    assert m_4.is_directed() == False
