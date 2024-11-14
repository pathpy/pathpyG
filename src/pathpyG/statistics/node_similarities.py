from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict

import numpy as _np
import scipy as _sp

from pathpyG.core.graph import Graph
from pathpyG.algorithms.shortest_paths import shortest_paths_dijkstra
from pathpyG.statistics.degrees import degree_sequence


def inverse_path_length(graph: Graph, v, w) -> float:
    dist, _ = shortest_paths_dijkstra(graph)
    d = dist[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]
    if d == 0:
        return _np.inf
    else:
        return 1 / d


def common_neighbors(graph: Graph, v, w) -> float:
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    return len(N_v.intersection(N_w))


def overlap_coefficient(graph: Graph, v, w) -> float:
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    return len(N_v.intersection(N_w)) / min(len(N_v), len(N_w))


def jaccard_similarity(graph: Graph, v, w) -> float:
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    intersection = N_v.intersection(N_w)
    if len(N_v) == 0 and len(N_w) == 0:
        return 1
    else:
        return len(intersection) / (len(N_v) + len(N_w) - len(intersection))


def adamic_adar_index(graph: Graph, v, w) -> float:
    A = 0
    N_v = set([x for x in graph.successors(v)])
    N_w = set([x for x in graph.successors(w)])
    for u in N_v.intersection(N_w):
        A += 1 / _np.log(graph.out_degrees[u])
    return A


def cosine_similarity(graph: Graph, v, w) -> float:
    if graph.degrees()[v] == 0 or graph.degrees()[w] == 0:
        return 0
    else:
        A = graph.sparse_adj_matrix().todense()
        v_v = A[graph.mapping.to_idx(v)].A1
        v_w = A[graph.mapping.to_idx(w)].A1
        return _np.dot(v_v, v_w) / (_np.linalg.norm(v_v) * _np.linalg.norm(v_w))


def katz_index(graph: Graph, v, w, beta) -> float:
    A = graph.sparse_adj_matrix()
    I = _sp.sparse.identity(graph.n)
    S = _sp.sparse.linalg.inv(I - beta * A) - I
    return S[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]


def LeichtHolmeNewman_index(graph: Graph, v, w, alpha) -> float:
    A = graph.sparse_adj_matrix()
    ev = _sp.sparse.linalg.eigs(A, which="LM", k=2, return_eigenvectors=False)
    if graph.is_directed():
        m = graph.m
    else:
        m = graph.m / 2
    eigenvalues_sorted = _np.sort(_np.absolute(ev))
    lambda_1 = eigenvalues_sorted[1]
    D = _sp.sparse.diags(degree_sequence(graph))
    I = _sp.sparse.identity(graph.n)
    S = (
        2
        * m
        * lambda_1
        * _sp.sparse.linalg.inv(D)
        * _sp.sparse.linalg.inv(I - alpha * A / lambda_1)
        * _sp.sparse.linalg.inv(D)
    )
    return S[graph.mapping.to_idx(v), graph.mapping.to_idx(w)]
