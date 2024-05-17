"""Algorithms to calculate connected components"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

import numpy as _np
from scipy.sparse.csgraph import connected_components as _cc

from pathpyG.core.Graph import Graph

def connected_components(graph: Graph, connection='weak') -> (int, _np.ndarray):

    m = graph.get_sparse_adj_matrix()
    n, labels = _cc(m, directed=graph.is_directed(), connection=connection, return_labels=True)
    return n, labels
