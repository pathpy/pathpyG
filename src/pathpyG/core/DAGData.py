from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
    Union,
    Any,
    Optional,
    Generator,
)

import torch
from torch import IntTensor, Tensor, cat
from torch_geometric import EdgeIndex
from torch_geometric.utils import degree

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.Graph import Graph
from pathpyG.algorithms.temporal import extract_causal_trees


class DAGData:
    """Class that can be used to store multiple observations of
    directed acyclic graphs.

    Example:
        ```py
        import pathpyG as pp
        from torch import IntTensor

        pp.config['torch']['device'] = 'cuda'

        # Generate toy example graph
        g = pp.Graph.from_edge_list([('a', 'c'),
                             ('b', 'c'),
                             ('c', 'd'),
                             ('c', 'e')])

        # Generate data on observed directed acyclic graphs
        paths = pp.DAGData(g.mapping)
        dag = IntTensor([[0,2,2], # a -> c, c -> d, c -> e
                  [2,3,4]])
        paths.add(dag, freq=1)
        dag = IntTensor([[1,2,2], # b -> c, c -> d, c -> e
                  [2,3,4]])
        paths.add(dag, freq=1)
        print(paths)

        print(paths.edge_index_k_weighted(k=2))
        ```
    """

    def __init__(self) -> None:
        self.dags = []
        self.weights = []

    def append(self, edge_index: torch.Tensor, weight: int = 1):
        self.dags.append(edge_index)
        self.weights.append(weight)

    def __str__(self) -> str:
        """Return string representation of DAGData object."""
        num_dags = len(self.dags)
        total = sum(self.weights)
        s = f"DAGData with {num_dags} dags and total weight {total}"
        return s