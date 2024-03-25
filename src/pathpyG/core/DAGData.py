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
from torch_geometric.utils import degree, coalesce
from torch_geometric.data import Data

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
        self.mapping = IndexMap()

    def append_walk(self, node_seq, weight: int=1):
        """Add an observation of a walk based on a sequence of node IDs or indices
        
        Example:
                ```py
                import torch
                import pathpyG as pp

                g = pp.Graph.from_edge_list([('a', 'c'),
                        ('b', 'c'),
                        ('c', 'd'),
                        ('c', 'e')])

                paths = pp.DAGData(g.mapping)
                paths.add_walk_seq(('a', 'c', 'd'), weight=2)
                paths.add_walk_seq(('b', 'c', 'e'), weight=2)
                ```
        """
        idx_seq = [ self.mapping.to_idx(v) for v in node_seq ]
        e_i = torch.tensor([idx_seq[:-1], idx_seq[1:]]) #.to(config['torch']['device'])
        self.append(e_i, weight)

    def append(self, edge_index, weight: int=1):
        edge_index = coalesce(edge_index.long())
        num_nodes = edge_index.max()+1
        node_idx = torch.arange(num_nodes)
        self.dags.append(Data(edge_index=edge_index, node_sequences=node_idx.unsqueeze(1), num_nodes=num_nodes, weight=torch.tensor(weight))) #.to(config['torch']['device']))

    def __str__(self) -> str:
        """Return string representation of DAGData object."""
        num_dags = len(self.dags)
        s = f"DAGData with {num_dags} dags"
        return s
    
    @staticmethod
    def from_ngram(file: str, sep: str=',', weight: bool=True) -> DAGData:
        dags = DAGData()
        mapping = IndexMap()
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                path = []
                w = 1
                fields = line.split(sep)
                if weight:
                    for v in fields[:-1]:
                        mapping.add_id(v)
                        path.append(mapping.to_idx(v))
                    w = int(float(fields[-1]))
                else:
                    for v in fields:
                        mapping.add_id(v)
                        path.append(mapping.to_idx(v))
                e_i = torch.tensor([path[:-1], path[1:]]) #.to(config['torch']['device'])
                dags.append(edge_index=e_i, weight=w)
        dags.mapping = mapping
        return dags