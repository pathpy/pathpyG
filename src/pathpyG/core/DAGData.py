
from __future__ import annotations
import copy
from typing import (
    Any,
    Dict,
    Set,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from enum import Enum

import torch
from torch import Tensor, IntTensor, cat, sum
from torch_geometric.utils import to_scipy_sparse_matrix

from torch_geometric.data.data import BaseData, size_repr
from torch_geometric.data.storage import BaseStorage, NodeStorage

from collections import defaultdict, Counter

from pathpyG import config
from pathpyG.core.Graph import Graph
from pathpyG.algorithms.temporal import extract_causal_trees

class DAGStorage(BaseStorage):
    def __init__(
            self,
    ):
        """ Initializes unified storage of walks, edges, and DAGs

        Paths of length n are represented as (2,n) tensors, which store
        an edge index with n topologically sorted source-target pairs.

        Examples:

        Single Edge (0,1):             tensor([0],
                                       [1])

        Walk (0,1,2,3):         tensor([0, 1, 2],
                                       [1, 2, 3])

        DAG (0,1), (1,2), (1,3) tensor([0, 1, 1],
                                       [1, 2, 3])
        """
        self['dag_index'] = {}
        self._num_paths = 0

    @property
    def num_paths(self) -> int:
        return self._num_paths

    @property
    def num_nodes(self) -> int:
        index = self.edge_index_k(k=1)
        return len(index.reshape(-1).unique(dim=0))

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(dim=1)

    def add_edge(self, p: Tensor):
        self.add_dag(p)

    def add_dag(self,p: Tensor):
        self['dag_index'][self._num_paths] = p
        self._num_paths += 1


    def edge_index_k_weighted(self, k, path_freq=None) -> Tuple[Tensor, Tensor]:
        """ Computes edge index of k-th order graph model of all paths """
        if k == 1:
            i = cat(list(self['dag_index'].values()), dim=1)
        else:
            i = cat(list(DAGStorage.edge_index_kth_order(x, k) for x in self['dag_index'].values()), dim=1)

        edge_index, edge_weights = i.unique(dim=1, return_counts=True)

        return edge_index.int(), edge_weights

    @staticmethod
    def edge_index_kth_order(edge_index, k):
        x = edge_index
        for i in range(1, k):
            x = DAGStorage.lift_order(x)
        return x

    @property
    def edge_index(self) -> Tensor:
        """ Returns edge index of a first-order graph """
        return self.edge_index_k(k=1)


    @staticmethod
    def from_dag(dag: Graph) -> DAGStorage:
        ds = DAGStorage()
        dags = extract_causal_trees(dag)
        for d in dags:
            src = [ [dag['node_idx', dag.node_index_to_id[s.item()]]] for s in dags[d][0]]
            dst = [ [dag['node_idx', dag.node_index_to_id[t.item()]]] for t in dags[d][1]]
            ds.add_dag(IntTensor([src, dst]))
        return ds

    @staticmethod
    def lift_order(edge_index):
        """Fast conversion of edge index of k-th order model to k+1-th order model"""

        a = edge_index[0].unique(dim=0)
        b = edge_index[1].unique(dim=0)
        # intersection of a and b corresponds to all nodes which have at least one incoming and one outgoing edge
        combined = torch.cat((a, b))
        uniques, counts = combined.unique(dim=0, return_counts=True)
        center_nodes = uniques[counts > 1]

        src = []
        dst = []

        # create edges of order k+1
        for v in center_nodes:
            # get all predecessors of v, i.e. elements in edge_index[0] where edge_index[1] == v
            srcs = edge_index[0][torch.all(edge_index[1]==v, axis=1).nonzero().flatten()]
            # get all successors of v, i.e. elements in edge_index[1] where edge_index[0] == v
            dsts = edge_index[1][torch.all(edge_index[0]==v, axis=1).nonzero().flatten()]
            for s in srcs:
                for d in dsts:
                    # print(torch.gather(s, 0, torch.tensor([0])))
                    # print(torch.gather(d, 0, torch.tensor([d.size()[0]-1])))
                    src.append(torch.cat((torch.gather(s, 0, torch.tensor([0])), v)))
                    dst.append(torch.cat((v, torch.gather(d, 0, torch.tensor([d.size()[0]-1])))))
                    #dst.append(torch.cat((torch.tensor([v]),torch.tensor([d]))))

        if len(src)>0:
            return torch.stack((torch.stack(src), torch.stack(dst)))
        else:
            return torch.tensor([])



    def to_scipy_sparse_matrix(self):
        """ Returns a sparse adjacency matrix of the underlying graph """
        return to_scipy_sparse_matrix(self.edge_index)


    def __str__(self):
        s = 'DAGStorage with {0} DAGs'.format(self.num_paths)
        return s


class AttrType(Enum):
    NODE = 'NODE'
    EDGE = 'EDGE'
    OTHER = 'OTHER'
    DAG = 'DAG'


class DAGData(DAGStorage, NodeStorage):

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int], Optional[int]], Optional[int]]:
        size = (self.num_nodes, self.num_nodes, self.num_paths)
        return size if dim is None else size[dim]

    def is_edge_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.NODE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False
        if key in self._cached_attr[AttrType.DAG]:
            return False

        value = self[key]

        if isinstance(value, (list, tuple)) and len(value) == self.num_edges:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if not isinstance(value, (Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if 'node' in key:
            self._cached_attr[AttrType.NODE].add(key)
            return False
        if 'dag' in key:
            self._cached_attr[AttrType.DAG].add(key)
            return False
        if 'edge' in key:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        return False

    def is_path_attr(self, key: str) -> bool:
        if '_cached_attr' not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return False
        if key in self._cached_attr[AttrType.EDGE]:
            return False
        if key in self._cached_attr[AttrType.OTHER]:
            return False
        if key in self._cached_attr[AttrType.DAG]:
            return True

        value = self[key]

        if isinstance(value, (list, tuple)) and len(value) == self.num_nodes:
            self._cached_attr[AttrType.NODE].add(key)
            return False

        if not isinstance(value, (Tensor, np.ndarray)):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        if value.ndim == 0:
            self._cached_attr[AttrType.DAG].add(key)
            return True

        if 'node' in key:
            self._cached_attr[AttrType.NODE].add(key)
            return False
        if 'edge' in key:
            self._cached_attr[AttrType.EDGE].add(key)
            return False
        if 'dag' in key:
            self._cached_attr[AttrType.DAG].add(key)
            return True

        return False
