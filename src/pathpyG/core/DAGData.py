
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


    def edge_index_k(self, k) -> Tensor:
        """ Computes edge index of k-th order graph model of all paths """
        if k == 1:
            return cat(list(self['dag_index'].values()), dim=1)
        else:
            return cat(list(DAGStorage.edge_index_kth_order(x, k) for x in self['dag_index'].values()), dim=1)


    def edge_index_k_weighted(self, k=1, path_freq=None) -> Tuple[Tensor, Tensor]:
        """ Computes edge index and edge weights of k-th order graph model of all paths """
        freq = []

        if k == 1:
            i = cat(list(self['path_index'].values()), dim=1)
            if path_freq:
                freq = cat(list(Tensor([self[path_freq][idx].item()]*(self['dag_index'][idx].size()[1]-k+1)).to(config['torch']['device']) for idx in range(self.num_paths)), dim=0)
        else:
            i = cat(list(DAGStorage.edge_index_kth_order(x, k) for x in self['dag_index'].values()), dim=1)
            if path_freq:
                # each path with length l leads to l-k+1 subpaths of length k
                freq = cat(list(Tensor([self[path_freq][idx].item()]*(self['dag_index'][idx].size()[1]-k+1)).to(config['torch']['device']) for idx in range(self.num_paths)), dim=0)

        if path_freq: # sum up frequencies of edges for all (possibly multiple) occurrences in paths

            # make edge index unique and keep reverse index, that maps each element in i to the corresponding element in edge_index
            edge_index, reverse_index = i.unique(dim=1, return_inverse=True)

            # for each edge in edge_index, the elements of x contain all indices in i that correspond to that edge
            x = list((reverse_index == idx).nonzero() for idx in range(edge_index.size()[1]))

            # for each edge, sum the weights of all
            edge_weights = Tensor([sum(freq[x[idx]]) for idx in range(edge_index.size()[1])])
        else:
            edge_index, edge_counts = i.unique(dim=1, return_counts=True)
            edge_weights = edge_counts

        return edge_index, edge_weights

    @property
    def edge_index(self) -> Tensor:
        """ Returns edge index of a first-order graph """
        return self.edge_index_k(k=1)

    @property
    def edge_index_weighted(self) -> Tuple[Tensor, Tensor]:
        """ Returns edge index and edge weights of a first-order graph """
        return self.edge_index_k_weighted(k=1)

    @staticmethod
    def from_dag(dag: Graph) -> DAGStorage:
        ds = DAGStorage()
        dags = extract_causal_trees(dag)
        for d in dags:
            src = [ dag['node_idx', dag.node_index_to_id[s.item()]] for s in dags[d][0]]
            dst = [ dag['node_idx', dag.node_index_to_id[t.item()]] for t in dags[d][1]]
            ds.add_dag(IntTensor([src, dst]))
        return ds

    @staticmethod
    def lift_order(edge_index):
        """ Compute edge index of k-th order graph for a specific dag given by edge_index

            The resulting k-th order edge_index naturally generalized first-order edge indices, i.e.
            for a  DAG (0,1), (1,2), (1,3) represented as
                    tensor([0, 1, 1],
                           [1, 2, 3])

            we get the following edge_index for a second-order graph:

            [ [[0,1], [0,1]],
              [[1,2], [1,3]] ]
        """
        # TODO compute second-order edge index

        src =[]
        dst= []



        return IntTensor([src, dst])


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
