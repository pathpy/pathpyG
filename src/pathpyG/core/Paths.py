
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
from torch.nested import nested_tensor
from torch_geometric.utils import to_scipy_sparse_matrix

from torch_geometric.data.data import BaseData, size_repr
from torch_geometric.data.storage import BaseStorage, NodeStorage

from collections import defaultdict

from pathpyG import config
from pathpyG.algorithms.temporal import extract_causal_trees
from pathpyG.core.Graph import Graph

class PathType(Enum):
    WALK = 0
    DAG = 1

class PathData:

    def __init__(self):
        self.paths = dict()
        self.path_types = dict()
        self.path_freq = dict()
        self.node_id = []

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def num_nodes(self) -> int:
        index = self.edge_index
        return len(index.reshape(-1).unique(dim=0))

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(dim=1)

    def add_edge(self, p: Tensor, freq = 1):
        self.add_walk(p, freq)

    def add_dag(self, p: Tensor, freq = 1):
        i = len(self.paths)
        self.paths[i] = p
        self.path_types[i] = PathType.DAG
        self.path_freq[i] = freq

    def add_walk(self, p: Tensor, freq=1):
        i = len(self.paths)
        self.paths[i] = p
        self.path_types[i] = PathType.WALK
        self.path_freq[i] = freq

    def to_scipy_sparse_matrix(self):
        """ Returns a sparse adjacency matrix of the underlying graph """
        return to_scipy_sparse_matrix(self.edge_index)

    @property
    def edge_index(self) -> Tensor:
        """ Returns edge index of a first-order graph representation of all paths """
        return self.edge_index_k_weighted(k=1)[0]

    @property
    def edge_index_weighted(self) -> Tuple[Tensor, Tensor]:
        """ Returns edge index and edge weights of a first-order graph representation of all paths """
        return self.edge_index_k_weighted(k=1)

    def edge_index_k_weighted(self, k=1) -> Tuple[Tensor, Tensor]:
        """ Computes edge index and edge weights of k-th order graph model of all paths """
        freq = []

        if k == 1:
            i = cat(list(self.paths.values()), dim=1)
            l_f = []
            for idx in self.paths:
                l_f.append(Tensor([self.path_freq[idx]]*self.paths[idx].size()[1]).to(config['torch']['device']))
            freq = cat(l_f, dim=0)
        else:
            l_p = []
            l_f = []
            for idx in self.paths:
                if self.path_types[idx] == PathType.WALK:
                    p = PathData.edge_index_kth_order_walk(self.paths[idx], k)
                    l_p.append(p)
                    l_f.append(Tensor([self.path_freq[idx]]*(self.paths[idx].size()[1]-k+1)).to(config['torch']['device']))
                else:
                    # we have to reshape tensors of the form [[0,1,2], [1,2,3]] to [[[0],[1],[2]],[[1],[2],[3]]]
                    p = PathData.edge_index_kth_order_dag(self.paths[idx].reshape(self.paths[idx].size()+(1,)), k)
                    if len(p)>0:
                        l_p.append(p)
                        l_f.append(Tensor([self.path_freq[idx]]*p.size()[1]).to(config['torch']['device']))
            i = cat(l_p, dim=1)
            freq = cat(l_f, dim=0)

        # make edge index unique and keep reverse index, that maps each element in i to the corresponding element in edge_index
        edge_index, reverse_index = i.unique(dim=1, return_inverse=True)
        # for each edge in edge_index, the elements of x contain all indices in i that correspond to that edge
        x = list((reverse_index == idx).nonzero() for idx in range(edge_index.size()[1]))

        # for each edge, sum the weights of all occurences
        edge_weights = Tensor([sum(freq[x[idx]]) for idx in range(edge_index.size()[1])])

        return edge_index, edge_weights

    # WALK METHODS

    @staticmethod
    def edge_index_kth_order_walk(edge_index, k=1):
        """ Compute edge index of k-th order graph for a specific walk given by edge_index

            The resulting k-th order edge_index naturally generalized first-order edge indices, i.e.
            for a walk (0,1,2,3,4,5) represented as
            [ [0,1,2,3,4],
              [1,2,3,4,5] ]

            we get the following edge_index for a second-order graph:

            [ [[0,1], [1,2], [2,3], [3,4]],
              [[1,2], [2,3], [3,4], [4,5]] ]

            while for k=3 we get

            [ [[0,1,2], [1,2,3], [2,3,4]],
              [[1,2,3], [2,3,4], [3,4,5]]
        """
        if k<=edge_index.size(dim=1):
            return edge_index.unfold(1,k,1)
        else:
            return IntTensor([]).to(config['torch']['device'])

    @staticmethod
    def walk_to_node_seq(walk):
        """ Turns an edge index for a walk into a node sequence """
        return cat([walk[:,0], walk[1,1:]])

    # DAG METHODS

    @staticmethod
    def edge_index_kth_order_dag(edge_index, k):
        """ Calculates the k-th order representation for the edge index of a single dag"""
        x = edge_index
        for i in range(1, k):
            x = PathData.lift_order_dag(x)
        return x

    @staticmethod
    def lift_order_dag(edge_index):
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
            srcs = edge_index[0][torch.all(edge_index[1]==v, axis=1).nonzero().flatten()] # type: ignore
            # get all successors of v, i.e. elements in edge_index[1] where edge_index[0] == v
            dsts = edge_index[1][torch.all(edge_index[0]==v, axis=1).nonzero().flatten()] # type: ignore
            for s in srcs:
                for d in dsts:
                    src.append(torch.cat((torch.gather(s, 0, torch.tensor([0])), v)))
                    dst.append(torch.cat((v, torch.gather(d, 0, torch.tensor([d.size()[0]-1])))))

        if len(src)>0:
            return torch.stack((torch.stack(src), torch.stack(dst)))
        else:
            return torch.tensor([])

    @staticmethod
    def from_temporal_dag(dag: Graph) -> PathData:
        ds = PathData()
        dags = extract_causal_trees(dag)
        for d in dags:
            src = [ dag['node_idx', dag.node_index_to_id[s.item()]] for s in dags[d][0]] # type: ignore
            dst = [ dag['node_idx', dag.node_index_to_id[t.item()]] for t in dags[d][1]] # type: ignore
            ds.add_dag(IntTensor([src, dst]).unique_consecutive(dim=1))
        return ds

    def __str__(self):
        s = 'PathData with {0} paths'.format(self.num_paths)
        return s

    @staticmethod
    def from_csv(file) -> PathData:
        p = PathData()
        name_map = defaultdict(lambda: len(name_map))
        freq = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                path = []
                fields = line.split(',')
                for v in fields[:-1]:
                    path.append(name_map[v])
                w = IntTensor([path[:-1], path[1:]]).to(config['torch']['device'])
                p.add_walk(w, int(float(fields[-1])))
        reverse_map = {k:i for i,k in name_map.items()}
        p.node_id = [reverse_map[i] for i in range(len(name_map))]
        return p