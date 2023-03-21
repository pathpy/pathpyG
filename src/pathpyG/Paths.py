#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""A class storing path data."""


from __future__ import annotations

import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from enum import Enum

from collections import defaultdict
from collections import Counter
import io


class Paths:

    def __init__(self):
        """ Initializes unified storage of edges, walks, and DAGs.

        All objects are represented as (2,n) tensors which store
        an edge index with n topologically sorted source-target pairs.

        Examples:
        Edge (0,1):             tensor([0],
                                       [1])
        Walk (0,1,2,3):         tensor([0, 1, 2],
                                       [1, 2, 3])

        DAG (0,1), (1,2), (1,3) tensor([0, 1, 1],
                                       [1, 2, 3])
        """
        self._data = defaultdict(torch.IntTensor)


    @staticmethod
    def from_csv(file):

        p = Paths()
        i = 0
        name_map = defaultdict(lambda: len(name_map))
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                path = []
                fields = line.split(',')[:-1]
                for v in fields:
                    path.append(name_map[v])
                p.add_walk(i, path)
                i += 1
        return p


    def add_edge(self, uid: int, source: int, target: int):
        """ Adds an edge, given as source and target """
        self.add_walk(uid, [source, target])


    def add_walk(self, uid: int, walk: list):
        """ Adds a walk, given as list of node indices """
        self._data[uid] = torch.IntTensor([walk[:-1], walk[1:]])


    def get_edge_index(self, k=1):
        """ Computes edge index of k-th order graph model of all paths """
        if k == 1:
            return torch.cat(list(self._data.values()), dim=1)
        else:
            return torch.cat(list(Paths.edge_index(x, k) for x in self._data.values()), dim=1)

    @staticmethod
    def edge_index(edge_index, k=1):
        """ Compute edge index of k-th order graph for a specific SEQ given by edge_index
            The resulting k-th order edge_index naturally generalized first-order edge indices, i.e.
            for a path (0,1,2,3,4,5)
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
            return torch.IntTensor([])


    def get_scipy_sparse_matrix(self):
        """ Returns a sparse adjacency matrix of the underlying graph """
        return to_scipy_sparse_matrix(self.get_edge_index())


    @staticmethod
    def to_node_seq(edge_index):
        """ Turns an edge index for a walk into a node sequence """
        return torch.cat([edge_index[:,0], edge_index[1,1:]])


    def get_edge_count(self, uid):
        """ Returns number of edges for observation with given UID """
        return self._data[uid].size(dim=1)


    def __len__(self):
        """ Returns number of observations """
        return len(self._data)