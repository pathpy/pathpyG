#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""A class representing (higher-order) graphs."""

from __future__ import annotations

import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from collections import defaultdict
from collections import Counter
import numpy as np

from Paths import Paths

class Graph:

    def __init__(self, paths: Paths, k: int=1, directed=True):
        """ Create a weighted k-th order network from a path object
            Mappings between higher-order and first-order node indices are stored in fo2ho and ho2fo
        """
        self.order = k
        index = paths.get_edge_index(k)

        # aggregate multi-edges and compute edge weights
        index, self.edge_weights = index.unique(dim=1, return_counts=True)

        if self.order>1:
            # get a tensor of unique higher-order nodes
            self.nodes = index.reshape(-1, index.size(dim=2)).unique(dim=0)

            # create mapping to first-order node indices
            self.ho2fo = {tuple(j.tolist()):i for i,j in enumerate(self.nodes)}

            # create new tensor with mapped node indices
            self.edge_index = torch.tensor( ([self.ho2fo[tuple(x.tolist())] for x in index[0,:]], [self.ho2fo[tuple(x.tolist())] for x in index[1,:]]))
        else:
            self.nodes = index.reshape(-1).unique(dim=0)
            self.ho2fo = {}
            self.edge_index = index

        self.sparse_adj_matrix = to_scipy_sparse_matrix(self.edge_index)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return self.edge_index.size(dim=1)

    def get_successors(self, node):
        return self.sparse_adj_matrix[node]

    def get_adj_matrix(self, sparse=True):
        """ Returns a (sparse) adjacency matrix of the graph """
        if sparse:
            return self.sparse_adj_matrix
        else:
            return self.sparse_adj_matrix.todense()