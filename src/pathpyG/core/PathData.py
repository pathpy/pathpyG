"""Manages data on paths in graphs."""

from __future__ import annotations
from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional
)

from abc import ABC, abstractmethod

from enum import Enum
from collections import defaultdict

import torch
from torch import Tensor, IntTensor, cat, sum
from torch_geometric.utils import to_scipy_sparse_matrix, degree

from pathpyG import Graph
from pathpyG import config
from pathpyG.core.IndexMap import IndexMap


class PathData(ABC):
    """Abstract base class for classes storing data on DAGs or walks

    Subclasses of PathData store observations of walks or directed acyclic graphs.
    They provide methods to generate edge indices of weighted higher-order De Bruijn
    graph models of paths and walks.
    """

    def __init__(self, mapping: Optional[IndexMap] = None) -> None:
        """Create an empty PathData object."""
        self.paths: Dict = {}
        self.path_freq: Dict = {}
        self.mapping: IndexMap = IndexMap()
        if mapping is not None:
            self.mapping = mapping
        # allows to map indices from time-unfolded nodes to first-order nodes
        self.index_translation: Dict = {}

    @property
    def num_paths(self) -> int:
        """Return the number of stored paths."""
        return len(self.paths)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the underlying graph."""
        index = self.edge_index
        return len(index.reshape(-1).unique(dim=0))

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the underlying graph."""
        return self.edge_index.size(dim=1)

    @property
    def edge_index(self) -> Tensor:
        """Return edge index of first-order graph model of all paths."""
        return self.edge_index_k_weighted(k=1)[0]

    @property
    def edge_index_weighted(self) -> Tuple[Tensor, Tensor]:
        """Return edge index and edge weights of a first-order graph 
        model of all walks or DAGs."""
        return self.edge_index_k_weighted(k=1)

    @staticmethod
    def map_nodes(edge_index: Tensor, index_translation: Dict) -> Tensor:
        """Efficiently map node indices in an edge_index tensor based on a mapping dictionary.

        Args:
            edge_index: the tensor for which indices shall be mapped
            index_translation: dictionary mapping incides in original tensor to new indices

        Example:
            ```py
            import pathpyG as pp
            edge_index = IntTensor([[0,1,2], [1,2,3]])

            print(edge_index)
            tensor([[0, 1, 3],
                    [1, 2, 3]])

            index_translation = {0: 1, 1: 0, 2: 3, 3: 2}
            mapped = pp.PathData.map_nodes(edge_index, index_translation)

            print(mapped)
            tensor([[1, 0, 3],
                    [0, 3, 2]])
            ```
        """
        # Inspired by `https://stackoverflow.com/questions/13572448`.
        palette, key = zip(*index_translation.items())
        key = torch.tensor(key).to(config['torch']['device'])
        palette = torch.tensor(palette).to(config['torch']['device'])

        index = torch.bucketize(edge_index.ravel(), palette)
        remapped = key[index].reshape(edge_index.shape)
        return remapped

    @abstractmethod
    def add(self, p: Tensor, freq: int = 1) -> None:
        pass
    
    @abstractmethod
    def edge_index_k_weighted(self, k: int = 1) -> Tuple[Tensor, Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def edge_index_kth_order(edge_index: Tensor, k: int = 1) -> Tensor:
        pass

