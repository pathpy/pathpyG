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
from torch.nested import nested_tensor
from torch_geometric import EdgeIndex

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.PathData import PathData

class WalkDataNested(PathData):
    """Class that can be used to store multiple observations of
    walks in a graph.

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

        # Generate data on observed walks
        walks = nested_

        paths = pp.WalkData(g.mapping, )
        walk = IntTensor([[0,2], # a -> c -> d
                [2,3]])
        paths.add(walk, freq=1)
        walk = IntTensor([[1,2], # b -> c -> e
                [2,4]])
        paths.add(walk, freq=1)
        print(paths)

        print(paths.edge_index_k_weighted(k=2))
        ```
    """

    def __init__(self, paths: list, path_freq: torch.tensor, mapping: IndexMap):
        self.paths = nested_tensor(paths, dtype=torch.long)
        self.path_freq = path_freq.reshape(-1,1)
        self.mapping = mapping
        self.index_translation: Dict = {}


    def add(self, p: Tensor, freq: int = 1) -> None:
        raise NotImplementedError()
        

    def __str__(self) -> str:
        """Return string representation of WalkData object."""
        num_walks = 0
        num_walks = self.paths.to_padded_tensor(-1).size()[0]
        total = self.path_freq.sum()
        s = f"PathData with {num_walks} walks and total weight {total}"
        return s

    @staticmethod
    def walk_to_node_seq(walk: Tensor) -> Tensor:
        """Turn `edge_index` for a walk into a sequence of traversed node indices.

        Args:
            walk: ordered `edge_index` of a given walk in a graph

        Example:
            ```py
            import pathpyG as pp
            s = pp.WalkData.walk_to_node_seq(torch.tensor([[0,2],[2,3]]))
            print(s)
            ```
        """
        return cat([walk[:, 0], walk[1, 1:]])

    @classmethod
    def edge_index_kth_order(edge_index, k):
        raise NotImplementedError()

    def edge_index_k_weighted(self, k: int = 1) -> Tuple[Tensor, Tensor]:
        """Compute edge index and edge weights of $k$-th order De Bruijn graph model.
        
        Args:
            k: order of the $k$-th order De Bruijn graph model
        """
        # apply padding and turn counts to long to avoid overflow
        padded_t = self.paths.to_padded_tensor(padding=-1)
        counts = self.path_freq.long()

        # apply unfolding 
        x = padded_t.unfold(2, k, 1)
        w = counts.unfold(1,1,1).squeeze()

        # returns a list telling where elements in the input ended up in the unique output
        ho_edge_index, inverse = x.unique(dim=0, return_inverse=True)

        # weights of the two unique higher-order edges should be N and 3*N
        # weights of k-th element in output = sum of all w at indices where inverse is k
        weights = torch.zeros(ho_edge_index.size()[0], device=config['torch']['device'], dtype=torch.long).index_add(0, inverse, w)
        
        # For padded tensors, we must remove all HO-edges (and counts) where -1 occurs as a node

        # get indices of all tensors that contain a padded value
        unpad = ho_edge_index < 0
        indices = unpad.nonzero()[:,0].unique()

        idx = torch.arange(ho_edge_index.size()[0], device=config['torch']['device'])
        if indices.size()[0]>0:  
            idx = idx[idx!=indices]

        return ho_edge_index[idx], weights[idx]
