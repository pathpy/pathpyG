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
import numpy as np
from torch_geometric.utils import cumsum
from torch_geometric.data import Data

from pathpyG.core.IndexMap import IndexMap


class PathData:
    """Class that can be used to store multiple observations of
    node sequences representing paths or walks

    Examples:
        >>> import pathpyG as pp
        >>> # Generate toy example graph
        >>> g = pp.Graph.from_edge_list([('a', 'c'),
        >>>                      ('b', 'c'),
        >>>                      ('c', 'd'),
        >>>                      ('c', 'e')])
        >>> # Store observations of walks using the index mapping
        >>> # from the graph above
        >>> paths = pp.PathData(g.mapping)
        >>> paths.append_walk(('a', 'c', 'd'), weight=2.0)
        >>> paths.append_walk(('b', 'c', 'e'), weight=2.0)
        >>> print(paths)
        PathData with 2 paths with total weight 4.0
    """

    def __init__(self, mapping: IndexMap | None = None) -> None:
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()
        self.data: Data = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            node_sequence=torch.empty((0, 1), dtype=torch.long),
            dag_weight=torch.empty(0, dtype=torch.float),
            dag_num_edges=torch.empty(0, dtype=torch.long),
            dag_num_nodes=torch.empty(0, dtype=torch.long),
        )
        self.data.num_nodes = 0

    @property
    def num_paths(self) -> int:
        """Return the number of stored paths."""
        return len(self.data.dag_num_edges)

    def _append_data(
        self,
        edge_index: torch.Tensor,
        node_sequence: torch.Tensor,
        weights: torch.Tensor,
        num_edges: torch.Tensor,
        num_nodes: torch.Tensor,
    ) -> None:
        """
        Append a edge_index and node_sequence to the PathData object and
        reassign the indices so that there is no overlap.

        Args:
            edge_index: Edge index of the new path(s)
            node_sequence: Node sequence of the new path(s)
            weights: Weights of the new path(s)
            num_edges: Number of edges in the new path(s)
            num_nodes: Number of nodes in the new path(s)
        """
        new_edge_index = edge_index + self.data.num_nodes
        self.data.edge_index = torch.cat([self.data.edge_index, new_edge_index], dim=1)
        self.data.node_sequence = torch.cat([self.data.node_sequence, node_sequence])
        self.data.dag_weight = torch.cat([self.data.dag_weight, weights])
        self.data.dag_num_edges = torch.cat([self.data.dag_num_edges, num_edges])
        self.data.dag_num_nodes = torch.cat([self.data.dag_num_nodes, num_nodes])
        self.data.num_nodes += num_nodes.sum().item()

    def append_walk(self, node_seq: list | tuple, weight: float = 1.0) -> None:
        """Add an observation of a walk based on a list or tuple of node IDs or indices

        Args:
            node_seq: List or tuple of node IDs
            weight: Weight of the walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(['a', 'b', 'c', 'd', 'e'])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walk(('a', 'c', 'd'), weight=2.0)
            >>> paths.append_walk(('b', 'c', 'e'), weight=1.0)
        """
        idx_seq = self.mapping.to_idxs(node_seq).unsqueeze(1)
        idx = torch.arange(len(node_seq), device=self.data.edge_index.device)
        edge_index = torch.stack([idx[:-1], idx[1:]])

        self._append_data(
            edge_index=edge_index,
            node_sequence=idx_seq,
            weights=torch.tensor([weight], device=self.data.edge_index.device),
            num_edges=torch.tensor([edge_index.shape[1]], device=self.data.edge_index.device),
            num_nodes=torch.tensor([len(node_seq)], device=self.data.edge_index.device),
        )

    def append_walks(self, node_seqs: list | tuple, weights: list | tuple) -> None:
        """Add multiple observations of walks based on lists or tuples of node IDs or indices

        Args:
            node_seqs: List or tuple of lists or tuples of node IDs
            weights: List or tuple of weights for each walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(['a', 'b', 'c', 'd', 'e'])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walks([['a', 'c', 'd'], ['b', 'c', 'e']], [2.0, 1.0])
        """
        idx_seqs = torch.cat([self.mapping.to_idxs(seq) for seq in node_seqs]).unsqueeze(1)
        dag_num_nodes = torch.tensor([len(seq) for seq in node_seqs], device=self.data.edge_index.device)

        big_idx = torch.arange(dag_num_nodes.sum(), device=self.data.edge_index.device)
        big_edge_index = torch.stack([big_idx[:-1], big_idx[1:]])

        # remove the edges that connect different walks
        mask = torch.ones(big_edge_index.size(1), dtype=torch.bool, device=self.data.edge_index.device)
        cum_sum = cumsum(dag_num_nodes, 0)
        mask[cum_sum[1:-1] - 1] = False
        big_edge_index = big_edge_index[:, mask]

        weights = torch.Tensor(weights, device=self.data.edge_index.device)

        self._append_data(
            edge_index=big_edge_index,
            node_sequence=idx_seqs,
            weights=weights,
            num_edges=dag_num_nodes - 1,
            num_nodes=dag_num_nodes,
        )

    def get_walk(self, i: int) -> tuple:
        """Return the i-th walk (based on when it was appended) as a tuple of node IDs

        Args:
            i: Index of the walk to retrieve

        Returns:
            Tuple of node IDs representing the i-th walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(['a', 'b', 'c', 'd', 'e'])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walk(('a', 'c', 'd'), weight=2.0)
            >>> walks.get_walk(0)
            ('a', 'c', 'd')
        """
        start = self.data.dag_num_nodes[:i].sum().item()
        end = start + self.data.dag_num_nodes[i].item()
        return tuple(self.mapping.to_ids(self.data.node_sequence[start:end].squeeze()))

    def map_node_seq(self, node_seq: list | tuple) -> list:
        """Map a sequence of node indices (e.g. representing a higher-order node) to node IDs"""
        return self.mapping.to_ids(node_seq)

    def __str__(self) -> str:
        """Return a string representation of the PathData object."""
        weight = self.data.dag_weight.sum().item()
        s = f"PathData with {self.num_paths} paths with total weight {weight}"
        return s

    @staticmethod
    def from_ngram(file: str, sep: str = ",", weight: bool = True) -> PathData:
        with open(file, "r", encoding="utf-8") as f:
            if weight:
                paths_and_weights = [line.split(sep) for line in f]
                paths = [path[:-1] for path in paths_and_weights]
                weights = [float(path[-1]) for path in paths_and_weights]
            else:
                paths = [line.split(sep) for line in f]
                weights = [1.0] * len(paths)

        mapping = IndexMap()
        mapping.add_ids(np.concatenate([np.array(path) for path in paths]))

        pathdata = PathData(mapping)
        pathdata.append_walks(node_seqs=paths, weights=weights)

        return pathdata
