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
from torch_geometric.utils import coalesce, cumsum
from torch_geometric.data import Data

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap


class DAGData:
    """Class that can be used to store multiple observations of
    directed acyclic graphs (or - as a special case - walks)

    Example:
        ```py
        import pathpyG as pp
        import torch

        pp.config['torch']['device'] = 'cuda'

        # Generate toy example graph
        g = pp.Graph.from_edge_list([('a', 'c'),
                             ('b', 'c'),
                             ('c', 'd'),
                             ('c', 'e')])

        # Store observations of walks or dags using the index mapping
        # from the graph above
        dags = pp.DAGData(g.mapping)

        # Append one observation of a DAG
        d = torch.tensor([[0,2,2], # a -> c, c -> d, c -> e
                  [2,3,4]])
        dags.append_dag(d, weight=1.0)

        # Append observation of a walk
        dags.append_walk(('a', 'c', 'd'), weight=2.0)
        print(dags)
        ```
    """

    def __init__(self, mapping: IndexMap | None = None) -> None:
        self.dags: list = []

        if mapping:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()
        # If the function add_walks is used, all walks are saved in one Data object
        # walk_index stores a tuple that contains the idx in the dag list, the start and end index of the walk
        self.walk_index: list[tuple[int, int, int]] = []

    @property
    def num_dags(self) -> int:
        """Return the number of stored dags."""
        return len(self.dags)

    def append_walk(self, node_seq: list | tuple, weight: float = 1.0) -> None:
        """Add an observation of a walk based on a list or tuple of node IDs or indices

        Example:
                ```py
                import torch
                import pathpyG as pp

                g = pp.Graph.from_edge_list([('a', 'c'),
                        ('b', 'c'),
                        ('c', 'd'),
                        ('c', 'e')])

                walks = pp.DAGData(g.mapping)
                walks.append_walk(('a', 'c', 'd'), weight=2.0)
                paths.append_walk(('b', 'c', 'e'), weight=1.0)
                ```
        """
        idx_seq = self.mapping.to_idxs(node_seq)
        idx = torch.arange(len(node_seq))
        edge_index = torch.stack([idx[:-1], idx[1:]])

        self.walk_index.append((len(self.dags), 0, len(node_seq)))
        self.dags.append(
            Data(
                edge_index=edge_index,
                node_sequence=idx_seq.unsqueeze(1),
                num_nodes=len(node_seq),
                edge_weight=torch.full((edge_index.size(1),), weight),
            )
        )

    def append_walks(self, node_seqs: list | tuple, weights: list | tuple) -> None:
        """Add multiple observations of walks based on lists or tuples of node IDs or indices"""
        idx_seqs = torch.cat([self.mapping.to_idxs(seq) for seq in node_seqs]).unsqueeze(1)
        path_lengths = torch.tensor([len(seq) for seq in node_seqs])
        big_idx = torch.arange(path_lengths.sum())
        big_edge_index = torch.stack([big_idx[:-1], big_idx[1:]])
        # remove the edges that connect different walks
        mask = torch.ones(big_edge_index.size(1), dtype=torch.bool)
        cum_sum = cumsum(path_lengths, 0)
        mask[cum_sum[1:-1] - 1] = False
        big_edge_index = big_edge_index[:, mask]

        self.walk_index += [
            (len(self.dags), start.item(), end.item()) for start, end in torch.vstack([cum_sum[:-1], cum_sum[1:]]).T
        ]
        self.dags.append(
            Data(
                edge_index=big_edge_index,
                node_sequence=idx_seqs,
                num_nodes=idx_seqs.max().item() + 1,
                edge_weight=torch.cat([torch.full((length,), w) for length, w in zip(path_lengths, weights)]),
            )
        )

    def get_walk(self, i: int) -> tuple:
        i_dag, start, end = self.walk_index[i]
        return tuple(self.mapping.to_ids(self.dags[i_dag].node_sequence[start:end].squeeze()))

    def append_dag(self, edge_index: torch.Tensor, weight: float = 1.0) -> None:
        """Add an observation of a DAG based on an edge index

        Example:
            ```py
            import torch
            import pathpyG as pp

            dags = pp.DAGData()

        """
        edge_index = coalesce(edge_index.long())
        num_nodes = edge_index.max().item() + 1
        node_idx = torch.arange(num_nodes)
        self.dags.append(
            Data(
                edge_index=edge_index,
                node_sequence=node_idx.unsqueeze(1),
                num_nodes=num_nodes,
                edge_weight=torch.full((edge_index.size(1),), weight),
                weight=torch.tensor(weight),
            )
        )

    def map_node_seq(self, node_seq: list | tuple) -> list:
        """Map a sequence of node indices (e.g. representing a higher-order node) to node IDs"""
        return self.mapping.to_ids(node_seq)

    def __str__(self) -> str:
        """Return a string representation of the DAGData object."""
        num_dags = len(self.dags)
        weight = sum([d.edge_weight.max().item() for d in self.dags])
        s = f"DAGData with {num_dags} dags with total weight {weight}"
        return s

    @staticmethod
    def from_ngram(file: str, sep: str = ",", weight: bool = True) -> DAGData:
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

        dags = DAGData()
        dags.mapping = mapping
        dags.append_walks(node_seqs=paths, weights=weights)

        return dags
