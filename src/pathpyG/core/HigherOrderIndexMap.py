from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Union, Tuple

import torch


class HigherOrderIndexMap:
    """Maps node indices to string ids"""

    def __init__(self, ho_nodes: torch.Tensor, fo_node_ids: Union[List[str], None] = None) -> None:
        """Initialize mapping from indices to node IDs."""
        if fo_node_ids is not None:
            assert len(fo_node_ids) == len(set(fo_node_ids)), "node_id entries must be unique"
            self.fo_node_ids = fo_node_ids
            # higher-order: nodes = tensors of integer indices
            self.idx_to_id = {i: tuple([fo_node_ids[v] for v in j.tolist()]) for i, j in enumerate(ho_nodes)}
            self.id_to_idx = {j: i for i, j in self.idx_to_id.items()}
            self.has_ids = True
        else:
            self.fo_node_ids = []
            self.idx_to_id = {}
            self.id_to_idx = {}
            self.has_ids = False

    def to_id(self, idx: int) -> Union[Tuple[int], Tuple[str]]:
        """Map index to ID if mapping is defined, return index otherwise."""
        if self.has_ids:
            return self.idx_to_id[idx]
        return idx

    def to_idx(self, node: Union[Tuple[str], int]) -> int:
        """Map argument (ID or index) to index if mapping is defined, return argument otherwise."""
        if self.has_ids:
            return self.id_to_idx[node]
        return node

    def __str__(self) -> str:
        s = ""
        for v in self.id_to_idx:
            s += str(v) + " -> " + str(self.to_idx(v)) + "\n"
        return s
