from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Union


class IndexMap:
    """Maps node indices to string ids"""
    def __init__(self, node_ids: Union[List[str], None] = None) -> None:
        """Initialize mapping from indices to node IDs."""
        if node_ids is not None:
            assert len(node_ids) == len(set(node_ids)), "node_id entries must be unique"
            self.node_ids = node_ids
            # first-order: nodes = integer indices 
            self.idx_to_id: Dict[int, str] = dict(enumerate(node_ids))
            self.id_to_idx: Dict[str, int] = {v: i for i, v in enumerate(node_ids)}
            # higher-order: nodes = tensors of integer indices
            # _nodes = list of unique higher-order nodes as tensors
            # self.idx_to_id = { i: tuple([node_ids[v] for v in j.tolist()]) for i, j in enumerate(self._nodes)}
            # self.id_to_idx = { j: i for i, j in self.node_index_to_id.items()}
            self.has_ids = True
        else:
            self.node_ids = []
            self.idx_to_id = {}
            self.id_to_idx = {}
            self.has_ids = False

    def num_ids(self):
        return len(self.node_ids)

    def add_id(self, id):
        """Assigns additional ID to next consecutive index."""
        if id not in self.id_to_idx:
            idx = self.num_ids()
            self.node_ids.append(id)
            self.idx_to_id[idx] = id
            self.id_to_idx[id] = idx
            self.has_ids = True

    def to_id(self, idx: int) -> Union[int, str]:
        """Map index to ID if mapping is defined, return index otherwise."""
        if self.has_ids:
            return self.idx_to_id[idx]
        else:
            return idx

    def to_idx(self, node: Union[str, int]) -> int:
        """Map argument (ID or index) to index if mapping is defined, return argument otherwise."""
        if self.has_ids:
            return self.id_to_idx[node]
        else:
            return node

    def __str__(self) -> str:
        s = ''
        for v in self.id_to_idx:
            s += str(v) + ' -> ' + str(self.to_idx(v)) + '\n'
        return s