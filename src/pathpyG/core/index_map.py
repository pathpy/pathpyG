from __future__ import annotations
from typing import TYPE_CHECKING, List, Union

import torch
import numpy as np


class IndexMap:
    """Maps node indices to string ids"""

    def __init__(self, node_ids: Union[List[str], None] = None) -> None:
        """Initialize mapping from indices to node IDs.

        Args:
            node_ids: List of node IDs to initialize mapping.

        Raises:
            ValueError: If IDs are not unique.
        """
        self.node_ids: np.ndarray | None = None
        self.id_to_idx: dict = {}
        self.id_shape: tuple = (-1,)  # If the index map is higher order, this will be the shape of the ID
        if node_ids is not None:
            self.add_ids(node_ids)

    @property
    def has_ids(self) -> bool:
        """Return whether mapping has IDs.

        Returns:
            Whether mapping has IDs.
        """
        return self.node_ids is not None

    def num_ids(self) -> int:
        """Return number of IDs.

        Returns:
            Number of IDs.
        """
        if self.node_ids is None:
            return 0
        else:
            return len(self.node_ids)

    def add_id(self, node_id: any) -> None:
        """Assigns additional ID to next consecutive index.

        Args:
            node_id: ID to assign.

        Raises:
            ValueError: If ID is already present in the mapping.
        """
        if node_id not in self.id_to_idx:
            idx = self.num_ids()
            if isinstance(node_id, (list, tuple)):
                node_id = np.array(node_id)
                self.id_shape = (-1, *node_id.shape)
            self.node_ids = (
                np.concatenate((self.node_ids, np.array([node_id])))
                if self.node_ids is not None
                else np.array([node_id])
            )
            self.id_to_idx[node_id] = idx
        else:
            raise ValueError("ID already present in the mapping.")

    def add_ids(self, node_ids: list | np.ndarray) -> None:
        """Assigns additional IDs to next consecutive indices. The order of IDs is preserved.

        Args:
            node_ids: IDs to assign

        Raises:
            ValueError: If IDs are not unique or already present in the mapping.
        """
        cur_num_ids = self.num_ids()
        if isinstance(node_ids, list) and isinstance(node_ids[0], (list, tuple)):
            self.id_shape = (-1, *np.array(node_ids[0]).shape)

        if not isinstance(node_ids, np.ndarray):
            node_ids = np.array(node_ids)

        all_ids = np.concatenate((self.node_ids, node_ids)) if self.node_ids is not None else node_ids
        unique_ids = np.unique(all_ids, axis=0 if self.id_shape != (-1,) else None)

        if len(unique_ids) != len(all_ids):
            raise ValueError("IDs are not unique or already present in the mapping.")

        self.node_ids = all_ids
        self.id_to_idx.update(
            {tuple(v) if self.id_shape != (-1,) else v: i + cur_num_ids for i, v in enumerate(node_ids)}
        )

    def to_id(self, idx: int) -> Union[int, str, tuple]:
        """Map index to ID if mapping is defined, return index otherwise.

        Args:
            idx: Index to map.

        Returns:
            ID if mapping is defined, index otherwise.
        """
        if self.has_ids:
            if self.id_shape == (-1,):
                return self.node_ids[idx]
            else:
                return tuple(self.node_ids[idx])
        else:
            return idx

    def to_ids(self, idxs: list | tuple | np.ndarray) -> np.ndarray:
        """Map list of indices to IDs if mapping is defined, return indices otherwise.

        Args:
            idxs: Indices to map.

        Returns:
            IDs if mapping is defined, indices otherwise.
        """
        if self.has_ids:
            if not isinstance(idxs, np.ndarray):
                idxs = np.array(idxs)
            return self.node_ids[idxs]
        else:
            return idxs

    def to_idx(self, node: Union[str, int]) -> int:
        """Map argument (ID or index) to index if mapping is defined, return argument otherwise.

        Args:
            node: ID or index to map.

        Returns:
            Index if mapping is defined, argument otherwise.
        """
        if self.has_ids:
            if self.id_shape != (-1,):
                node = tuple(node)
            elif isinstance(node, str) and np.issubdtype(self.node_ids.dtype, int) and node.isnumeric():
                node = int(node)
            return self.id_to_idx[node]
        else:
            return node

    def to_idxs(self, nodes: list | tuple | np.ndarray) -> torch.Tensor:
        """Map list of arguments (IDs or indices) to indices if mapping is defined, return argument otherwise.

        Args:
            nodes: IDs or indices to map.

        Returns:
            Indices if mapping is defined, arguments otherwise.
        """
        if self.has_ids:
            if not isinstance(nodes, np.ndarray):
                nodes = np.array(nodes)

            if np.issubdtype(self.node_ids.dtype, int) and np.issubdtype(nodes.dtype, str) and np.char.isnumeric(nodes).all():
                nodes = nodes.astype(int)
            shape = nodes.shape

            if self.id_shape == (-1,):
                return torch.tensor([self.id_to_idx[node] for node in nodes.flatten()]).reshape(shape)
            else:
                return torch.tensor([self.id_to_idx[tuple(node)] for node in nodes.reshape(self.id_shape)]).reshape(
                    shape[: -len(self.id_shape) + 1]
                )
        else:
            return torch.tensor(nodes)

    def __str__(self) -> str:
        s = ""
        for v in self.id_to_idx:
            s += str(v) + " -> " + str(self.to_idx(v)) + "\n"
        return s
