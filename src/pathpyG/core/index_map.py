"""IndexMap class for mapping node indices to IDs."""

from __future__ import annotations
from typing import List, Optional, Union, Any

import torch
import numpy as np

from pathpyG.utils.convert import to_numpy


class IndexMap:
    """Maps node indices to IDs.

    This class keeps a mapping from any node ID, e.g. names (strings) or higher-order IDs (tuples),
    to an index of the corresponding node in the initial list of IDs, enabling fast lookup of node IDs
    from a `torch_geometric.data.Data` object.

    Attributes:
        node_ids: `numpy.ndarray` storing the node IDs, enabling fast lookup of multiple node IDs from indices.
        id_to_idx: `dict` mapping each node ID to its index.
        id_shape: `tuple` storing the shape of the ID. The default shape is (-1,) for first-order IDs.
            For higher-order IDs, the shape will be `(-1, k)` with order `k`.

    Examples:
        Initialize an `IndexMap` object with a list of string IDs:

        >>> index_map = IndexMap(["A", "B", "C"])
        >>> print(index_map)
        A -> 0
        B -> 1
        C -> 2

        Add additional IDs to the mapping:

        >>> index_map.add_id("D")
        >>> print(index_map.to_idx("D"))
        3

        Map indices to IDs. Use `to_id` for single indices and `to_ids` for multiple indices.
        Note that the shape of the given index list will be preserved in the output:

        >>> print(index_map.to_id(1))
        B
        >>> print(index_map.to_ids([0, 2]))
        ['A' 'C']

        Map IDs to indices. Works analogously to the reversed mapping and can, e.g., be used to
        create an `edge_index` tensor from a list of edges given by source and destination node IDs:

        >>> edge_index = index_map.to_idxs([["A", "B"], ["B", "C"], ["C", "D"]]).T

        Create a higher-order ID mapping:

        >>> index_map = IndexMap([("A", "B"), ("A", "C"), ("B", "C")])
        >>> print(index_map)
        ('A', 'B') -> 0
        ('A', 'C') -> 1
        ('B', 'C') -> 2

        The methods above work analogously for higher-order IDs:

        >>> print(index_map.to_id(1))
        ('A', 'C')
        >>> print(index_map.to_ids([[0], [2]]))
        [[('A', 'B')], [('B', 'C')]]
    """

    def __init__(self, node_ids: Union[List[str], None] = None) -> None:
        """Initialize mapping from indices to node IDs.

        The mapping will keep the ordering of the IDs as provided by `node_ids`. If the IDs are not unique,
        an error will be raised.

        Args:
            node_ids: List of node IDs to initialize mapping.

        Raises:
            ValueError: If IDs are not unique.

        Examples:
            Initialize an `IndexMap` object with a list of string IDs:

            >>> index_map = IndexMap(["A", "C", "B"])
            >>> print(index_map)
            A -> 0
            C -> 1
            B -> 2

            Handle non-unique IDs and sort IDs lexicographically:

            >>> node_ids = ["A", "C", "B", "A"]
            >>> index_map = IndexMap(np.unique(node_ids))
            >>> print(index_map)
            A -> 0
            B -> 1
            C -> 2
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

        Examples:
            Check if mapping has IDs:

            >>> index_map = IndexMap()
            >>> print(index_map.has_ids)
            False

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.has_ids)
            True
        """
        return self.node_ids is not None

    def num_ids(self) -> int:
        """Return number of IDs. If mapping is not defined, return 0.

        Returns:
            Number of IDs.

        Examples:
            Get number of IDs:

            >>> index_map = IndexMap()
            >>> print(index_map.num_ids())
            0

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.num_ids())
            3

            >>> index_map = IndexMap([("A", "B"), ("A", "C"), ("B", "C")])
            >>> print(index_map.num_ids())
            3
        """
        if self.node_ids is None:
            return 0
        else:
            return len(self.node_ids)

    def add_id(self, node_id: Any) -> None:
        """Assigns additional ID to the next consecutive index.

        Args:
            node_id: ID to assign.

        Raises:
            ValueError: If ID is already present in the mapping.

        Examples:
            Add an additional ID to the mapping:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> index_map.add_id("D")
            >>> print(index_map)
            A -> 0
            B -> 1
            C -> 2
            D -> 3
        """
        if node_id not in self.id_to_idx:
            idx = self.num_ids()
            if isinstance(node_id, (list, tuple)):
                node_id = to_numpy(node_id)
                self.id_shape = (-1, *node_id.shape)
            self.node_ids = (
                np.concatenate((self.node_ids, to_numpy([node_id])))
                if self.node_ids is not None
                else to_numpy([node_id])
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

        Examples:
            Add additional IDs to the mapping:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> index_map.add_ids(["E", "D"])
            >>> print(index_map)
            A -> 0
            B -> 1
            C -> 2
            E -> 3
            D -> 4
        """
        cur_num_ids = self.num_ids()
        if isinstance(node_ids, list) and isinstance(node_ids[0], (list, tuple)):
            self.id_shape = (-1, *to_numpy(node_ids[0]).shape)

        if not isinstance(node_ids, np.ndarray):
            node_ids = to_numpy(node_ids)

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

        Examples:
            Map index to ID:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.to_id(1))
            B

            No mapping defined:

            >>> index_map = IndexMap()
            >>> print(index_map.to_id(1))
            1
        """
        if self.has_ids:
            if self.id_shape == (-1,):
                if isinstance(self.node_ids, np.ndarray) and self.node_ids.dtype.type is np.str_:
                    return str(self.node_ids[idx])
                else:
                    return self.node_ids[idx]  # type: ignore
            else:
                return tuple(self.node_ids[idx])  # type: ignore
        else:
            return idx

    def to_ids(self, idxs: list | tuple | np.ndarray) -> np.ndarray:
        """Map list of indices to IDs if mapping is defined, return indices otherwise. The shape of the given index
        list will be preserved in the output.

        Args:
            idxs: Indices to map.

        Returns:
            IDs if mapping is defined, indices otherwise.

        Examples:
            Map list of indices to IDs:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.to_ids([0, 2]))
            ['A' 'C']

            No mapping defined:

            >>> index_map = IndexMap()
            >>> print(index_map.to_ids(torch.tensor([0, 2])))
            tensor([0 2])

            Map edge_index tensor to array of edges:

            >>> edge_index = torch.tensor([[0, 2, 2, 3], [1, 1, 3, 0]])
            >>> index_map = IndexMap(["A", "B", "C", "D"])
            >>> print(index_map.to_ids(edge_index.T))
            [['A' 'B']
             ['C' 'B']
             ['C' 'D']
             ['D' 'A']]
        """
        if self.has_ids:
            if not isinstance(idxs, np.ndarray):
                idxs = to_numpy(idxs)
            return self.node_ids[idxs]  # type: ignore
        else:
            return idxs  # type: ignore

    def to_idx(self, node: str | int | tuple[str] | tuple[int]) -> int | tuple[int]:
        """Map argument (ID or index) to index if mapping is defined, return argument otherwise.

        Args:
            node: ID or index to map.

        Returns:
            Index if mapping is defined, argument otherwise.

        Examples:
            Map ID to index:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.to_idx("B"))
            1

            No mapping defined:

            >>> index_map = IndexMap()
            >>> print(index_map.to_idx(1))
            1
        """
        n: str | int | tuple[str] | tuple[int] = node
        if self.has_ids:
            if self.id_shape != (-1,):
                n = tuple(n)
            return self.id_to_idx[n]
        else:
            return n

    def to_idxs(self, nodes: list | tuple | np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        """Map list of arguments (IDs or indices) to indices if mapping is defined, return argument otherwise. The shape
        of the given argument list will be preserved in the output.

        Args:
            nodes: IDs or indices to map.

        Returns:
            Indices if mapping is defined, arguments otherwise.

        Examples:
            Map list of IDs to indices:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map.to_idxs(["B", "A"]))
            tensor([1, 0])

            No mapping defined:

            >>> index_map = IndexMap()
            >>> print(index_map.to_idxs(torch.tensor([1, 0])))
            tensor([1, 0])

            Map list of edges to edge_index tensor:

            >>> edges = [["A", "B"], ["B", "C"], ["C", "D"]]
            >>> index_map = IndexMap(np.unique(edges))
            >>> print(index_map.to_idxs(edges).T)
            tensor([[0, 1, 2],
                    [1, 2, 3]])
        """
        if self.has_ids:
            if not isinstance(nodes, np.ndarray):
                nodes = to_numpy(nodes)

            shape = nodes.shape
            if self.id_shape == (-1,):
                return torch.tensor([self.id_to_idx[node] for node in nodes.flatten()], device=device).reshape(shape)
            else:
                return torch.tensor([self.id_to_idx[tuple(node)] for node in nodes.reshape(self.id_shape)], device=device).reshape(
                    shape[: -len(self.id_shape) + 1]
                )
        else:
            return torch.tensor(nodes, device=device)

    def __str__(self) -> str:
        """Return string representation of the mapping.

        Returns:
            String representation of the mapping.

        Examples:
            Print string representation of the mapping:

            >>> index_map = IndexMap(["A", "B", "C"])
            >>> print(index_map)
            A -> 0
            B -> 1
            C -> 2
        """
        s = ""
        for v in self.id_to_idx:
            s += str(v) + " -> " + str(self.to_idx(v)) + "\n"
        return s
