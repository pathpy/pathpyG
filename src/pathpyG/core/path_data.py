"""Module for storing path data in a PyTorch Geometric Data object."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import cumsum

from pathpyG.core.index_map import IndexMap


class PathData:
    """Class that can be used to store multiple observations of node sequences representing paths or walks.

    Attributes:
        data (Data): PyG Data object containing paths and attributes.
        mapping (IndexMap): Mapping from node IDs to indices.

    Info:
        The `data` attribute is a PyG Data object that contains the following attributes:

        - `edge_index`: Edge index (as [Tensor][torch.Tensor]) of all stored paths concatenated to a graph with multiple components.
            Each node is mapped to a unique index so that it is possible to store multiple paths with the same nodes
            in different paths and repeating nodes in the same path without ambiguity.
        - `node_sequence`: Node sequence [tensor][torch.Tensor] of shape `(total_nodes, 1)` where each entry corresponds
            to the index of the node in the underlying graph and mapping.
        - `dag_weight`: [Tensor][torch.Tensor] of shape `(num_paths,)` containing the weight of each stored path.
        - `dag_num_edges`: [Tensor][torch.Tensor] of shape `(num_paths,)` containing the number of edges in each stored path.
        - `dag_num_nodes`: [Tensor][torch.Tensor] of shape `(num_paths,)` containing the number of nodes in each stored path.

    Examples:
        >>> import pathpyG as pp
        >>> # Generate toy example graph
        >>> g = pp.Graph.from_edge_list([("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")])
        >>> # Store observations of walks using the index mapping
        >>> # from the graph above
        >>> paths = pp.PathData(g.mapping)
        >>> paths.append_walk(("a", "c", "d"), weight=2.0)
        >>> paths.append_walk(("b", "c", "e"), weight=2.0)
        >>> print(paths)
        PathData with 2 paths with total weight 4.0
    """

    def __init__(self, mapping: IndexMap | None = None, device: torch.device | None = None) -> None:
        """Initialize an empty PathData object.

        Args:
            mapping: IndexMap object to map node IDs to indices and vice versa. If None, a new IndexMap will be created.
            device: Device to store the data on. If None, the default device will be used.
        """
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()
        self.data: Data = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            node_sequence=torch.empty((0, 1), dtype=torch.long, device=device),
            dag_weight=torch.empty(0, dtype=torch.float, device=device),
            dag_num_edges=torch.empty(0, dtype=torch.long, device=device),
            dag_num_nodes=torch.empty(0, dtype=torch.long, device=device),
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
        """Append an edge_index and a node_sequence to the PathData object and reassign the indices so that there is no overlap.

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

    def to(self, device: torch.device) -> "PathData":
        """Moves all paths to the given device."""
        self.data = self.data.to(device)
        return self

    def append_walk(self, node_seq: list | tuple, weight: float = 1.0) -> None:
        """Add an observation of a walk based on a list or tuple of node IDs or indices.

        Args:
            node_seq: List or tuple of node IDs
            weight: Weight of the walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(["a", "b", "c", "d", "e"])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walk(("a", "c", "d"), weight=2.0)
            >>> walks.append_walk(("b", "c", "e"), weight=1.0)
        """
        idx_seq = self.mapping.to_idxs(node_seq, device=self.data.edge_index.device).unsqueeze(1)
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
        """Add multiple observations of walks based on lists or tuples of node IDs or indices.

        Args:
            node_seqs: List or tuple of lists or tuples of node IDs
            weights: List or tuple of weights for each walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(["a", "b", "c", "d", "e"])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walks([["a", "c", "d"], ["b", "c", "e"]], [2.0, 1.0])
        """
        idx_seqs = torch.cat(
            [self.mapping.to_idxs(seq, device=self.data.edge_index.device) for seq in node_seqs]
        ).unsqueeze(1)
        dag_num_nodes = torch.tensor([len(seq) for seq in node_seqs], device=self.data.edge_index.device)

        big_idx = torch.arange(dag_num_nodes.sum().item(), device=self.data.edge_index.device)
        big_edge_index = torch.stack([big_idx[:-1], big_idx[1:]])

        # remove the edges that connect different walks
        mask = torch.ones(big_edge_index.size(1), dtype=torch.bool, device=self.data.edge_index.device)
        cum_sum = cumsum(dag_num_nodes, 0)
        mask[cum_sum[1:-1] - 1] = False
        big_edge_index = big_edge_index[:, mask]

        self._append_data(
            edge_index=big_edge_index,
            node_sequence=idx_seqs,
            weights=torch.tensor(weights, device=self.data.edge_index.device),
            num_edges=dag_num_nodes - 1,
            num_nodes=dag_num_nodes,
        )

    @classmethod
    def from_walks(
        cls,
        node_sequence: torch.Tensor,
        weights: torch.Tensor | None = None,
        num_nodes_per_walk: torch.Tensor | None = None,
        mapping: IndexMap | None = None,
    ) -> "PathData":
        """Build a `PathData` object from walks given as node *indices*, without a Python loop.

        This is the vectorised counterpart of [`append_walks`][pathpyG.core.path_data.PathData.append_walks]: it
        takes indices rather than node IDs and builds the whole multi-component DAG in a handful of
        tensor operations, which matters when the walks come out of a tensor computation such as
        [`extract_time_respecting_walks`][pathpyG.algorithms.extract_time_respecting_walks].

        Args:
            node_sequence: Either a tensor of shape `(num_walks, num_walk_nodes)` holding walks that
                all have the same length, or a flat tensor of concatenated walks, in which case
                `num_nodes_per_walk` is required.
            weights: Weight of each walk. Defaults to one per walk.
            num_nodes_per_walk: Number of nodes of each walk, required for a flat `node_sequence`.
            mapping: Mapping from node IDs to indices.

        Returns:
            PathData: The walks, stored as a single DAG with one component per walk.

        Examples:
            >>> import torch
            >>> import pathpyG as pp
            >>> walks = pp.PathData.from_walks(torch.tensor([[0, 2, 3], [1, 2, 4]]), mapping=pp.IndexMap(list("abcde")))
            >>> print(walks.get_walk(0), walks.get_walk(1))
            ('a', 'c', 'd') ('b', 'c', 'e')
        """
        device = node_sequence.device
        if num_nodes_per_walk is None:
            if node_sequence.dim() != 2:
                raise ValueError(
                    "`node_sequence` must be of shape (num_walks, num_walk_nodes) when "
                    "`num_nodes_per_walk` is not given"
                )
            num_nodes_per_walk = torch.full(
                (node_sequence.size(0),), node_sequence.size(1), dtype=torch.long, device=device
            )
        flat_node_sequence = node_sequence.reshape(-1)

        path_data = cls(mapping=mapping, device=device)
        num_edges_per_walk = num_nodes_per_walk - 1
        total_edges = int(num_edges_per_walk.sum().item())
        if total_edges > 0:
            # Each walk contributes the edges between consecutive nodes of its own block, so an
            # edge is located by its walk's start offset plus its position within that walk.
            walk_start = cumsum(num_nodes_per_walk)[:-1]
            source = torch.repeat_interleave(walk_start, num_edges_per_walk) + (
                torch.arange(total_edges, device=device)
                - torch.repeat_interleave(cumsum(num_edges_per_walk)[:-1], num_edges_per_walk)
            )
            edge_index = torch.stack([source, source + 1])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if weights is None:
            weights = torch.ones(num_nodes_per_walk.size(0), dtype=torch.float, device=device)

        path_data._append_data(
            edge_index=edge_index,
            node_sequence=flat_node_sequence.reshape(-1, 1),
            weights=weights.to(torch.float),
            num_edges=num_edges_per_walk,
            num_nodes=num_nodes_per_walk,
        )
        return path_data

    def get_walk(self, i: int) -> tuple:
        """Return the i-th walk (based on when it was appended) as a tuple of node IDs.

        Args:
            i: Index of the walk to retrieve

        Returns:
            Tuple of node IDs representing the i-th walk

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(["a", "b", "c", "d", "e"])
            >>> walks = pp.PathData(mapping)
            >>> walks.append_walk(("a", "c", "d"), weight=2.0)
            >>> walks.get_walk(0)
            ('a', 'c', 'd')
        """
        start = self.data.dag_num_nodes[:i].sum().item()
        end = start + self.data.dag_num_nodes[i].item()
        return tuple(self.mapping.to_ids(self.data.node_sequence[start:end].squeeze()).tolist())

    def map_node_seq(self, node_seq: list | tuple) -> list:
        """Map a sequence of node indices (e.g. representing a higher-order node) to node IDs.

        Args:
            node_seq: List or tuple of node indices

        Returns:
            List of node IDs corresponding to the input node indices

        Examples:
            >>> import pathpyG as pp
            >>> mapping = pp.IndexMap(["a", "b", "c", "d", "e"])
            >>> walks = pp.PathData(mapping)
            >>> walks.map_node_seq([0, 2, 3])
            ['a', 'c', 'd']
        """
        return self.mapping.to_ids(node_seq).tolist()

    def __str__(self) -> str:
        """Return a string representation of the PathData object."""
        weight = self.data.dag_weight.sum().item()
        s = f"PathData with {self.num_paths} paths with total weight {weight}"
        return s
