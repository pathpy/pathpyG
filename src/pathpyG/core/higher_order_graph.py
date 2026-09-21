"""Higher-order De Bruijn graph representation and related operations."""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from pathpyG.algorithms.lift_order import aggregate_edge_index
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")


class HigherOrderGraph(Graph):
    """A De Bruijn graph of order `k`, whose nodes are paths of `k` first-order nodes.

    A `HigherOrderGraph` has one node per distinct path of length `k`
    in the underlying first-order graph. Repeated observations of the same path are
    aggregated into an `edge_weight`. Timestamps are not represented: this
    is a model of how paths flow rather than a record of what happened.

    The order `k` is the memory of the model: the next step of a path depends on the
    last `k` first-order nodes visited.

    - Order 1 is simply the weighted first-order graph, with plain node IDs rather than tuples.
    - Order 0 is the memoryless model, in which every step is an independent draw from a fixed
        distribution over first-order nodes. It has a single node, the empty path `()`, and one
        self-loop per first-order node, weighted by how often that node is visited. Since all
        loops share the same endpoints, the first-order node each loop emits is stored in the
        `edge_first_order_node` edge attribute. The transition probabilities of the loops are
        the node visitation probabilities.

    Info:
        In addition to the attributes of [`Graph`][pathpyG.Graph], the `data` object holds:

        - `node_sequence`: [Tensor][torch.Tensor] of shape `(num_nodes, order)`, the
            first-order node indices making up the path each higher-order node represents.
        - `edge_weight`: [Tensor][torch.Tensor] with the aggregated weight of each transition.
        - `inverse_idx`: [Tensor][torch.Tensor] mapping each row of the *pre-aggregation*
            node sequence to the index of the higher-order node it was merged into.
        - `edge_first_order_node` (order 0 only): [Tensor][torch.Tensor] with the index of
            the first-order node emitted by each self-loop.

    Attributes:
        data (Data): PyG Data object containing edges and attributes.
        mapping (IndexMap): Mapping from higher-order node IDs (tuples, for order other than 1) to indices.
        first_order_mapping (IndexMap): Mapping of the underlying first-order node IDs to indices.
        n_first_order (int): Number of first-order nodes the higher-order nodes are built from.

    Examples:
        >>> import pathpyG as pp
        >>> from pathpyG.core.higher_order_graph import HigherOrderGraph
        >>> g = pp.Graph.from_edge_list([("a", "c"), ("c", "d")])
        >>> h = HigherOrderGraph.from_graph(g)
        >>> print(h.order, h.nodes)
        1 ['a', 'c', 'd']
    """

    _internal_node_attrs: frozenset[str] = frozenset({"node_sequence"})

    def __init__(
        self,
        data: Data,
        order: Optional[int] = None,
        first_order_mapping: Optional[IndexMap] = None,
        n_first_order: Optional[int] = None,
        mapping: Optional[IndexMap] = None,
    ) -> None:
        """Create a HigherOrderGraph from a `Data` object carrying a `node_sequence`.

        Args:
            data: PyG `Data` object with an `edge_index` and a `node_sequence` of shape
                `(num_nodes, order)`. For order 1, the `node_sequence` may be omitted and
                is then taken to be the identity.
            order: Expected order `k`. If given, it is validated against the width of the
                node sequence; if omitted, the order is inferred from it.
            first_order_mapping: Mapping of the underlying first-order node IDs. Defaults
                to an empty mapping.
            n_first_order: Number of first-order nodes. Defaults to the number of IDs in
                `first_order_mapping`, or the largest index in the node sequence plus one.
                Required for order 0 if `first_order_mapping` has no IDs.
            mapping: Mapping of higher-order node IDs to indices. For order 1 this must
                use plain IDs; for any other order it must use tuple IDs.

        Raises:
            ValueError: If the order, the node sequence, and the mapping disagree, or if
                the node sequence refers to first-order nodes that do not exist.
        """
        if "node_sequence" not in data and order not in (None, 1):
            raise ValueError(f"A HigherOrderGraph of order {order} requires a `node_sequence` node attribute.")

        super().__init__(data, mapping=mapping)

        if "node_sequence" not in self.data:
            # An order-1 node is a first-order node, so the node sequence is the identity.
            self.data.node_sequence = torch.arange(self.data.num_nodes, device=self.device).unsqueeze(1)

        if order is not None and order != self.order:
            raise ValueError(f"order={order} does not match node sequence of width {self.order}")

        if first_order_mapping is not None:
            self.first_order_mapping = first_order_mapping
        elif self.order == 1:
            # For order 1 the higher-order nodes *are* the first-order nodes.
            self.first_order_mapping = self.mapping
        else:
            self.first_order_mapping = IndexMap()

        if n_first_order is not None:
            self._n_first_order = int(n_first_order)
        elif self.first_order_mapping.has_ids:
            self._n_first_order = self.first_order_mapping.num_ids()
        elif self.order == 0:
            # The empty path does not refer to any first-order node to infer the count from.
            raise ValueError("an order-0 graph requires `n_first_order` or a `first_order_mapping` with IDs")
        elif self.data.node_sequence.numel() > 0:
            self._n_first_order = int(self.data.node_sequence.max().item()) + 1
        else:
            self._n_first_order = 0

        self._validate()

    def _validate(self) -> None:
        """Check that order, node sequence, mapping and first-order node set agree."""
        if self.data.node_sequence.numel() > 0:
            max_idx = int(self.data.node_sequence.max().item())
            if max_idx >= self._n_first_order:
                raise ValueError(
                    f"node sequence refers to first-order node {max_idx}, "
                    f"but there are only {self._n_first_order} first-order nodes"
                )

        if self.order == 0:
            if self.n > 1:
                raise ValueError(f"an order-0 graph has at most one node (the empty path), got {self.n}")
            if "edge_first_order_node" not in self.data:
                raise ValueError("an order-0 graph requires an `edge_first_order_node` edge attribute")
            emitted = self.data.edge_first_order_node
            if emitted.numel() > 0 and int(emitted.max().item()) >= self._n_first_order:
                raise ValueError(
                    f"edge_first_order_node refers to first-order node {int(emitted.max().item())}, "
                    f"but there are only {self._n_first_order} first-order nodes"
                )

        if self.mapping.has_ids:
            # Higher-order nodes are paths and are identified by tuples (the empty tuple
            # for order 0); first-order nodes are entities and are identified by plain IDs.
            if self.mapping.has_tuple_ids != (self.order != 1):
                raise ValueError(
                    f"a mapping for a graph of order {self.order} must "
                    f"{'use' if self.order != 1 else 'not use'} tuple IDs"
                )
            if self.mapping.num_ids() != self.n:
                logger.warning(
                    "mapping has %s IDs but graph has %s nodes", self.mapping.num_ids(), self.n
                )

    @property
    def order(self) -> int:
        """Return the order of the graph, i.e. the number of first-order nodes in each node's path."""
        return self.data.node_sequence.size(1)

    def to(self, device: torch.device) -> HigherOrderGraph:
        """Move all tensors to the given device.

        Args:
            device: torch device to which all tensors shall be moved

        Returns:
            HigherOrderGraph: self
        """
        super().to(device)
        self.data.node_sequence = self.data.node_sequence.to(device)
        if "inverse_idx" in self.data:
            self.data.inverse_idx = self.data.inverse_idx.to(device)
        return self

    @staticmethod
    def _validate_order(order: int) -> None:
        """Reject orders for which no De Bruijn graph is defined."""
        if order < 0:
            logger.error("order must be at least 0, got %s", order)
            raise ValueError(f"order must be at least 0, got {order}")

    @staticmethod
    def _build_mapping(node_sequence: torch.Tensor, first_order_mapping: IndexMap) -> IndexMap:
        """Build the higher-order `IndexMap` naming each node by the path it represents."""
        # TODO: Is it better to have a single HigherOrderMapping class?
        order = node_sequence.size(1)
        if node_sequence.size(0) == 0:
            # An order beyond the longest observed path yields a graph without nodes,
            # and `IndexMap` cannot be built from an empty list of IDs.
            return IndexMap()
        if order == 0:
            # The only order-0 node is the empty path.
            return IndexMap([()])
        if order == 1:
            # Order-1 node indices are first-order node indices, so the mapping carries over.
            return first_order_mapping
        if first_order_mapping.has_ids:
            return IndexMap([tuple(first_order_mapping.to_ids(v.cpu())) for v in node_sequence])
        return IndexMap([tuple(v.tolist()) for v in node_sequence])

    @classmethod
    def aggregate(
        cls,
        edge_index: torch.Tensor,
        node_sequence: torch.Tensor,
        first_order_mapping: Optional[IndexMap] = None,
        edge_weight: Optional[torch.Tensor] = None,
        n_first_order: Optional[int] = None,
        aggr: str = "sum",
    ) -> HigherOrderGraph:
        """Aggregate a (possibly duplicated) higher-order edge index into a De Bruijn graph.

        Args:
            edge_index: Edge index whose nodes are indices into `node_sequence`.
            node_sequence: Tensor of shape `(num_nodes, order)` with the first-order path
                each (not yet aggregated) node represents.
            first_order_mapping: Mapping of the underlying first-order node IDs.
            edge_weight: Weight of each edge prior to aggregation. Defaults to ones.
            n_first_order: Number of first-order nodes, including isolated ones.
            aggr: Reduction used for the edge weights. One of "sum", "mean", "min", "max".

        Returns:
            HigherOrderGraph: The aggregated higher-order graph.

        Raises:
            ValueError: If `node_sequence` has width 0. Use
                [`from_node_weights`][pathpyG.HigherOrderGraph.from_node_weights] to build
                an order-0 graph.
        """
        order = node_sequence.size(1)
        if order == 0:
            raise ValueError("order-0 graphs cannot be aggregated from an edge index; use from_node_weights")
        if first_order_mapping is None:
            first_order_mapping = IndexMap()
        if n_first_order is None:
            if first_order_mapping.has_ids:
                n_first_order = first_order_mapping.num_ids()
            else:
                n_first_order = int(node_sequence.max().item()) + 1 if node_sequence.numel() > 0 else 0

        data = aggregate_edge_index(edge_index, node_sequence, edge_weight, aggr=aggr)

        if order == 1 and n_first_order > data.num_nodes:
            # Order-1 indices are first-order indices, so first-order nodes that are not
            # traversed by any path are simply isolated nodes of the order-1 graph.
            data.num_nodes = n_first_order
            data.node_sequence = torch.arange(n_first_order, device=edge_index.device).unsqueeze(1)

        return cls(
            data,
            order=order,
            first_order_mapping=first_order_mapping,
            n_first_order=n_first_order,
            mapping=cls._build_mapping(data.node_sequence, first_order_mapping),
        )

    @classmethod
    def from_node_weights(
        cls, node_weight: torch.Tensor, first_order_mapping: Optional[IndexMap] = None
    ) -> HigherOrderGraph:
        """Create the order-0 graph for the given visitation weights of first-order nodes.

        The result has a single node, the empty path `()`, with one self-loop per first-order
        node. Each loop carries the node's weight as `edge_weight` and the node's index as
        `edge_first_order_node`. Nodes with weight 0 keep their loop, so that every first-order
        node is represented.

        Args:
            node_weight: Tensor of shape `(n_first_order,)` with the weight of each first-order node.
            first_order_mapping: Mapping of the underlying first-order node IDs.

        Returns:
            HigherOrderGraph: A higher-order graph of order 0.

        Examples:
            >>> import torch
            >>> import pathpyG as pp
            >>> h = pp.HigherOrderGraph.from_node_weights(
            ...     torch.tensor([3.0, 1.0, 4.0, 4.0, 0.0]), first_order_mapping=pp.IndexMap(list("abcde"))
            ... )
            >>> print(h.order, h.nodes, h.m)
            0 [()] 5
            >>> print(h.transition_probabilities(edge_attr="edge_weight"))
            tensor([0.2500, 0.0833, 0.3333, 0.3333, 0.0000])
        """
        n_first_order = node_weight.size(0)
        if first_order_mapping is None:
            first_order_mapping = IndexMap()
        elif first_order_mapping.has_ids and first_order_mapping.num_ids() != n_first_order:
            raise ValueError(
                f"first_order_mapping has {first_order_mapping.num_ids()} IDs, "
                f"but {n_first_order} node weights were given"
            )

        device = node_weight.device
        data = Data(
            edge_index=torch.zeros((2, n_first_order), dtype=torch.long, device=device),
            num_nodes=1,
            node_sequence=torch.empty((1, 0), dtype=torch.long, device=device),
            edge_weight=node_weight,
            edge_first_order_node=torch.arange(n_first_order, device=device),
        )
        return cls(
            data,
            order=0,
            first_order_mapping=first_order_mapping,
            n_first_order=n_first_order,
            mapping=IndexMap([()]),
        )

    @classmethod
    def from_graph(cls, g: Graph, weight: str = "edge_weight") -> HigherOrderGraph:
        """Create the order-1 graph corresponding to a first-order graph.

        Multi-edges are coalesced into a single weighted edge.

        Args:
            g: First-order graph.
            weight: Name of the edge attribute to use as edge weight. If absent, each
                edge counts once.

        Returns:
            HigherOrderGraph: A higher-order graph of order 1.
        """
        edge_index = g.data.edge_index.as_tensor()
        if weight in g.data:
            edge_weight = g.data[weight]
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        node_sequence = torch.arange(g.n, device=edge_index.device).unsqueeze(1)

        return cls.aggregate(
            edge_index,
            node_sequence,
            first_order_mapping=g.mapping,
            edge_weight=edge_weight,
            n_first_order=g.n,
        )

    @classmethod
    def from_temporal_graph(
        cls,
        g: TemporalGraph,
        order: int = 1,
        delta: float | int = 1,
        weight: str = "edge_weight",
    ) -> HigherOrderGraph:
        """Create the De Bruijn graph of order `k` for time-respecting paths in a temporal graph.

        Order 1 is simply the weighted static graph and ignores `delta`; for higher orders
        the nodes are the time-respecting paths of `k` nodes, i.e. those whose consecutive
        interactions are at most `delta` apart. Orders above 2 are reached by repeatedly
        lifting the unaggregated data.

        Args:
            g: The temporal graph.
            order: The order `k` of the graph to compute.
            delta: The maximum time difference between two consecutive interactions of a path.
            weight: The edge attribute of `g` to use as edge weight.

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            there is no time-respecting path of that length.

        Note:
            Each call rebuilds the whole chain of lifts from order 1. To obtain several
            orders, build a [`MultiOrderModel`][pathpyG.MultiOrderModel] with
            `cached=True` once and read its `layers` instead.

        Examples:
            >>> import pathpyG as pp
            >>> t = pp.TemporalGraph.from_edge_list([("a", "c", 1), ("c", "d", 2)])
            >>> print(pp.HigherOrderGraph.from_temporal_graph(t, order=2, delta=1).nodes)
            [('a', 'c'), ('c', 'd')]
        """
        cls._validate_order(order)
        from pathpyG.core.multi_order_model import MultiOrderModel

        return MultiOrderModel.from_temporal_graph(
            g, delta=delta, max_order=order, weight=weight, cached=False
        ).layers[order]

    @classmethod
    def from_path_data(cls, path_data: PathData, order: int = 1, mode: str = "propagation") -> HigherOrderGraph:
        """Create the De Bruijn graph of order `k` modelling paths in [`PathData`][pathpyG.PathData].

        Args:
            path_data: The observed paths.
            order: The order `k` of the graph to compute. Order 0 yields the memoryless
                model of node visitation frequencies.
            mode: The process that we assume. Either "diffusion" or "propagation".

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            no observed path is that long.

        Note:
            Each call rebuilds the whole chain of lifts from order 1. To obtain several
            orders, build a [`MultiOrderModel`][pathpyG.MultiOrderModel] with
            `cached=True` once and read its `layers` instead.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("acd")))
            >>> paths.append_walk(("a", "c", "d"), weight=2)
            >>> print(pp.HigherOrderGraph.from_path_data(paths, order=2).nodes)
            [('a', 'c'), ('c', 'd')]
        """
        cls._validate_order(order)
        from pathpyG.core.multi_order_model import MultiOrderModel

        return MultiOrderModel.from_path_data(path_data, max_order=order, mode=mode, cached=False).layers[order]

    @classmethod
    def from_event_graph(cls, eg: EventGraph, order: int = 2) -> HigherOrderGraph:
        """Aggregate an [`EventGraph`][pathpyG.core.event_graph.EventGraph] into an order-`k` graph.

        Equivalent to [`from_temporal_graph`][pathpyG.HigherOrderGraph.from_temporal_graph]
        on the underlying temporal graph with the event graph's `delta`.

        Args:
            eg: The second-order temporal event graph to aggregate.
            order: The order `k` of the graph to compute.

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            there is no time-respecting path of that length.

        Note:
            Each call rebuilds the whole chain of lifts from order 1. To obtain several
            orders, build a [`MultiOrderModel`][pathpyG.MultiOrderModel] with
            `cached=True` once and read its `layers` instead.
        """
        cls._validate_order(order)
        from pathpyG.core.multi_order_model import MultiOrderModel

        return MultiOrderModel.from_event_graph(eg, max_order=order, cached=False).layers[order]

    def to_first_order(self, mode: str = "last") -> Graph:
        """Project the higher-order graph back onto the first-order nodes.

        Each higher-order node is replaced by one of the first-order nodes of its path,
        and the weights of higher-order edges mapping to the same first-order edge are
        summed. First-order nodes not traversed by any path remain as isolated nodes.
        
        Warning: This is a projection, not an inverse transformation
            This method does not reconstruct the original first-order graph from
            which this higher-order graph was built. Instead, it maps each higher-
            order node to either the first or last first-order node in its represented path.
            
            Consequently, the result preserves flow encoded by the higher-order model
            under the selected projection, but may differ from the original graph in its
            edge set. In particular, isolated first-order edges cannot be recovered.

        Args:
            mode: Which first-order node of the path represents it. Either "last" or "first".

        Returns:
            Graph: A weighted first-order graph.

        Raises:
            ValueError: If the graph has order 0, whose only node refers to no first-order node.
        """
        if self.order == 0:
            raise ValueError("an order-0 graph cannot be projected onto first-order nodes")
        if mode == "last":
            projection = self.data.node_sequence[:, -1]
        elif mode == "first":
            projection = self.data.node_sequence[:, 0]
        else:
            raise ValueError(f"Unknown mode {mode}. Only 'last' and 'first' are accepted.")

        edge_index = projection[self.data.edge_index.as_tensor()]
        if "edge_weight" in self.data:
            edge_weight = self.data.edge_weight
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_weight = coalesce(
            edge_index, edge_attr=edge_weight, num_nodes=self.n_first_order, reduce="sum"
        )

        return Graph(
            Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=self.n_first_order),
            mapping=self.first_order_mapping,
        )

    def bipartite_edge_index(
        self,
        first_order_graph: Optional[Graph] = None,
        mapping: str = "last",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the edge index connecting higher-order nodes to first-order nodes.

        Args:
            first_order_graph: The first-order graph. Optional; accepted so that call
                sites read symmetrically, and used only for its device.
            mapping: Which first-order nodes to connect to. One of "last", "first" or "both".
            device: Device on which to create the tensor.

        Returns:
            torch.Tensor: Edge index of shape `(2, ·)`, higher-order nodes in the first row.

        Raises:
            ValueError: If the graph has order 0, whose only node refers to no first-order node.
        """
        if self.order == 0:
            raise ValueError("an order-0 graph has no first-order nodes to connect to")
        if device is None:
            device = first_order_graph.device if first_order_graph is not None else self.device

        node_sequence = self.data.node_sequence
        ho_idx = torch.arange(self.n, device=device)

        if mapping == "last":
            fo_idx = node_sequence[:, -1].to(device)
        elif mapping == "first":
            fo_idx = node_sequence[:, 0].to(device)
        elif mapping == "both":
            fo_idx = torch.cat([node_sequence[:, 0], node_sequence[:, -1]]).to(device)
            ho_idx = torch.cat([ho_idx, ho_idx])
        else:
            raise ValueError(f"Unknown mapping {mapping}. Only 'last', 'first' and 'both' are accepted.")

        return torch.stack([ho_idx, fo_idx])

    @property
    def n_first_order(self) -> int:
        """Number of first-order nodes underlying the higher-order nodes."""
        return self._n_first_order

    def node_id(self, idx: int) -> Union[str, int, tuple]:
        """Return the first-order path represented by the higher-order node `idx`."""
        seq = self.data.node_sequence[idx]
        if self.order == 0:
            return ()
        if self.order == 1:
            return self.first_order_mapping.to_id(int(seq[0].item()))
        if self.first_order_mapping.has_ids:
            return tuple(self.first_order_mapping.to_ids(seq.cpu()).tolist())
        return tuple(seq.tolist())

    def __add__(self, other: Graph, reduce: str = "sum") -> HigherOrderGraph:
        """Combine this higher-order graph with another one of the same order.

        Nodes are matched by the paths they represent, as described for
        [`Graph.__add__`][pathpyG.Graph.__add__]. The first-order node sets are joined as well.

        Args:
            other: Higher-order graph of the same order to be combined with this graph
            reduce: Reduction method for node attributes of nodes that are present in both graphs.

        Returns:
            HigherOrderGraph: The combined higher-order graph.

        Raises:
            TypeError: If `other` is not a `HigherOrderGraph`.
            ValueError: If the graphs have different orders, or only one of them has first-order node IDs.
        """
        if not isinstance(other, HigherOrderGraph):
            raise TypeError("a HigherOrderGraph can only be combined with another HigherOrderGraph")
        if other.order != self.order:
            raise ValueError(f"cannot combine graphs of order {self.order} and {other.order}")

        data, mapping = self._add_data(other, reduce)
        device = data.edge_index.device

        fo_self, fo_other = self.first_order_mapping, other.first_order_mapping
        if self.order == 1:
            # Order-1 nodes are first-order nodes, so the joint mapping is the first-order mapping.
            first_order_mapping = mapping
            n_first_order = data.num_nodes
        elif fo_self.has_ids and fo_other.has_ids:
            if np.array_equal(fo_self.node_ids, fo_other.node_ids):  # type: ignore[arg-type]
                first_order_mapping = fo_self
            else:
                first_order_mapping = IndexMap(
                    np.unique(np.concatenate([fo_self.node_ids, fo_other.node_ids])).tolist()
                )
            n_first_order = first_order_mapping.num_ids()
        elif not fo_self.has_ids and not fo_other.has_ids:
            first_order_mapping = IndexMap()
            n_first_order = max(self.n_first_order, other.n_first_order)
        else:
            raise ValueError("cannot combine graphs where only one has first-order node IDs")

        # Rebuild the node sequence from the joint mapping, whose IDs are the paths.
        if self.order == 0:
            data.node_sequence = torch.empty((data.num_nodes, 0), dtype=torch.long, device=device)
            data.edge_first_order_node = first_order_mapping.to_idxs(
                np.concatenate(
                    [
                        fo_self.to_ids(self.data.edge_first_order_node.cpu()),
                        fo_other.to_ids(other.data.edge_first_order_node.cpu()),
                    ]
                ),
                device=device,
            )
        elif self.order == 1:
            data.node_sequence = torch.arange(data.num_nodes, device=device).unsqueeze(1)
        else:
            data.node_sequence = first_order_mapping.to_idxs(
                mapping.to_ids(np.arange(data.num_nodes)), device=device
            ).reshape(data.num_nodes, self.order)

        # Each pre-aggregation row keeps pointing at the (renumbered) node it was merged into.
        if "inverse_idx" in data:
            data.inverse_idx = mapping.to_idxs(
                np.concatenate(
                    [self.mapping.to_ids(self.data.inverse_idx.cpu()), other.mapping.to_ids(other.data.inverse_idx.cpu())]
                ),
                device=device,
            )

        return HigherOrderGraph(
            data,
            order=self.order,
            first_order_mapping=first_order_mapping,
            n_first_order=n_first_order,
            mapping=mapping,
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the higher-order graph."""
        s = (
            f"Higher-order graph of order {self.order} with {self.n} nodes and {self.m} edges\n"
            f"(over {self.n_first_order} first-order nodes)\n"
        )
        return s + "\n".join(super().__str__().split("\n")[1:])
