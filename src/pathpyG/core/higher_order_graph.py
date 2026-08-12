"""Higher-order De Bruijn graph representation and related operations."""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from pathpyG.algorithms.lift_order import aggregate_edge_index, lift_order_edge_index_weighted
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")


class HigherOrderGraph(Graph):
    """A De Bruijn graph of order `k`, whose nodes are paths of `k` first-order nodes.

    Where a [`Graph`][pathpyG.Graph] has one node per entity and an
    [`EventGraph`][pathpyG.core.event_graph.EventGraph] has one node per observed
    interaction, a `HigherOrderGraph` has one node per *distinct* path of length `k`
    in the underlying first-order graph. Repeated observations of the same path are
    aggregated into an `edge_weight`, so timestamps are no longer represented: this
    is a model of how paths flow rather than a record of what happened.

    Order 1 is the degenerate case and is simply the weighted first-order graph, with
    plain node IDs rather than tuples.

    Info:
        In addition to the attributes of [`Graph`][pathpyG.Graph], the `data` object holds:

        - `node_sequence`: [Tensor][torch.Tensor] of shape `(num_nodes, order)`, the
            first-order node indices making up the path each higher-order node represents.
        - `edge_weight`: [Tensor][torch.Tensor] with the aggregated weight of each transition.
        - `inverse_idx`: [Tensor][torch.Tensor] mapping each row of the *pre-aggregation*
            node sequence to the index of the higher-order node it was merged into.

    Attributes:
        data (Data): PyG Data object containing edges and attributes.
        mapping (IndexMap): Mapping from higher-order node IDs (tuples, for order > 1) to indices.
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
            mapping: Mapping of higher-order node IDs to indices. For order > 1 this must
                use tuple IDs; for order 1 it must not.

        Raises:
            ValueError: If the order, the node sequence, and the mapping disagree, or if
                the node sequence refers to first-order nodes that do not exist.
        """
        if "node_sequence" not in data and order not in (None, 1):
            raise ValueError(f"A HigherOrderGraph of order {order} requires a `node_sequence` node attribute.")

        if isinstance(data.edge_index, EdgeIndex):
            # `Graph.__init__` re-sorts the edge index and reindexes every edge attribute by
            # the returned permutation - but `EdgeIndex.sort_by` returns `None` for an index
            # already known to be sorted, and `attr[None]` would add a dimension. Higher-order
            # graphs are routinely built from already-aggregated (hence sorted) data, so hand
            # the base class a plain tensor and let it derive a real permutation.
            data.edge_index = data.edge_index.as_tensor()

        super().__init__(data, mapping=mapping)

        # `Graph` creates an identity node sequence if none is given, so `self.order`
        # (inherited: the width of the node sequence) is now well-defined.
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

        if self.mapping.has_ids:
            # Higher-order nodes are paths and are identified by tuples; first-order
            # nodes are entities and are identified by plain IDs.
            if self.mapping.has_tuple_ids != (self.order > 1):
                raise ValueError(
                    f"a mapping for a graph of order {self.order} must "
                    f"{'use' if self.order > 1 else 'not use'} tuple IDs"
                )
            if self.mapping.num_ids() != self.n:
                logger.warning(
                    "mapping has %s IDs but graph has %s nodes", self.mapping.num_ids(), self.n
                )

    @staticmethod
    def _validate_order(order: int) -> None:
        """Reject orders for which no De Bruijn graph is defined."""
        if order < 1:
            logger.error("order must be at least 1, got %s", order)
            raise ValueError(f"order must be at least 1, got {order}")

    @staticmethod
    def _build_mapping(node_sequence: torch.Tensor, first_order_mapping: IndexMap) -> IndexMap:
        """Build the higher-order `IndexMap` naming each node by the path it represents."""
        # TODO: Is it better to have a single HigherOrderMapping class?
        order = node_sequence.size(1)
        if node_sequence.size(0) == 0:
            # An order beyond the longest observed path yields a graph without nodes,
            # and `IndexMap` cannot be built from an empty list of IDs.
            return IndexMap()
        if order == 1:
            # Order-1 node indices are first-order node indices, so the mapping carries over.
            return first_order_mapping
        if first_order_mapping.has_ids:
            return IndexMap([tuple(first_order_mapping.to_ids(v.cpu())) for v in node_sequence])
        return IndexMap([tuple(v.tolist()) for v in node_sequence])

    @classmethod
    def from_aggregated(
        cls,
        edge_index: torch.Tensor,
        node_sequence: torch.Tensor,
        first_order_mapping: Optional[IndexMap] = None,
        edge_weight: Optional[torch.Tensor] = None,
        n_first_order: Optional[int] = None,
        aggr: str = "sum",
    ) -> HigherOrderGraph:
        """Aggregate a (possibly duplicated) higher-order edge index into a De Bruijn graph.

        This is the single place where higher-order nodes get their identity: duplicate
        node sequences are merged, edge weights are aggregated, and the higher-order
        `IndexMap` naming each node by its path is built.

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
        """
        if isinstance(edge_index, torch.Tensor) and hasattr(edge_index, "as_tensor"):
            edge_index = edge_index.as_tensor()

        order = node_sequence.size(1)
        if first_order_mapping is None:
            first_order_mapping = IndexMap()
        if n_first_order is None:
            if first_order_mapping.has_ids:
                n_first_order = first_order_mapping.num_ids()
            else:
                n_first_order = int(node_sequence.max().item()) + 1 if node_sequence.numel() > 0 else 0

        data = aggregate_edge_index(edge_index, node_sequence, edge_weight, aggr=aggr).data

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
    def from_aggregated_graph(
        cls,
        g: Graph,
        first_order_mapping: Optional[IndexMap] = None,
        n_first_order: Optional[int] = None,
    ) -> HigherOrderGraph:
        """Adopt an already-aggregated [`Graph`][pathpyG.Graph] as a higher-order graph.

        Used to give the layers computed by a multi-order model their proper type. The
        underlying `data` object is shared, not copied.

        Args:
            g: Aggregated graph carrying a `node_sequence` of shape `(num_nodes, order)`.
            first_order_mapping: Mapping of the underlying first-order node IDs.
            n_first_order: Number of first-order nodes.

        Returns:
            HigherOrderGraph: The same graph, typed as a higher-order graph.
        """
        if isinstance(g, HigherOrderGraph):
            return g
        return cls(
            g.data,
            first_order_mapping=first_order_mapping,
            n_first_order=n_first_order,
            mapping=g.mapping,
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

        return cls.from_aggregated(
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
        lifting the *unaggregated* data, so the edge weights count observed paths rather
        than being implied by lower-order statistics (unlike [`lift`][pathpyG.HigherOrderGraph.lift]).

        Args:
            g: The temporal graph.
            order: The order `k` of the graph to compute.
            delta: The maximum time difference between two consecutive interactions of a path.
            weight: The edge attribute of `g` to use as edge weight.

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            there is no time-respecting path of that length.

        Examples:
            >>> import pathpyG as pp
            >>> t = pp.TemporalGraph.from_edge_list([("a", "c", 1), ("c", "d", 2)])
            >>> print(pp.HigherOrderGraph.from_temporal_graph(t, order=2, delta=1).nodes)
            [('a', 'c'), ('c', 'd')]
        """
        cls._validate_order(order)
        # Imported here because `MultiOrderModel` builds `HigherOrderGraph` layers itself.
        from pathpyG.core.multi_order_model import MultiOrderModel

        return MultiOrderModel.from_temporal_graph(
            g, delta=delta, max_order=order, weight=weight, cached=False
        ).layers[order]

    @classmethod
    def from_path_data(cls, path_data: PathData, order: int = 1, mode: str = "propagation") -> HigherOrderGraph:
        """Create the De Bruijn graph of order `k` modelling paths in [`PathData`][pathpyG.PathData].

        Args:
            path_data: The observed paths.
            order: The order `k` of the graph to compute.
            mode: The process that we assume. Either "diffusion" or "propagation".

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            no observed path is that long.

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

        For the default order 2, every event whose underlying `(u, v)` pair is the same
        collapses into a single second-order node, and repeated continuations become an
        edge weight. Timestamps and the time window `delta` are not represented in the
        result. Other orders are computed from the time-respecting paths that the event
        graph encodes, which for orders above 2 means lifting its continuations further.

        Args:
            eg: The second-order temporal event graph to aggregate.
            order: The order `k` of the graph to compute.

        Returns:
            HigherOrderGraph: A higher-order graph of order `order`. It has no nodes if
            there is no time-respecting path of that length.
        """
        if order != 2:
            cls._validate_order(order)
            from pathpyG.core.multi_order_model import MultiOrderModel

            return MultiOrderModel.from_event_graph(eg, max_order=order, cached=False).layers[order]

        edge_index = eg.data.edge_index.as_tensor()
        # Each continuation carries the weight of the event it starts from, matching the
        # "src" aggregation used when building order-2 layers from a temporal graph.
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        return cls.from_aggregated(
            edge_index,
            eg.data.node_sequence,
            first_order_mapping=eg.first_order_mapping,
            edge_weight=edge_weight,
            n_first_order=eg.n_first_order,
        )

    def lift(self, aggr: str = "src") -> HigherOrderGraph:
        """Return the De Bruijn graph of order `k + 1` obtained by lifting this graph.

        Nodes of the result are the edges of this graph, i.e. the paths of length `k + 1`
        that exist in this graph's topology.

        Warning:
            This lifts an *aggregated* graph, so the resulting edge weights are those
            implied by the order-`k` statistics rather than counts of observed paths of
            length `k + 1`. To fit a layer to observations, use
            [`MultiOrderModel`][pathpyG.MultiOrderModel], which lifts the unaggregated data.

        Args:
            aggr: Aggregation used for the lifted edge weights. One of "src", "dst",
                "max", "mul" or "add".

        Returns:
            HigherOrderGraph: A higher-order graph of order `k + 1`.
        """
        edge_index = self.data.edge_index.as_tensor()
        if "edge_weight" in self.data:
            edge_weight = self.data.edge_weight
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        ho_index, ho_weight = lift_order_edge_index_weighted(
            edge_index, edge_weight=edge_weight, num_nodes=self.n, aggr=aggr
        )
        node_sequence = torch.cat(
            [self.data.node_sequence[edge_index[0]], self.data.node_sequence[edge_index[1]][:, -1:]], dim=1
        )

        return HigherOrderGraph.from_aggregated(
            ho_index,
            node_sequence,
            first_order_mapping=self.first_order_mapping,
            edge_weight=ho_weight,
            n_first_order=self.n_first_order,
        )

    def to_first_order(self, mode: str = "last") -> Graph:
        """Project the higher-order graph back onto the first-order nodes.

        Each higher-order node is replaced by one of the first-order nodes of its path,
        and the weights of higher-order edges mapping to the same first-order edge are
        summed. First-order nodes not traversed by any path remain as isolated nodes.

        Args:
            mode: Which first-order node of the path represents it. Either "last" or "first".

        Returns:
            Graph: A weighted first-order graph.
        """
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

        Used by the [DBGNN][pathpyG.nn.dbgnn.DBGNN] model to pass messages from
        higher-order node representations to first-order ones. Unlike the free function
        [`generate_bipartite_edge_index`][pathpyG.utils.dbgnn.generate_bipartite_edge_index],
        this works for any order: "last" refers to the last node of the path, whatever
        its length.

        Args:
            first_order_graph: The first-order graph. Optional; accepted so that call
                sites read symmetrically, and used only for its device.
            mapping: Which first-order nodes to connect to. One of "last", "first" or "both".
            device: Device on which to create the tensor.

        Returns:
            torch.Tensor: Edge index of shape `(2, ·)`, higher-order nodes in the first row.
        """
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
        if self.order == 1:
            return self.first_order_mapping.to_id(int(seq[0].item()))
        if self.first_order_mapping.has_ids:
            return tuple(self.first_order_mapping.to_ids(seq.cpu()).tolist())
        return tuple(seq.tolist())

    def __str__(self) -> str:
        """Return a human-readable summary of the higher-order graph."""
        s = (
            f"Higher-order graph of order {self.order} with {self.n} nodes and {self.m} edges\n"
            f"(over {self.n_first_order} first-order nodes)\n"
        )
        return s + "\n".join(super().__str__().split("\n")[1:])
