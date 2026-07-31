"""MultiOrderModel module."""

import logging
from typing import (
    Optional,
)

import torch
from scipy.stats import chi2
from torch_geometric.data import Data
from torch_geometric.utils import cumsum, degree

from pathpyG.algorithms.lift_order import (
    aggregate_edge_index,
    aggregate_node_attributes,
    lift_order_edge_index,
    lift_order_edge_index_weighted,
)
from pathpyG.algorithms.temporal import lift_order_temporal, walk_counts
from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils.dbgnn import generate_bipartite_edge_index

logger = logging.getLogger("root")


class MultiOrderModel:
    """MultiOrderModel based on [torch_geometric.data.Data][].

    This class stores multiple higher-order De Bruijn graphs as layers in a dictionary.
    Each layer corresponds to a De Bruijn graph of order k, where k is the key in the dictionary.
    Each graph layer is represented as a [pathpyG.Graph][] object.
    This class provides methods to search for the optimal order of the model based on likelihood ratio tests,
    as well as methods to compute the log-likelihood of observed paths given the model.

    Attributes:
        layers (dict[int, Graph]): A dictionary mapping the order k to the corresponding
            higher-order De Bruijn graph of order k.

    Examples:
        Example where the optimal order is 1:
        >>> import pathpyG as pp
        >>> paths = PathData(IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=3)
        >>> paths.append_walk(("b", "c", "e"), weight=3)
        >>> m = MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(m.estimate_order(paths, max_order=2))
        1

        Example where the optimal order is 2:
        >>> paths = PathData(IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=4)
        >>> paths.append_walk(("b", "c", "e"), weight=4)
        >>> m = MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(m.estimate_order(paths, max_order=2))
        2
    """

    def __init__(self) -> None:
        """Initialize an empty MultiOrderModel."""
        self.layers: dict[int, Graph] = {}

    def __str__(self) -> str:
        """Return a string representation of the higher-order graph."""
        max_order = max(list(self.layers.keys())) if self.layers else 0
        s = f"MultiOrderModel with max. order {max_order}"
        return s

    def to(self, device: torch.device) -> "MultiOrderModel":
        """Convert the graph layers to the given device.

        Args:
            device: The device to convert the graph layers to.

        Returns: The MultiOrderModel with graph layers on the given device.
        """
        for g in self.layers.values():
            g.to(device)
        return self

    @staticmethod
    def iterate_lift_order(
        edge_index: torch.Tensor,
        node_sequence: torch.Tensor,
        mapping: IndexMap,
        edge_weight: torch.Tensor | None = None,
        aggr: str = "src",
        save: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Graph | None]:
        """Lift order by one and save the result in the layers dictionary of the object.

        This is a helper function that should not be called directly.
        Only use for edge_indices after the special cases have been handled e.g.
        in the from_temporal_graph (filtering non-time-respecting paths of order 2).

        Args:
            edge_index: The edge index of the (k-1)-th order graph.
            node_sequence: The node sequences of the (k-1)-th order graph.
            mapping: The [IndexMap][pathpyG.IndexMap] mapping higher-order nodes to first-order nodes.
            edge_weight: The edge weights of the (k-1)-th order graph.
            k: The order of the graph that should be computed.
            aggr: The aggregation method to use. One of "src", "dst", "max", "mul".
            save: Whether to compute the aggregated graph and later save it in the layers dictionary.
        """
        # Lift order
        if edge_weight is None:
            ho_index = lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
        else:
            ho_index, edge_weight = lift_order_edge_index_weighted(
                edge_index, edge_weight=edge_weight, num_nodes=node_sequence.size(0), aggr=aggr
            )
        node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)

        # Aggregate
        if save:
            gk = aggregate_edge_index(ho_index, node_sequence, edge_weight)
            gk.mapping = IndexMap([tuple(mapping.to_ids(v.cpu())) for v in gk.data.node_sequence])
        else:
            gk = None
        return ho_index, node_sequence, edge_weight, gk

    @staticmethod
    def attach_path_statistics(layer: Graph, dag_num_nodes: torch.Tensor, dag_weight: torch.Tensor, order: int) -> None:
        """Attach the per-layer statistics needed to evaluate multi-order likelihoods.

        The likelihood of a set of observed paths under a multi-order model needs, besides the
        aggregated De Bruijn layers themselves, only two statistics:

        - `path_start_weight`: for each order-`k` node, the total weight of observed paths whose
          **first** order-`k` node it is. Paths with fewer than `k` nodes have no order-`k` node
          and drop out automatically.
        - `node_instance_weight` (order 1 only): for each first-order node, the total weight of
          its occurrences across all observed paths. This is the sufficient statistic for the
          zeroth-order node emission probabilities.

        Storing these on the layers removes the need to pass the original path DAG to the
        likelihood methods, and lets models built from different sources share one code path.

        Args:
            layer: The De Bruijn graph of order `order` to attach the statistics to.
            dag_num_nodes: Number of first-order nodes of each observed path.
            dag_weight: Weight (observation frequency) of each observed path.
            order: The order of `layer`.
        """
        device = dag_weight.device
        inverse_idx = layer.data.inverse_idx
        num_nodes = layer.data.num_nodes

        # A path with n first-order nodes contains n - order + 1 order-`order` nodes. Instances are
        # laid out grouped by path, so the running sum over paths locates each path's first one.
        num_instances = dag_num_nodes - order + 1
        long_enough = num_instances > 0
        start_instances = cumsum(num_instances[long_enough])[:-1]

        path_start_weight = torch.zeros(num_nodes, dtype=dag_weight.dtype, device=device)
        path_start_weight.scatter_add_(0, inverse_idx[start_instances], dag_weight[long_enough])
        layer.data.path_start_weight = path_start_weight

        if order == 1:
            node_instance_weight = torch.zeros(num_nodes, dtype=dag_weight.dtype, device=device)
            node_instance_weight.scatter_add_(0, inverse_idx, dag_weight.repeat_interleave(dag_num_nodes))
            layer.data.node_instance_weight = node_instance_weight

    def _require_statistic(self, order: int, name: str) -> torch.Tensor:
        """Return a path statistic stored on a layer, with an actionable error if it is missing."""
        if order not in self.layers:
            logger.error("Layer of order %s is required to compute the likelihood but is missing.", order)
            raise ValueError(
                f"Layer of order {order} is required to compute the likelihood but is missing. "
                "Build the model with `cached=True` so that all intermediate orders are kept."
            )
        if name not in self.layers[order].data:
            logger.error("Layer of order %s does not carry the statistic '%s'.", order, name)
            raise ValueError(
                f"Layer of order {order} does not carry '{name}', so its likelihood cannot be computed. "
                "Likelihoods need a fixed observation set: build the model with "
                "`MultiOrderModel.from_path_data` or `MultiOrderModel.from_time_respecting_walks`. "
                "`from_temporal_graph` weights each layer by the number of distinct walks of that "
                "length, which makes the observed data depend on the order being tested."
            )
        return self.layers[order].data[name]

    @staticmethod
    def from_temporal_graph(
        g: TemporalGraph,
        delta: float | int = 1,
        max_order: int = 1,
        weight: str = "edge_weight",
        cached: bool = True,
        event_graph: Optional[torch.Tensor] = None,
    ) -> "MultiOrderModel":
        """Creates multiple higher-order De Bruijn graph models for paths in a temporal graph.

        Args:
            g: The temporal graph.
            delta: The maximum time difference between two consecutive edges in a path.
            max_order: The maximum order of the MultiOrderModel that should be computed.
            weight: The edge attribute to use as edge weight.
            cached: Whether to save the aggregated higher-order graphs smaller than max order in the MultiOrderModel.
            event_graph: precomputed event graph edge index for given delta to be used for model generation. Useful to prevent the same event graph
            from being computed twice.

        Returns:
            MultiOrderModel: A multi-order model where each layer is a De Bruijn graph with order k.
        """
        m = MultiOrderModel()
        if not g.data.is_sorted_by_time():
            data = g.data.sort_by_time()
        else:
            data = g.data
        edge_index = data.edge_index
        node_sequence = torch.arange(data.num_nodes, device=edge_index.device).unsqueeze(1)
        if weight in data:
            edge_weight = data[weight]
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        if cached or max_order == 1:
            m.layers[1] = aggregate_edge_index(
                edge_index=edge_index, node_sequence=node_sequence, edge_weight=edge_weight
            )
            m.layers[1].mapping = g.mapping

        if max_order > 1:
            node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)
            if event_graph is None:
                edge_index = lift_order_temporal(g, delta)
            else:
                edge_index = event_graph
            edge_weight = aggregate_node_attributes(edge_index, edge_weight, "src")

            # Aggregate
            if cached or max_order == 2:
                m.layers[2] = aggregate_edge_index(
                    edge_index=edge_index, node_sequence=node_sequence, edge_weight=edge_weight
                )
                m.layers[2].mapping = IndexMap(
                    [tuple(g.mapping.to_ids(v.cpu())) for v in m.layers[2].data.node_sequence]
                )

            for k in range(3, max_order + 1):
                edge_index, node_sequence, edge_weight, gk = MultiOrderModel.iterate_lift_order(
                    edge_index=edge_index,
                    node_sequence=node_sequence,
                    mapping=g.mapping,
                    edge_weight=edge_weight,
                    aggr="src",
                    save=cached or k == max_order,
                )
                if cached or k == max_order:
                    m.layers[k] = gk  # type: ignore[assignment]
        return m

    @staticmethod
    def _aggregate_observed(
        edge_index: torch.Tensor, node_sequence: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[Graph, torch.Tensor, torch.Tensor]:
        """Aggregate only the part of a lifted graph that some observation actually realizes.

        The event graph contains walks that cannot be completed to the full observation length;
        those carry weight zero and are not part of the observed data. Dropping them keeps the
        model over the same support a model fitted on the enumerated observations would have,
        and avoids `log(0)` in the likelihood.

        Returns:
            The aggregated graph, the retained node indices, and the map from old node indices
            to positions in that retained set (`-1` where a node was dropped).
        """
        keep = edge_weight > 0
        kept_index = edge_index[:, keep]
        kept_weight = edge_weight[keep]
        # Every observed node is an endpoint of an observed edge, because an observation spanning
        # k-th order nodes always contains at least one k-th order edge.
        used_nodes = torch.unique(kept_index)
        remap = torch.full((node_sequence.size(0),), -1, dtype=torch.long, device=edge_index.device)
        remap[used_nodes] = torch.arange(used_nodes.size(0), device=edge_index.device)
        graph = aggregate_edge_index(remap[kept_index], node_sequence[used_nodes], kept_weight)
        return graph, used_nodes, remap

    @staticmethod
    def from_time_respecting_walks(
        g: TemporalGraph,
        delta: float | int = 1,
        max_order: int = 1,
        walk_length: Optional[int] = None,
        cached: bool = True,
        event_graph: Optional[torch.Tensor] = None,
    ) -> "MultiOrderModel":
        """Create a multi-order model of the time-respecting walks in a temporal graph.

        Unlike [from_temporal_graph][pathpyG.MultiOrderModel.from_temporal_graph], which weights a
        layer-`k` edge by the number of *distinct* time-respecting walks of `k` events, this
        treats a fixed observation set - **all time-respecting walks of exactly `walk_length`
        events** - as the observed data, and weights every layer by how often it is realized
        within that set.

        This distinction matters for model selection. A likelihood ratio test requires both
        hypotheses to be likelihoods of the *same* data, so the observation set must not depend
        on the order being tested. Counting distinct `k`-event walks makes the data a function of
        `k` and silently invalidates the test; fixing `walk_length` independently of `k` does not.

        The statistics are computed directly from the temporal event graph by a depth-bounded
        dynamic program (see [walk_counts][pathpyG.algorithms.walk_counts]), so no walk is ever
        materialized and the cost stays linear in the size of the event graph.

        Args:
            g: The temporal graph.
            delta: The maximum time difference between two consecutive edges in a walk.
            max_order: The maximum order of the MultiOrderModel that should be computed.
            walk_length: Number of events in each observed walk. Defaults to `max_order`, and
                must be at least `max_order`. Models built with different `walk_length` describe
                different observation sets and their likelihoods are not comparable.
            cached: Whether to keep the aggregated higher-order graphs below the maximum order.
                Required for [estimate_order][pathpyG.MultiOrderModel.estimate_order].
            event_graph: Precomputed event graph edge index for the given delta, to avoid
                recomputing it.

        Returns:
            MultiOrderModel: A multi-order model whose layers carry the path statistics needed to
                evaluate multi-order likelihoods.

        Examples:
            >>> import pathpyG as pp
            >>> g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 5), ("c", "d", 9)])
            >>> m = pp.MultiOrderModel.from_time_respecting_walks(g, delta=4, max_order=2)
            >>> print(m)
            MultiOrderModel with max. order 2
        """
        if max_order < 1:
            logger.error("max_order must be at least one")
            raise ValueError(f"max_order must be at least one, got {max_order}")
        if walk_length is None:
            walk_length = max_order
        if walk_length < max_order:
            logger.error("walk_length must be at least max_order")
            raise ValueError(f"walk_length ({walk_length}) must be at least max_order ({max_order})")

        length = walk_length
        data = g.data if g.data.is_sorted_by_time() else g.data.sort_by_time()
        edge_index = data.edge_index
        flat_edge_index = edge_index.as_tensor() if hasattr(edge_index, "as_tensor") else edge_index
        num_events = flat_edge_index.size(1)
        device = flat_edge_index.device

        if event_graph is None:
            event_graph = lift_order_temporal(g, delta)
        # Number of walks of m events continuing after (resp. preceding) each event.
        num_following = walk_counts(event_graph, num_events, length)
        num_preceding = walk_counts(event_graph, num_events, length, reverse=True)

        def realization_weight(first: torch.Tensor, last: torch.Tensor, num_walk_events: int) -> torch.Tensor:
            """Count (observed walk, position) pairs realizing a walk of `num_walk_events` events.

            A walk that spans `j` events sits inside an observation of `length` events in as many
            ways as its surroundings can be filled in, so the count convolves the number of
            possible predecessors with the number of possible continuations.
            """
            total = torch.zeros(first.size(0), dtype=torch.float64, device=device)
            for before in range(length - num_walk_events + 1):
                total += num_preceding[before][first] * num_following[length - num_walk_events - before][last]
            return total

        m = MultiOrderModel()

        # --- First order: every event is a walk of one event. ---
        node_sequence = torch.arange(data.num_nodes, device=device).unsqueeze(1)
        events = torch.arange(num_events, device=device)
        edge_weight = realization_weight(events, events, 1)
        observed_event = edge_weight > 0
        if not bool(observed_event.any()):
            logger.error("No time-respecting walk of length %s exists for delta=%s", length, delta)
            raise ValueError(
                f"The temporal graph contains no time-respecting walk of {length} events for delta={delta}, "
                "so there is nothing to fit. Use a smaller walk_length or a larger delta."
            )

        layer_one, _, remap = MultiOrderModel._aggregate_observed(flat_edge_index, node_sequence, edge_weight)
        layer_one.mapping = g.mapping
        m.layers[1] = layer_one

        # An event that realizes no observation also starts none, so restricting to observed
        # events below loses nothing.
        kept_src = remap[flat_edge_index[0][observed_event]]
        kept_dst = remap[flat_edge_index[1][observed_event]]
        num_first_order_nodes = layer_one.data.num_nodes

        # An observation starts at the source of its first event.
        path_start_weight = torch.zeros(num_first_order_nodes, dtype=torch.float64, device=device)
        path_start_weight.scatter_add_(0, kept_src, num_following[length - 1][observed_event])
        layer_one.data.path_start_weight = path_start_weight
        # Every event contributes its source once per realization; the final node of an
        # observation is instead the destination of its last event.
        node_instance_weight = torch.zeros(num_first_order_nodes, dtype=torch.float64, device=device)
        node_instance_weight.scatter_add_(0, kept_src, edge_weight[observed_event])
        node_instance_weight.scatter_add_(0, kept_dst, num_preceding[length - 1][observed_event])
        layer_one.data.node_instance_weight = node_instance_weight

        if max_order > 1:
            # Nodes of the working graph are walks of k-1 events; its edges are walks of k events.
            node_sequence = torch.cat(
                [node_sequence[flat_edge_index[0]], node_sequence[flat_edge_index[1]][:, -1:]], dim=1
            )
            working_index = event_graph
            node_first_event = events.clone()
            node_last_event = events.clone()

            for k in range(2, max_order + 1):
                if k > 2:
                    ho_index = lift_order_edge_index(working_index, num_nodes=node_sequence.size(0))
                    node_sequence = torch.cat(
                        [node_sequence[working_index[0]], node_sequence[working_index[1]][:, -1:]], dim=1
                    )
                    # Nodes of the lifted graph are the edges of the previous one.
                    node_first_event = node_first_event[working_index[0]]
                    node_last_event = node_last_event[working_index[1]]
                    working_index = ho_index

                edge_weight = realization_weight(
                    node_first_event[working_index[0]], node_last_event[working_index[1]], k
                )

                if cached or k == max_order:
                    gk, used_nodes, _ = MultiOrderModel._aggregate_observed(
                        working_index, node_sequence, edge_weight
                    )
                    gk.mapping = IndexMap([tuple(g.mapping.to_ids(v.cpu())) for v in gk.data.node_sequence])
                    # An observation starts with this order-k node if its remaining events can
                    # still be completed to the full walk length.
                    path_start_weight = torch.zeros(gk.data.num_nodes, dtype=torch.float64, device=device)
                    path_start_weight.scatter_add_(
                        0, gk.data.inverse_idx, num_following[length - k + 1][node_last_event[used_nodes]]
                    )
                    gk.data.path_start_weight = path_start_weight
                    m.layers[k] = gk

        return m

    @staticmethod
    def from_path_data(
        path_data: PathData, max_order: int = 1, mode: str = "propagation", cached: bool = True
    ) -> "MultiOrderModel":
        """Creates multiple higher-order De Bruijn graphs modelling paths in [PathData][pathpyG.PathData].

        Args:
            path_data: [PathData][pathpyG.PathData] object containing paths as list of [Data][torch_geometric.data.Data] objects
                with sorted edge indices, node sequences and num_nodes.
            max_order: The maximum order of the [MultiOrderModel][pathpyG.MultiOrderModel] that should be computed
            mode: The process that we assume. Can be "diffusion" or "propagation".
            cached: Whether to save the aggregated higher-order graphs smaller than max order
                in the [MultiOrderModel][pathpyG.MultiOrderModel].

        Returns:
            MultiOrderModel: The MultiOrderModel.
        """
        m = MultiOrderModel()

        # We assume that paths are sorted
        path_graph = path_data.data
        edge_index = path_graph.edge_index
        node_sequence = path_graph.node_sequence
        edge_weight = path_graph.dag_weight.repeat_interleave(path_graph.dag_num_edges)
        if mode == "diffusion":
            edge_weight = (
                edge_weight / degree(edge_index[0], dtype=torch.long, num_nodes=node_sequence.size(0))[edge_index[0]]
            )
            aggr = "mul"
        elif mode == "propagation":
            aggr = "src"

        m.layers[1] = aggregate_edge_index(edge_index=edge_index, node_sequence=node_sequence, edge_weight=edge_weight)
        m.layers[1].mapping = path_data.mapping
        MultiOrderModel.attach_path_statistics(
            m.layers[1], path_graph.dag_num_nodes, path_graph.dag_weight, order=1
        )

        for k in range(2, max_order + 1):
            edge_index, node_sequence, edge_weight, gk = MultiOrderModel.iterate_lift_order(
                edge_index=edge_index,
                node_sequence=node_sequence,
                mapping=m.layers[1].mapping,
                edge_weight=edge_weight,
                aggr=aggr,
                save=cached or k == max_order,
            )
            if cached or k == max_order:
                m.layers[k] = gk  # type: ignore[assignment]
                MultiOrderModel.attach_path_statistics(
                    m.layers[k], path_graph.dag_num_nodes, path_graph.dag_weight, order=k
                )

        return m

    def get_mon_dof(self, max_order: Optional[int] = None, assumption: str = "paths") -> int:
        """Calculate the degrees of freedom of the multi-order model.

        The degrees of freedom for the kth layer of a multi-order model. This depends on the number of different paths of exactly length `k` in the graph.
        Therefore, we can obtain these values by summing the entries of the `k`-th power of the binary adjacency matrix of the graph.
        Finally, we must consider that, due the conservation of probablility, all non-zero rows of the transition matrix of the higher-order network must sum to one.
        This poses one additional constraint per row that respects the condition, which should be removed from the total count of degrees of freedom.

        Args:
            m (MultiOrderModel): The multi-order model.
            max_order (int, optional): The maximum order up to which model layers
                shall be taken into account. Defaults to None, meaning it considers
                all available layers.
            assumption (str, optional): If set to 'paths', only paths in the
                first-order network topology will be considered for the degree of
                freedom calculation. If set to 'ngrams', all possible n-grams will
                be considered, independent of whether they are valid paths in the
                first-order network or not. Defaults to 'paths'.

        Returns:
            int: The degrees of freedom for the multi-order model.

        Raises:
            ValueError: If max_order is larger than the maximum order of
                the multi-order network.
            ValueError: If the assumption is not 'paths' or 'ngrams'.
        """
        if max_order is None:
            max_order = max(self.layers)

        if max_order > max(self.layers):
            logger.error("max_order cannot be larger than maximum order of multi-order network")
            raise ValueError("max_order cannot be larger than maximum order of multi-order network")

        dof = self.layers[1].data.num_nodes - 1  # Degrees of freedom for zeroth order

        if assumption == "paths":
            # COMPUTING CONTRIBUTION FROM NUM PATHS AND NONZERO OUTDEGREES SEPARATELY
            # TODO: CAN IT BE DONE TOGETHER?

            edge_index = self.layers[1].data.edge_index
            # Adding dof from Number of paths of length k
            for k in range(1, max_order + 1):
                if k > 1:
                    num_nodes = 0 if edge_index.numel() == 0 else edge_index.max().item() + 1
                    edge_index = lift_order_edge_index(edge_index, num_nodes)
                # counting number of len k paths
                num_len_k_paths = edge_index.shape[1]  # edge_index.max().item() +1  # Number of paths of length k
                dof += num_len_k_paths

            # removing dof from total probability of nonzero degree nodes
            for k in range(1, max_order + 1):
                if k == 1:
                    # edge_index of temporal graph is sorted by time by default
                    # For matrix multiplication, we need to sort it by row
                    edge_index_adj = self.layers[1].data.edge_index.sort_by("row")[0]
                    edge_index = edge_index_adj
                else:
                    edge_index, _ = edge_index.matmul(edge_index_adj)
                num_nonzero_outdegrees = torch.unique(edge_index[0]).size(0)
                dof -= num_nonzero_outdegrees

        elif assumption == "ngrams":
            for order in range(1, max_order + 1):
                dof += (self.layers[1].data.num_nodes ** order) * (self.layers[1].data.num_nodes - 1)
        else:
            logger.error("Unknown assumption %s. Only 'path' and 'ngram' are accepted.", assumption)
            raise ValueError(f"Unknown assumption {assumption}. Only 'path' and 'ngram' are accepted.")

        return int(dof)

    def get_zeroth_order_log_likelihood(self, dag_graph: Optional[Data] = None) -> float:
        """Compute the zeroth order log likelihood.

        Args:
            dag_graph (Data, optional): Deprecated and ignored. The required statistics are now
                stored on the model layers by `attach_path_statistics`.

        Returns:
            float: Zeroth order log likelihood.
        """
        node_instance_weight = self._require_statistic(1, "node_instance_weight")
        path_start_weight = self._require_statistic(1, "path_start_weight")

        node_emission_probabilities = node_instance_weight / node_instance_weight.sum()
        # A node that starts no path contributes nothing; skip it so that nodes with zero
        # emission probability cannot turn the sum into NaN.
        llh = torch.where(
            path_start_weight > 0,
            path_start_weight * torch.log(node_emission_probabilities),
            torch.zeros_like(path_start_weight),
        )
        return llh.sum().item()

    def get_intermediate_order_log_likelihood(self, dag_graph: Optional[Data] = None, order: int = 1) -> float:
        """Compute the intermediate order log likelihood.

        For each observed path long enough to reach order `order`, this accounts for the single
        transition that takes the model from order `order` to order `order + 1`, i.e. the one
        emitting the path's `(order + 1)`-th node given its first `order` nodes.

        Args:
            dag_graph (Data, optional): Deprecated and ignored. The required statistics are now
                stored on the model layers by `attach_path_statistics`.
            order (int): Order of the intermediate log likelihood.

        Returns:
            float: Intermediate order log likelihood.
        """
        # An order-(k+1) node *is* an order-k edge, and `aggregate_edge_index` sorts both
        # lexicographically, so the two index spaces coincide elementwise.
        path_start_weight = self._require_statistic(order + 1, "path_start_weight")
        transition_probabilities = self.layers[order].transition_probabilities(edge_attr="edge_weight")

        llh = torch.where(
            path_start_weight > 0,
            path_start_weight * torch.log(transition_probabilities),
            torch.zeros_like(path_start_weight),
        )
        return llh.sum().item()

    def get_mon_log_likelihood(self, dag_graph: Optional[Data] = None, max_order: int = 1) -> float:
        """Compute the likelihood of the walks given a multi-order model.

        Args:
            dag_graph (Data, optional): Deprecated and ignored. The required statistics are now
                stored on the model layers by `attach_path_statistics`.
            max_order (int, optional): The maximum order up to which model layers
                shall be taken into account. Defaults to 1.

        Returns:
            float: The log likelihood of the walks given the multi-order model.
        """
        if max_order == 0:
            # A zeroth-order model emits every node instance independently, so each occurrence
            # contributes, rather than only the first node of each path.
            node_instance_weight = self._require_statistic(1, "node_instance_weight")
            node_emission_probabilities = node_instance_weight / node_instance_weight.sum()
            llh_by_node = torch.where(
                node_instance_weight > 0,
                node_instance_weight * torch.log(node_emission_probabilities),
                torch.zeros_like(node_instance_weight),
            )
            return llh_by_node.sum().item()

        llh = 0.0

        # Adding likelihood of zeroth order
        llh += self.get_zeroth_order_log_likelihood()

        # Adding the likelihood for all the intermediate orders
        for order in range(1, max_order):
            llh += self.get_intermediate_order_log_likelihood(order=order)

        # Adding the likelihood of highest/stationary order
        transition_probabilities = self.layers[max_order].transition_probabilities(edge_attr="edge_weight")
        log_transition_probabilities = torch.log(transition_probabilities)
        llh_by_subpath = log_transition_probabilities * self.layers[max_order].data.edge_weight
        llh += llh_by_subpath.sum().item()

        return llh

    def likelihood_ratio_test(
        self,
        dag_graph: Optional[Data] = None,
        max_order_null: int = 0,
        max_order: int = 1,
        assumption: str = "paths",
        significance_threshold: float = 0.01,
    ) -> tuple:
        """Perform a likelihood ratio test to compare two models of different order.

        Args:
            dag_graph (Data, optional): Deprecated and ignored. The required statistics are now
                stored on the model layers by `attach_path_statistics`.
            max_order_null (int, optional): The maximum order of the null hypothesis model.
                Defaults to 0.
            max_order (int, optional): The maximum order of the alternative hypothesis model.
                Defaults to 1.
            assumption (str, optional): The assumption to use for the degrees of freedom calculation.
                Can be 'paths' or 'ngrams'. Defaults to 'paths'.
            significance_threshold (float, optional): The significance threshold for the test.
                Defaults to 0.01.

        Returns:
            tuple: A tuple containing a boolean indicating whether the null hypothesis is rejected
                and the p-value of the test.
        """
        if max_order_null >= max_order:
            logger.error("order of null hypothesis must be smaller than order of alternative hypothesis")
            raise ValueError("order of null hypothesis must be smaller than order of alternative hypothesis")
        if max_order > max(self.layers):
            logger.error("order of hypotheses must be smaller than max. order of MultiOrderModel")
            raise ValueError(
                f"order of hypotheses ({max_order_null} and {max_order}) must be smaller than max. order of MultiOrderModel {max(self.layers)}"
            )
        # let L0 be the likelihood for the null model and L1 be the likelihood for the alternative model

        # we first compute a test statistic x = -2 * log (L0/L1) = -2 * (log L0 - log L1)
        x = -2 * (
            self.get_mon_log_likelihood(max_order=max_order_null)
            - self.get_mon_log_likelihood(max_order=max_order)
        )

        # we calculate the additional degrees of freedom in the alternative model
        dof_diff = self.get_mon_dof(max_order, assumption=assumption) - self.get_mon_dof(
            max_order_null, assumption=assumption
        )

        # if the p-value is *below* the significance threshold, we reject the null hypothesis
        p = 1 - chi2.cdf(x, dof_diff)
        return (p < significance_threshold), p

    def estimate_order(
        self,
        dag_data: Optional[PathData] = None,
        max_order: Optional[int] = None,
        significance_threshold: float = 0.01,
    ) -> int:
        """Estimate the optimal maximum order of the multi-order network model.

        Selects the optimal maximum order of a multi-order network model for the
        observed paths, based on a likelihood ratio test with p-value threshold of p
        By default, all orders up to the maximum order of the multi-order model will be tested.

        The path statistics needed for the test are stored on the model layers when the model is
        built, so no path data has to be passed in.

        Args:
            dag_data: Deprecated. If given, only used to check that its nodes match those of the
                model; the likelihood is always computed from the statistics stored on the layers.
            max_order (int, optional): The maximum order to consider during the estimation process.
                If not provided, the maximum order of the multi-order model is used.
            significance_threshold (float, optional): The p-value threshold for the likelihood ratio test.
                An order is accepted if the improvement in likelihood is significant at this threshold.

        Returns:
            int: The estimated optimal maximum order for the multi-order network model.

        Raises:
            ValueError: If the provided max_order is larger than the maximum order of the multi-order model
                or if the input does not have the same set of nodes as the multi-order network
        """
        if max_order is None:
            max_order = max(self.layers)
        if max_order > max(self.layers):
            logger.error("max_order cannot be larger than maximum order of multi-order network")
            raise ValueError("max_order cannot be larger than maximum order of multi-order network")
        if max_order <= 1:
            logger.error("max_order must be larger than one")
            raise ValueError("max_order must be larger than one")
        if dag_data is not None:
            if set(dag_data.mapping.node_ids).intersection(set(self.layers[1].mapping.node_ids)) != set(  # type: ignore[arg-type]
                dag_data.mapping.node_ids  # type: ignore[arg-type]
            ):
                logger.error("Input paths do not have same set of nodes as multi-order network")
                raise ValueError("Input paths do not have same set of nodes as multi-order network")

        max_accepted_order = 1

        # Test for highest order that passes
        # likelihood ratio test against null model
        for k in range(2, max_order + 1):
            if self.likelihood_ratio_test(
                max_order_null=k - 1, max_order=k, significance_threshold=significance_threshold
            )[0]:
                max_accepted_order = k

        return max_accepted_order

    def to_dbgnn_data(self, max_order: int = 2, mapping: str = "last") -> Data:
        """Convert the MultiOrderModel to a De Bruijn graph for the given maximum order that can be used in the [DBGNN][pathpyG.nn.dbgnn.DBGNN]-model.

        Args:
            max_order: The maximum order of the De Bruijn graph to be computed.
            mapping: The mapping to use for the bipartite edge index. One of "last", "first", or "both".

        Returns:
            Data: The De Bruijn graph data.
        """
        if max_order not in self.layers:
            logger.error("Higher-order graph of specified order not found.")
            raise ValueError(f"Higher-order graph of order {max_order} not found.")

        g = self.layers[1]
        g_max_order = self.layers[max_order]
        num_nodes = g.data.num_nodes
        num_ho_nodes = g_max_order.data.num_nodes
        if g.data.x is not None:
            x = g.data.x
        else:
            x = torch.eye(num_nodes, num_nodes, device=g.data.edge_index.device)
        x_max_order = torch.eye(num_ho_nodes, num_ho_nodes, device=g_max_order.data.edge_index.device)
        edge_index = g.data.edge_index
        edge_index_max_order = g_max_order.data.edge_index
        edge_weight = g.data.edge_weight
        edge_weight_max_order = g_max_order.data.edge_weight
        bipartite_edge_index = generate_bipartite_edge_index(g, g_max_order, mapping=mapping, device=edge_index.device)

        if g.data.y is not None:
            y = g.data.y

        return Data(
            num_nodes=num_nodes,
            num_ho_nodes=num_ho_nodes,
            x=x,
            x_h=x_max_order,
            edge_index=edge_index,
            edge_index_higher_order=edge_index_max_order,
            edge_weights=edge_weight.float(),
            edge_weights_higher_order=edge_weight_max_order.float(),
            bipartite_edge_index=bipartite_edge_index,
            y=y if "y" in locals() else None,
        )
