"""Multi-order De Bruijn graph models and statistical model selection for path data."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from scipy.stats import chi2
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from pathpyG.algorithms.lift_order import (
    lift_node_sequence,
    lift_order_edge_index,
    lift_order_step,
)
from pathpyG.algorithms.temporal import walk_counts
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.graph import Graph
from pathpyG.core.higher_order_graph import HigherOrderGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")

ORDER_SELECTION_RULES = (
    "largest_significant",
    "dominates_all_lower_orders",
    "stop_at_first_insignificant",
)
SELECTION_METHODS = ("likelihood_ratio", "bayesian")
NORMALIZATION_CONSTRAINTS = ("per_history", "per_first_order_node")
DEFAULT_ORDER_SELECTION = {
    "likelihood_ratio": "largest_significant",
    "bayesian": "dominates_all_lower_orders",
}


def _as_tensor(edge_index: torch.Tensor) -> torch.Tensor:
    """Return a plain tensor view of an edge index that may be a `torch_geometric.EdgeIndex`."""
    return edge_index.as_tensor() if hasattr(edge_index, "as_tensor") else edge_index


def _sorted_by_source(edge_index: torch.Tensor) -> torch.Tensor:
    """Group an edge index by source node, as the line-graph transformation requires."""
    edge_index = _as_tensor(edge_index)
    if edge_index.size(1) == 0:
        return edge_index
    order = torch.argsort(edge_index[0], stable=True)
    return edge_index[:, order]


def _align_rows(source_rows: torch.Tensor, target_rows: torch.Tensor) -> torch.Tensor:
    """Locate each row of `source_rows` in `target_rows`.

    Rows are matched as whole sequences, without encoding them into a single integer, so this is
    safe for any number of first-order nodes and any order.

    Args:
        source_rows: Rows to look up, of shape `(num_source, width)`.
        target_rows: Rows to look them up in, of shape `(num_target, width)`.

    Returns:
        Tensor of shape `(num_source,)` with the position of each source row in `target_rows`,
        or `-1` where the row does not occur there.
    """
    device = source_rows.device
    num_target = target_rows.size(0)
    if source_rows.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if num_target == 0:
        return torch.full((source_rows.size(0),), -1, dtype=torch.long, device=device)

    combined = torch.cat([target_rows, source_rows], dim=0)
    _, inverse = torch.unique(combined, dim=0, return_inverse=True)
    target_key, source_key = inverse[:num_target], inverse[num_target:]
    lookup = torch.full((int(inverse.max().item()) + 1,), -1, dtype=torch.long, device=device)
    lookup[target_key] = torch.arange(num_target, device=device)
    return lookup[source_key]


def _index_translation(source: IndexMap, target: IndexMap, num_source_nodes: int, device: torch.device) -> torch.Tensor:
    """Map first-order node indices of `source` onto the indices the `target` mapping uses."""
    if not source.has_ids or not target.has_ids:
        return torch.arange(num_source_nodes, device=device)
    identifiers = source.to_ids(torch.arange(num_source_nodes).numpy())
    try:
        return target.to_idxs(identifiers, device=device)
    except Exception as error:  # pragma: no cover - depends on the IndexMap implementation
        logger.error("first-order node sets cannot be aligned: %s", error)
        raise ValueError(
            "the first-order node sets cannot be aligned: the other object refers to nodes "
            "that this model does not know about"
        ) from error


class MultiOrderModel:
    r"""A multi-order De Bruijn graph model of path data, and the statistics to select its order.

    The model stores one [`HigherOrderGraph`][pathpyG.HigherOrderGraph] per order $k$ in `layers`.
    Layer $k$ has one node per observed path of $k$ first-order nodes and one edge per observed path
    of $k+1$ first-order nodes, weighted by how often each occurs in the data.

    Together with the `node_weight` of each layer, that is everything the multi-order likelihood
    needs, so none of the methods below takes the observed data again: a fitted model *is* its own
    sufficient statistic.

    Info: Choosing the order
        Two selectors are available and both are reached through
        [`estimate_order`][pathpyG.core.multi_order_model.MultiOrderModel.estimate_order]:

        - `method="likelihood_ratio"` compares nested models with a $\chi^2$ likelihood ratio test.
        - `method="bayesian"` compares the models' marginal likelihoods (Bayes factors), which needs
          considerably less data and degrades more gracefully when the underlying graph is only
          partially known.

    ??? reference
        The multi-order model and the likelihood ratio test are due to Scholtes[^1]; the Bayesian
        selector and the corrected parameter count are due to Petrović and Scholtes[^2].

    [^1]: *I Scholtes: When is a Network a Network? Multi-Order Graphical Model Selection in
        Pathways and Temporal Networks, In Proc. of SIGKDD 2017.
        [doi:10.1145/3097983.3098145](https://doi.org/10.1145/3097983.3098145)*
    [^2]: *L Petrović, I Scholtes: Learning the Markov Order of Paths in Graphs, In Proc. of the ACM
        Web Conference (WWW'22), 2022.
        [doi:10.1145/3485447.3512091](https://doi.org/10.1145/3485447.3512091)*

    Attributes:
        layers (dict[int, HigherOrderGraph]): The De Bruijn graph of each computed order.
        walk_length (int | None): For a model of a temporal graph, the number of events of each
            observed time-respecting walk. `None` for a model of explicitly observed paths.
        num_observations (float): Total weight of the observed paths or walks.

    Examples:
        Example where the optimal order is 1:
        >>> import pathpyG as pp
        >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=3)
        >>> paths.append_walk(("b", "c", "e"), weight=3)
        >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(model.estimate_order(max_order=2))
        1

        Example where the optimal order is 2:
        >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=4)
        >>> paths.append_walk(("b", "c", "e"), weight=4)
        >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(model.estimate_order(max_order=2))
        2

        Each layer knows the first-order path that each of its nodes represents:
        >>> print(model.layers[2].order, model.layers[2].nodes)
        2 [('a', 'c'), ('b', 'c'), ('c', 'd'), ('c', 'e')]
    """

    def __init__(self) -> None:
        """Initialize an empty MultiOrderModel."""
        self.layers: dict[int, HigherOrderGraph] = {}
        self.walk_length: Optional[int] = None
        self.num_observations: float = 0.0

    def __str__(self) -> str:
        """Return a string representation of the multi-order model."""
        max_order = max(self.layers) if self.layers else 0
        summary = f"MultiOrderModel with max. order {max_order}"
        if self.walk_length is not None:
            summary += f" of time-respecting walks of {self.walk_length} events"
        return summary

    def to(self, device: torch.device) -> MultiOrderModel:
        """Convert the graph layers to the given device.

        Args:
            device: The device to convert the graph layers to.

        Returns: The MultiOrderModel with graph layers on the given device.
        """
        for layer in self.layers.values():
            layer.to(device)
        return self

    # ------------------------------------------------------------------ construction

    @staticmethod
    def iterate_lift_order(
        edge_index: torch.Tensor,
        node_sequence: torch.Tensor,
        mapping: IndexMap,
        edge_weight: torch.Tensor | None = None,
        aggr: str = "src",
        save: bool = True,
        n_first_order: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, HigherOrderGraph | None]:
        """Lift order by one and optionally aggregate the result into a De Bruijn graph.

        This is a helper function that should not be called directly. Only use it for edge indices
        after the special cases have been handled, e.g. in `from_temporal_graph`, which filters
        non-time-respecting paths of order 2.

        A node of the lifted graph is an edge of the input graph, so the input's `edge_weight` is
        exactly the per-instance `node_weight` of the lifted graph.

        Args:
            edge_index: The edge index of the (k-1)-th order graph.
            node_sequence: The node sequences of the (k-1)-th order graph.
            mapping: The [IndexMap][pathpyG.IndexMap] mapping higher-order nodes to first-order nodes.
            edge_weight: The edge weights of the (k-1)-th order graph.
            aggr: The aggregation method to use. One of "src", "dst", "max", "mul".
            save: Whether to compute the aggregated graph and later save it in the layers dictionary.
            n_first_order: The number of first-order nodes the node sequences refer to.
                Defaults to the number of IDs in `mapping`.

        Returns:
            The lifted edge index, the lifted node sequences, the lifted edge weights and, if
            `save` is set, the aggregated De Bruijn graph of the lifted order.
        """
        lifted_edge_index, lifted_node_sequence, lifted_edge_weight = lift_order_step(
            edge_index, node_sequence, edge_weight=edge_weight, aggr=aggr
        )

        layer = None
        if save:
            layer = HigherOrderGraph.aggregate(
                lifted_edge_index,
                lifted_node_sequence,
                first_order_mapping=mapping,
                edge_weight=lifted_edge_weight,
                node_weight=edge_weight,
                n_first_order=n_first_order,
            )
        return lifted_edge_index, lifted_node_sequence, lifted_edge_weight, layer

    @staticmethod
    def from_path_data(path_data: PathData, max_order: int = 1, cached: bool = True) -> MultiOrderModel:
        """Create De Bruijn graph models for the paths in a [PathData][pathpyG.PathData] object.

        Args:
            path_data: The observed paths.
            max_order: The maximum order of the model that should be computed.
            cached: Whether to keep the aggregated higher-order graphs below the maximum order.
                Required for every model-selection method.

        Returns:
            MultiOrderModel: A multi-order model where each layer is a De Bruijn graph of order k.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abc")))
            >>> paths.append_walk(("a", "b", "c"), weight=2)
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> print(model.layers[1].data.node_weight.tolist())
            [2.0, 2.0, 2.0]
        """
        if max_order < 1:
            logger.error("max_order must be at least one, got %s", max_order)
            raise ValueError(f"max_order must be at least one, got {max_order}")

        model = MultiOrderModel()
        data = path_data.data
        edge_index = _as_tensor(data.edge_index)
        node_sequence = data.node_sequence
        # Each path contributes its weight to every node and every edge it consists of.
        edge_weight = data.dag_weight.repeat_interleave(data.dag_num_edges)
        node_weight = data.dag_weight.repeat_interleave(data.dag_num_nodes)

        observed_nodes = int(node_sequence.max().item()) + 1 if node_sequence.numel() > 0 else 0
        # Nodes that are not traversed by any path are still first-order nodes of the model.
        num_first_order_nodes = max(path_data.mapping.num_ids(), observed_nodes)

        model.num_observations = float(data.dag_weight.sum().item())
        model.layers[1] = HigherOrderGraph.aggregate(
            edge_index=edge_index,
            node_sequence=node_sequence,
            first_order_mapping=path_data.mapping,
            edge_weight=edge_weight,
            node_weight=node_weight,
            n_first_order=num_first_order_nodes,
        )

        for order in range(2, max_order + 1):
            edge_index, node_sequence, edge_weight, layer = MultiOrderModel.iterate_lift_order(
                edge_index=edge_index,
                node_sequence=node_sequence,
                mapping=path_data.mapping,
                edge_weight=edge_weight,
                aggr="src",
                save=cached or order == max_order,
                n_first_order=num_first_order_nodes,
            )
            if layer is not None:
                model.layers[order] = layer

        return model

    @staticmethod
    def from_temporal_graph(
        graph: TemporalGraph,
        delta: float | int = 1,
        max_order: int = 1,
        walk_length: Optional[int] = None,
        weight: str = "edge_weight",
        cached: bool = True,
        event_graph: Optional[torch.Tensor] = None,
    ) -> MultiOrderModel:
        r"""Create De Bruijn graph models for the time-respecting walks in a temporal graph.

        The observed data is the set of **all time-respecting walks of exactly `walk_length`
        events**: sequences of time-stamped edges where each edge starts where the previous one
        ended and follows it by at most `delta`. Each layer is weighted by how often it is realized
        within that set.

        Fixing `walk_length` independently of the order matters for model selection: a likelihood
        ratio test compares two likelihoods *of the same data*, so the observation set must not
        depend on the order being tested. The number of *distinct* `k`-event walks, which is what
        `edge_num_walks` counts, does depend on `k` and is therefore kept separately.

        Info: Layer attributes
            Each layer of order $k$ carries

            - `node_weight` and `edge_weight`: how often each walk of $k$ resp. $k+1$ first-order
              nodes is realized inside an observation, counting a walk once per position it can
              take. These are the statistics every model-selection method uses.
            - `edge_num_walks`: the number of *distinct* time-respecting walks of $k$ events that
              realize the edge. This is the classical De Bruijn graph weight.

            The two coincide for $k =$ `walk_length`, and differ below it.

            A layer keeps the full De Bruijn topology of the temporal graph, including walks that no
            observation of `walk_length` events can contain -- an event isolated in the event graph,
            for instance. Those carry `node_weight` and `edge_weight` zero and contribute nothing to
            any likelihood, but they remain part of the graph.

        Note: Numerics
            The counts are computed in `float64` by a dynamic program (see
            [`walk_counts`][pathpyG.algorithms.walk_counts]) whose values grow like the branching
            factor of the event graph to the power of `walk_length`. Overflow is detected and
            reported rather than silently producing infinities.

        Args:
            graph: The temporal graph.
            delta: The maximum time difference between two consecutive events of a walk.
            max_order: The maximum order of the model that should be computed.
            walk_length: Number of events of each observed walk. Defaults to `max_order` and must be
                at least `max_order`. Models built with a different `walk_length` describe different
                observation sets and their likelihoods are not comparable.
            weight: The edge attribute of the temporal graph to use as edge weight.
            cached: Whether to keep the aggregated higher-order graphs below the maximum order.
                Required for every model-selection method.
            event_graph: Precomputed event graph edge index for the given delta, to avoid computing
                the same event graph twice.

        Returns:
            MultiOrderModel: A multi-order model where each layer is a De Bruijn graph of order k.

        Examples:
            >>> import pathpyG as pp
            >>> t = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 3)])
            >>> model = pp.MultiOrderModel.from_temporal_graph(t, delta=1, max_order=2)
            >>> print(model)
            MultiOrderModel with max. order 2 of time-respecting walks of 2 events
        """
        if max_order < 1:
            logger.error("max_order must be at least one, got %s", max_order)
            raise ValueError(f"max_order must be at least one, got {max_order}")
        if walk_length is None:
            walk_length = max_order
        if walk_length < max_order:
            logger.error("walk_length %s is smaller than max_order %s", walk_length, max_order)
            raise ValueError(f"walk_length ({walk_length}) must be at least max_order ({max_order})")

        data = graph.data if graph.data.is_sorted_by_time() else graph.data.sort_by_time()
        edge_index = _as_tensor(data.edge_index)
        num_events = edge_index.size(1)
        device = edge_index.device

        if weight in data:
            # The De Bruijn weights keep the dtype of the temporal graph; only the observation
            # statistics need the range of float64.
            num_walks_per_event = data[weight]
        else:
            num_walks_per_event = torch.ones(num_events, device=device)

        if event_graph is None:
            event_graph = EventGraph.build_edge_index(graph, delta)
        event_graph = _as_tensor(event_graph)

        num_following = walk_counts(event_graph, num_events, walk_length)
        num_preceding = walk_counts(event_graph, num_events, walk_length, reverse=True)
        MultiOrderModel._check_walk_count_overflow(num_following, num_preceding, walk_length, delta)

        def realization_weight(
            first_event: torch.Tensor, last_event: torch.Tensor, num_walk_events: int
        ) -> torch.Tensor:
            r"""Count the (observation, position) pairs that realize a walk of `num_walk_events`.

            A walk spanning $j$ events sits inside an observation of `walk_length` events in as many
            ways as its surroundings can be filled in, so the count convolves the number of possible
            predecessors with the number of possible continuations:

            $$
            \sum_{b=0}^{L-j} N^{\text{pre}}_{b}(\text{first}) \cdot N^{\text{post}}_{L-j-b}(\text{last}).
            $$
            """
            span = walk_length - num_walk_events
            preceding = num_preceding[: span + 1]
            following = num_following[: span + 1].flip(0)
            return (preceding[:, first_event] * following[:, last_event]).sum(0)

        model = MultiOrderModel()
        model.walk_length = walk_length

        events = torch.arange(num_events, device=device)
        # How often each event is used, counting once per observation and position.
        event_realizations = realization_weight(events, events, 1)
        if not bool((event_realizations > 0).any()):
            # The De Bruijn layers are still well defined -- they describe the topology -- but no
            # observation realizes any of them, so every statistic is zero and no likelihood exists.
            logger.warning(
                "the temporal graph contains no time-respecting walk of %s events for delta=%s, "
                "so the layers carry no observation statistics and model selection is not possible",
                walk_length,
                delta,
            )
        # An observation of `walk_length` events is identified by its last event, which is preceded
        # by exactly `walk_length - 1` events.
        observations_ending_at = num_preceding[walk_length - 1]
        model.num_observations = float(observations_ending_at.sum().item())

        # A first-order node occurs in an observation either as the source of one of its events or,
        # once per observation, as the destination of its last event.
        node_weight = torch.zeros(graph.n, dtype=torch.float64, device=device)
        node_weight.scatter_add_(0, edge_index[0], event_realizations)
        node_weight.scatter_add_(0, edge_index[1], observations_ending_at)

        node_sequence = torch.arange(graph.n, device=device).unsqueeze(1)
        if cached or max_order == 1:
            model.layers[1] = HigherOrderGraph.aggregate(
                edge_index=edge_index,
                node_sequence=node_sequence,
                edge_weight=event_realizations,
                node_weight=node_weight,
                edge_attrs={"edge_num_walks": num_walks_per_event},
                first_order_mapping=graph.mapping,
                n_first_order=graph.n,
            )

        if max_order > 1:
            # Nodes of the working graph are walks of k-1 events, its edges walks of k events.
            node_sequence = lift_node_sequence(edge_index, node_sequence)
            working_index = _sorted_by_source(event_graph)
            first_event = events.clone()
            last_event = events.clone()
            num_walks_per_edge = num_walks_per_event[working_index[0]]

            for order in range(2, max_order + 1):
                if order > 2:
                    node_sequence = lift_node_sequence(working_index, node_sequence)
                    lifted_index = lift_order_edge_index(working_index, num_nodes=first_event.size(0))
                    # Nodes of the lifted graph are the edges of the previous one.
                    first_event = first_event[working_index[0]]
                    last_event = last_event[working_index[1]]
                    num_walks_per_edge = num_walks_per_edge[lifted_index[0]]
                    working_index = lifted_index

                node_weight = realization_weight(first_event, last_event, order - 1)
                edge_weight = realization_weight(first_event[working_index[0]], last_event[working_index[1]], order)
                if cached or order == max_order:
                    model.layers[order] = HigherOrderGraph.aggregate(
                        edge_index=working_index,
                        node_sequence=node_sequence,
                        edge_weight=edge_weight,
                        node_weight=node_weight,
                        edge_attrs={"edge_num_walks": num_walks_per_edge},
                        first_order_mapping=graph.mapping,
                        n_first_order=graph.n,
                    )

        return model

    @staticmethod
    def _check_walk_count_overflow(
        num_following: torch.Tensor, num_preceding: torch.Tensor, walk_length: int, delta: float | int
    ) -> None:
        """Refuse to return silently wrong statistics when the walk counts leave the float64 range."""
        largest = max(float(num_following.max().item()), float(num_preceding.max().item()))
        if not torch.isfinite(num_following).all() or not torch.isfinite(num_preceding).all():
            logger.error("time-respecting walk counts overflowed for walk_length=%s", walk_length)
            raise ValueError(
                f"the number of time-respecting walks of {walk_length} events exceeds the range of "
                f"float64 for delta={delta}. Reduce walk_length or delta."
            )
        # The layer statistics multiply a predecessor count by a successor count.
        if largest > 1e150:
            logger.error("time-respecting walk counts are close to overflowing (max %g)", largest)
            raise ValueError(
                f"the number of time-respecting walks of {walk_length} events ({largest:g}) is so "
                f"large that the layer statistics would overflow float64 for delta={delta}. "
                "Reduce walk_length or delta."
            )

    @classmethod
    def from_event_graph(
        cls,
        event_graph: EventGraph,
        max_order: int = 2,
        walk_length: Optional[int] = None,
        cached: bool = True,
    ) -> MultiOrderModel:
        """Create a multi-order model from a pre-built event graph.

        Args:
            event_graph: The second-order temporal `EventGraph` to build the model from.
            max_order: The maximum order of the model to compute.
            walk_length: Number of events of each observed walk. Defaults to `max_order`.
            cached: Whether to also keep the aggregated layers below `max_order`.

        Returns:
            MultiOrderModel: A multi-order model equivalent to
            `MultiOrderModel.from_temporal_graph(event_graph.to_temporal_graph(), delta=event_graph.delta, ...)`.
        """
        return cls.from_temporal_graph(
            event_graph.to_temporal_graph(),
            delta=event_graph.delta,
            max_order=max_order,
            walk_length=walk_length,
            cached=cached,
            event_graph=_as_tensor(event_graph.data.edge_index),
        )

    # ------------------------------------------------------------------ statistics

    def _require_layer(self, order: int) -> HigherOrderGraph:
        """Return the layer of the given order, with an actionable error if it is missing."""
        if order not in self.layers:
            logger.error("layer of order %s is missing", order)
            raise ValueError(
                f"the layer of order {order} is required but missing. Build the model with "
                "`cached=True` so that all intermediate orders are kept."
            )
        layer = self.layers[order]
        if "node_weight" not in layer.data:
            logger.error("layer of order %s does not carry node weights", order)
            raise ValueError(
                f"the layer of order {order} does not carry a `node_weight`, so no likelihood can "
                "be computed for it. Build the model with `from_path_data` or `from_temporal_graph`."
            )
        return layer

    def _start_weight(self, order: int) -> torch.Tensor:
        r"""Weight of the observations that *start* with each higher-order node of the given order.

        Derived from the node weight identity
        $\mathrm{node\_weight}_k = \mathrm{start\_weight}_k + \sum_{u \to w} \mathrm{edge\_weight}_k$,
        so no separate statistic has to be stored.
        """
        layer = self._require_layer(order)
        edge_index = _as_tensor(layer.data.edge_index)
        incoming_weight = scatter(layer.data.edge_weight, edge_index[1], dim=0, dim_size=layer.n, reduce="sum")
        return layer.data.node_weight - incoming_weight

    def _edge_sequences(self, order: int) -> torch.Tensor:
        """First-order node sequences of the edges of a layer, of shape `(num_edges, order + 1)`."""
        layer = self._require_layer(order)
        return lift_node_sequence(_as_tensor(layer.data.edge_index), layer.data.node_sequence)

    @staticmethod
    def _weighted_log(counts: torch.Tensor, probabilities: torch.Tensor, positions: torch.Tensor) -> float:
        """Sum `counts * log(probabilities[positions])`, ignoring zero counts.

        Returns negative infinity if something that was observed has no counterpart in the model.
        """
        observed = counts > 0
        if not bool(observed.any()):
            return 0.0
        selected = positions[observed]
        if bool((selected < 0).any()):
            return float("-inf")
        return float((counts[observed].double() * torch.log(probabilities[selected].double())).sum().item())

    def _translation_from(self, data: MultiOrderModel) -> torch.Tensor:
        """Map the first-order node indices of another model onto this model's indices."""
        layer_one = self._require_layer(1)
        other_layer_one = data._require_layer(1)
        return _index_translation(
            other_layer_one.first_order_mapping,
            layer_one.first_order_mapping,
            other_layer_one.n_first_order,
            layer_one.data.edge_index.device,
        )

    def log_likelihood(self, max_order: int = 1, data: Optional[MultiOrderModel] = None) -> float:
        r"""Log-likelihood of the observed paths under the multi-order model of the given order.

        A multi-order model of maximum order $K$ generates a path $v_0 \to \dots \to v_l$ by drawing
        its first node from a zeroth-order distribution, the next $K-1$ nodes from the layers of
        increasing order, and every remaining node from the layer of order $K$:

        $$
        P^{(K)}(v_0 \to \dots \to v_l) = \prod_{k=0}^{K-1} P^{(k)}(v_k \mid v_0 \to \dots \to v_{k-1})
            \cdot \prod_{i=K}^{l} P^{(K)}(v_i \mid v_{i-K} \to \dots \to v_{i-1}).
        $$

        Summed over the observed data, and writing $s_k$ for the weight of the observations that
        start with a given node of layer $k$ and $w_K$ for the edge weights of layer $K$, this is

        $$
        \log \mathcal{L}(K) = \sum_{v} s_1(v) \log P^{(0)}(v)
            + \sum_{k=1}^{K-1} \sum_{e \in E_k} s_{k+1}(e) \log P^{(k)}(e)
            + \sum_{e \in E_K} w_K(e) \log P^{(K)}(e),
        $$

        where the zeroth-order distribution $P^{(0)}$ is the relative visitation frequency of the
        first-order nodes and $P^{(k)}$ are the relative transition frequencies of layer $k$. For
        $K = 0$ every node of every path is drawn independently, so the first sum uses the full node
        weights rather than $s_1$.

        Note: The intermediate orders use *observed* transition frequencies
            Every layer is fitted to all transitions of its order that occur in the data, and the
            terms above then evaluate those fitted probabilities on the transitions each order is
            responsible for. Scoring the intermediate orders with uniform probabilities instead
            biases model selection towards lower orders.

        Args:
            max_order: The maximum order $K$ of the multi-order model.
            data: Another model whose observations should be scored under *this* model's
                probabilities. Defaults to this model, giving the in-sample log-likelihood.

        Returns:
            float: The log-likelihood. Negative infinity if `data` contains a transition that this
            model assigns no probability to.

        Raises:
            ValueError: If a required layer is missing, or if the first-order node sets of the two
                models cannot be aligned.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
            >>> paths.append_walk(("a", "c", "d"))
            >>> paths.append_walk(("b", "c", "e"))
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> print(round(model.log_likelihood(max_order=2), 4))
            -3.5835
        """
        if max_order < 0:
            logger.error("max_order must not be negative, got %s", max_order)
            raise ValueError(f"max_order must not be negative, got {max_order}")

        in_sample = data is None or data is self
        if data is None:
            data = self
        if not in_sample and data.walk_length != self.walk_length:
            logger.warning(
                "scoring observations of a model built with walk_length=%s under a model built with "
                "walk_length=%s; the two describe different observation sets",
                data.walk_length,
                self.walk_length,
            )

        layer_one = self._require_layer(1)
        node_weight = layer_one.data.node_weight
        total_node_weight = float(node_weight.sum().item())
        if total_node_weight == 0:
            logger.error("the model carries no observations, so it has no likelihood")
            raise ValueError(
                "the model carries no observations at all, so no likelihood is defined for it. "
                "For a temporal graph this means no time-respecting walk of `walk_length` events "
                "exists; use a smaller walk_length or a larger delta."
            )
        node_probabilities = node_weight / total_node_weight

        if in_sample:
            translation = torch.arange(layer_one.n, device=node_weight.device)
        else:
            translation = self._translation_from(data)

        zeroth_order_counts = data._require_layer(1).data.node_weight if max_order == 0 else data._start_weight(1)
        total = MultiOrderModel._weighted_log(zeroth_order_counts, node_probabilities, translation)

        for order in range(1, max_order + 1):
            if order < max_order:
                # The first `max_order` transitions of each observation are drawn at increasing
                # order; an order-(k+1) node is an order-k edge, so its start weight is the count.
                counts = data._start_weight_on_edges(order)
            else:
                counts = data._require_layer(order).data.edge_weight
            total += self._order_log_likelihood(order, counts, data, translation, in_sample)
            if total == float("-inf"):
                return total

        return total

    def _start_weight_on_edges(self, order: int) -> torch.Tensor:
        """Start weights of the order-(k+1) nodes, indexed by the edges of layer k they correspond to.

        An order-(k+1) node and an order-k edge are the same sequence of k+1 first-order nodes, so
        the two layers describe the same set. They are **not** stored in the same order, though:
        `Graph` sorts an edge index by source node only, which leaves the edges of one source in an
        arbitrary order, so the two have to be matched by their sequences rather than by position.
        """
        layer = self._require_layer(order)
        next_layer = self._require_layer(order + 1)
        start_weight = self._start_weight(order + 1)
        edge_sequences = self._edge_sequences(order)
        if edge_sequences.shape == next_layer.data.node_sequence.shape and torch.equal(
            edge_sequences, next_layer.data.node_sequence
        ):
            # The orders happen to agree, which they do on small graphs. Nothing relies on it.
            return start_weight
        positions = _align_rows(edge_sequences, next_layer.data.node_sequence)
        counts = torch.zeros(layer.m, dtype=start_weight.dtype, device=start_weight.device)
        known = positions >= 0
        counts[known] = start_weight[positions[known]]
        return counts

    def _order_log_likelihood(
        self,
        order: int,
        counts: torch.Tensor,
        data: MultiOrderModel,
        translation: torch.Tensor,
        in_sample: bool,
    ) -> float:
        """Contribution of one order to the multi-order log-likelihood."""
        layer = self._require_layer(order)
        probabilities = layer.transition_probabilities(edge_attr="edge_weight")
        if in_sample:
            positions = torch.arange(layer.m, device=probabilities.device)
        else:
            positions = _align_rows(translation[data._edge_sequences(order)], self._edge_sequences(order))
        return MultiOrderModel._weighted_log(counts, probabilities, positions)

    # ------------------------------------------------------------------ model selection

    def _constraint_topology(self, constraint_graph: Optional[Graph] = None) -> tuple[torch.Tensor, int]:
        """Return the deduplicated first-order topology the model space is defined over."""
        layer_one = self._require_layer(1)
        observed_edges = torch.unique(_as_tensor(layer_one.data.edge_index), dim=1)
        if constraint_graph is None:
            return _sorted_by_source(observed_edges), layer_one.n_first_order

        device = observed_edges.device
        translation = _index_translation(
            constraint_graph.mapping, layer_one.first_order_mapping, constraint_graph.n, device
        )
        edge_index = torch.unique(translation[_as_tensor(constraint_graph.data.edge_index).to(device)], dim=1)
        num_first_order_nodes = layer_one.n_first_order
        if translation.numel() > 0:
            num_first_order_nodes = max(num_first_order_nodes, int(translation.max().item()) + 1)

        positions = _align_rows(observed_edges.t(), edge_index.t())
        if bool((positions < 0).any()):
            missing = int((positions < 0).sum().item())
            logger.error("constraint graph is missing %s observed transitions", missing)
            raise ValueError(
                f"the constraint graph does not contain {missing} transition(s) that the model's "
                "first-order layer has, so the fitted probabilities would lie outside the declared "
                "model space. Pass a graph that contains every transition of the first-order layer."
            )
        return _sorted_by_source(edge_index), num_first_order_nodes

    def degrees_of_freedom(
        self,
        max_order: Optional[int] = None,
        normalization_constraints: str = "per_history",
        constraint_graph: Optional[Graph] = None,
    ) -> int:
        r"""Number of free parameters of the multi-order model of the given maximum order.

        A $k$-th order layer has one free transition probability per pair of a history $h$ -- a path
        of $k$ first-order nodes -- and an admissible successor $v \in S(h)$, minus one normalisation
        constraint. Counting only paths that actually exist in the first-order topology $G$,

        $$
        \mathrm{dof}(K) = (|V| - 1) + \sum_{k=1}^{K} \Big[
            \sum_{h \in \mathcal{P}_G(k)} |S(h)| \; - \; C_k \Big],
        $$

        where $\mathcal{P}_G(k)$ are the paths of $k$ nodes in $G$ and $C_k$ counts the normalisation
        constraints. Both terms are read off the same $k$-fold line-graph lift of $G$, whose edges are
        the paths of $k+1$ nodes and whose nodes are the paths of $k$ nodes.

        Info: The two conventions
            - `"per_history"` counts one constraint per history that has at least one successor. This
              is the actual number of free parameters and follows Petrović and Scholtes[^2].
            - `"per_first_order_node"` counts one constraint per first-order node from which a path of
              length $k$ starts, which is the formula printed by Scholtes[^1]. The two agree for
              $k = 1$ and differ from $k = 2$ whenever two different histories start at the same node,
              in which case the first-order variant subtracts too little.

        Warning:
            The number of paths of length $k$ in $G$ can grow quickly with $k$, and they are
            materialised one order at a time. High orders on dense graphs are expensive.

        Args:
            max_order: The maximum order to count parameters for. Defaults to the model's maximum.
            normalization_constraints: `"per_history"` or `"per_first_order_node"`.
            constraint_graph: First-order topology defining which transitions are possible. Defaults
                to the model's own order-1 layer, i.e. to the transitions the data actually takes.

        Returns:
            int: The number of free parameters.

        Raises:
            ValueError: For an unknown convention, a `max_order` above the model's, or a constraint
                graph that does not contain every observed transition.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
            >>> paths.append_walk(("a", "c", "d"))
            >>> paths.append_walk(("b", "c", "e"))
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> print([model.degrees_of_freedom(max_order=k) for k in (0, 1, 2)])
            [4, 5, 7]
        """
        if normalization_constraints not in NORMALIZATION_CONSTRAINTS:
            logger.error("unknown normalization_constraints %s", normalization_constraints)
            raise ValueError(
                f"unknown normalization_constraints {normalization_constraints!r}. "
                f"Expected one of {NORMALIZATION_CONSTRAINTS}."
            )
        if max_order is None:
            max_order = max(self.layers)
        if max_order > max(self.layers):
            logger.error("max_order %s exceeds the maximum order of the model", max_order)
            raise ValueError(
                f"max_order ({max_order}) cannot be larger than the maximum order of the "
                f"multi-order model ({max(self.layers)})"
            )

        edge_index, num_first_order_nodes = self._constraint_topology(constraint_graph)
        degrees_of_freedom = num_first_order_nodes - 1
        first_node = torch.arange(num_first_order_nodes, device=edge_index.device)

        for order in range(1, max_order + 1):
            if edge_index.size(1) == 0:
                break
            degrees_of_freedom += edge_index.size(1)
            if normalization_constraints == "per_history":
                degrees_of_freedom -= int(torch.unique(edge_index[0]).numel())
            else:
                degrees_of_freedom -= int(torch.unique(first_node[edge_index[0]]).numel())
            if order < max_order:
                next_first_node = first_node[edge_index[0]]
                edge_index = lift_order_edge_index(edge_index, num_nodes=first_node.size(0))
                first_node = next_first_node

        return int(degrees_of_freedom)

    def likelihood_ratio_test(
        self,
        null_max_order: int = 0,
        max_order: int = 1,
        significance_threshold: float = 0.01,
        normalization_constraints: str = "per_history",
        constraint_graph: Optional[Graph] = None,
    ) -> tuple[bool, float]:
        r"""Test whether a multi-order model of higher order fits the data significantly better.

        The two models are nested, so twice the log-likelihood ratio is asymptotically $\chi^2$
        distributed with as many degrees of freedom as the alternative has extra parameters:

        $$
        x = -2 \log \frac{\mathcal{L}(K_0)}{\mathcal{L}(K)}
          = -2 \big( \log \mathcal{L}(K_0) - \log \mathcal{L}(K) \big),
        \qquad p = P(\chi^2_{\xi} > x),
        $$

        with $\xi = \mathrm{dof}(K) - \mathrm{dof}(K_0)$. The null hypothesis -- that the simpler
        model suffices -- is rejected when $p$ falls below the significance threshold.

        Args:
            null_max_order: Maximum order of the null hypothesis model.
            max_order: Maximum order of the alternative hypothesis model. Must be larger.
            significance_threshold: The $p$-value below which the null hypothesis is rejected, i.e.
                the tolerated probability of wrongly preferring the higher order.
            normalization_constraints: Convention used to count parameters, see
                [`degrees_of_freedom`][pathpyG.core.multi_order_model.MultiOrderModel.degrees_of_freedom].
            constraint_graph: First-order topology defining which transitions are possible.

        Returns:
            tuple: Whether the null hypothesis is rejected, and the $p$-value of the test.

        Raises:
            ValueError: If the null order is not smaller than the alternative order, or the
                alternative order exceeds the model's maximum order.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
            >>> paths.append_walk(("a", "c", "d"), weight=4)
            >>> paths.append_walk(("b", "c", "e"), weight=4)
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> rejected, p_value = model.likelihood_ratio_test(null_max_order=1, max_order=2)
            >>> print(rejected, round(p_value, 6))
            True 0.003906
        """
        if null_max_order >= max_order:
            logger.error("null order %s is not smaller than alternative order %s", null_max_order, max_order)
            raise ValueError(
                f"the order of the null hypothesis ({null_max_order}) must be smaller than the "
                f"order of the alternative hypothesis ({max_order})"
            )
        if max_order > max(self.layers):
            logger.error("alternative order %s exceeds the maximum order of the model", max_order)
            raise ValueError(
                f"the order of the alternative hypothesis ({max_order}) cannot be larger than the "
                f"maximum order of the multi-order model ({max(self.layers)})"
            )

        statistic = -2 * (self.log_likelihood(max_order=null_max_order) - self.log_likelihood(max_order=max_order))
        delta_degrees_of_freedom = self.degrees_of_freedom(
            max_order,
            normalization_constraints=normalization_constraints,
            constraint_graph=constraint_graph,
        ) - self.degrees_of_freedom(
            null_max_order,
            normalization_constraints=normalization_constraints,
            constraint_graph=constraint_graph,
        )

        p_value = float(chi2.sf(statistic, delta_degrees_of_freedom))
        return bool(p_value < significance_threshold), p_value

    def _constraint_out_degrees(self, constraint_graph: Optional[Graph] = None) -> tuple[torch.Tensor, int]:
        """Number of admissible successors of each first-order node, and the number of nodes."""
        edge_index, num_first_order_nodes = self._constraint_topology(constraint_graph)
        out_degree = torch.bincount(edge_index[0], minlength=num_first_order_nodes)
        return out_degree, num_first_order_nodes

    @staticmethod
    def _log_dirichlet_multinomial(
        counts: torch.Tensor,
        history: torch.Tensor,
        num_histories: int,
        support_size: torch.Tensor,
        dirichlet_prior: float,
    ) -> float:
        r"""Sum of $\log B(\vec\alpha_h + \vec\mu_h) - \log B(\vec\alpha_h)$ over the histories.

        Successors that were never observed contribute nothing, because their prior and posterior
        pseudo-counts agree, so only the observed transitions have to be touched. Histories without
        any admissible successor -- dead ends of the topology -- carry no parameter at all and are
        skipped; nothing can have been observed after them.
        """
        counts = counts.double()
        concentration = dirichlet_prior * support_size.double()
        prior = torch.tensor(dirichlet_prior, dtype=torch.float64, device=counts.device)

        observed_terms = torch.lgamma(counts + prior) - torch.lgamma(prior)
        totals = scatter(counts, history, dim=0, dim_size=num_histories, reduce="sum")
        parameterized = concentration > 0
        history_terms = torch.lgamma(concentration[parameterized]) - torch.lgamma(
            concentration[parameterized] + totals[parameterized]
        )
        return float(observed_terms.sum().item() + history_terms.sum().item())

    def log_evidence(
        self,
        max_order: int = 1,
        dirichlet_prior: float = 1.0,
        constraint_graph: Optional[Graph] = None,
    ) -> float:
        r"""Log marginal likelihood (model evidence) of the observed paths.

        Rather than fitting the transition probabilities and then penalising their number, the
        Bayesian view integrates them out under a Dirichlet prior. With a flat prior of concentration
        $\alpha$ on each of the $|S(h)|$ successors a history $h$ may take, and $\vec\mu_h$ the
        transitions observed after $h$, conjugacy gives a closed form:

        $$
        \log p(D \mid \alpha, K, G) = \sum_{k=0}^{K} \sum_{h \in \mathcal{P}_G(k)}
            \Big[ \log B(\vec\alpha_h + \vec\mu_h) - \log B(\vec\alpha_h) \Big],
        $$

        where $\log B(\vec x) = \sum_i \log\Gamma(x_i) - \log\Gamma\big(\sum_i x_i\big)$. Histories
        that were never observed contribute exactly zero, so only the nodes the model actually has
        need to be touched. The zeroth-order term is the single empty history, whose successors are
        all first-order nodes.

        The constant counting the permutations of the observed paths is the same for every order and
        is omitted, so this value is a log evidence up to an additive constant. Differences between
        orders -- that is, the Bayes factors that model selection uses -- are exact.

        ??? reference
            This is the model evidence of Petrović and Scholtes[^2], which selects the Markov order
            more reliably than the likelihood ratio test and needs considerably less data.

        Args:
            max_order: The maximum order $K$ of the multi-order model.
            dirichlet_prior: The concentration $\alpha$ of the flat Dirichlet prior placed on the
                admissible successors of every history. One corresponds to a uniform prior over the
                probability simplex.
            constraint_graph: First-order topology defining which successors are admissible. Defaults
                to the model's own order-1 layer, i.e. to the transitions the data actually takes.

        Returns:
            float: The log evidence, up to an order-independent additive constant.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
            >>> paths.append_walk(("a", "c", "d"), weight=4)
            >>> paths.append_walk(("b", "c", "e"), weight=4)
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> print(model.log_evidence(max_order=2) > model.log_evidence(max_order=1))
            True
        """
        if max_order < 0:
            logger.error("max_order must not be negative, got %s", max_order)
            raise ValueError(f"max_order must not be negative, got {max_order}")
        if dirichlet_prior <= 0:
            logger.error("dirichlet_prior must be positive, got %s", dirichlet_prior)
            raise ValueError(f"dirichlet_prior must be positive, got {dirichlet_prior}")

        out_degree, num_first_order_nodes = self._constraint_out_degrees(constraint_graph)
        device = out_degree.device

        # Order zero: a single empty history whose successors are all first-order nodes.
        zeroth_order_counts = self._require_layer(1).data.node_weight if max_order == 0 else self._start_weight(1)
        total = MultiOrderModel._log_dirichlet_multinomial(
            counts=zeroth_order_counts,
            history=torch.zeros(zeroth_order_counts.size(0), dtype=torch.long, device=device),
            num_histories=1,
            support_size=torch.tensor([num_first_order_nodes], device=device),
            dirichlet_prior=dirichlet_prior,
        )

        for order in range(1, max_order + 1):
            layer = self._require_layer(order)
            counts = self._start_weight_on_edges(order) if order < max_order else layer.data.edge_weight
            total += MultiOrderModel._log_dirichlet_multinomial(
                counts=counts,
                history=_as_tensor(layer.data.edge_index)[0],
                num_histories=layer.n,
                support_size=out_degree[layer.data.node_sequence[:, -1]],
                dirichlet_prior=dirichlet_prior,
            )

        return total

    def bayes_factor(
        self,
        null_max_order: int = 0,
        max_order: int = 1,
        dirichlet_prior: float = 1.0,
        constraint_graph: Optional[Graph] = None,
    ) -> float:
        r"""Log Bayes factor in favour of the higher-order model.

        The difference of two log evidences,
        $\log \frac{p(D \mid K)}{p(D \mid K_0)}$, in which the omitted permutation constant cancels.
        A positive value favours `max_order`, and the usual reading is that values above about 1 are
        substantial and above about 5 are strong evidence.

        Args:
            null_max_order: Maximum order of the simpler model.
            max_order: Maximum order of the richer model.
            dirichlet_prior: Concentration of the flat Dirichlet prior.
            constraint_graph: First-order topology defining which successors are admissible.

        Returns:
            float: The log Bayes factor.
        """
        return self.log_evidence(
            max_order, dirichlet_prior=dirichlet_prior, constraint_graph=constraint_graph
        ) - self.log_evidence(null_max_order, dirichlet_prior=dirichlet_prior, constraint_graph=constraint_graph)

    def estimate_order(
        self,
        max_order: Optional[int] = None,
        method: str = "likelihood_ratio",
        order_selection: Optional[str] = None,
        significance_threshold: Optional[float] = None,
        normalization_constraints: Optional[str] = None,
        dirichlet_prior: Optional[float] = None,
        log_bayes_factor_threshold: Optional[float] = None,
        constraint_graph: Optional[Graph] = None,
    ) -> int:
        """Estimate the optimal maximum order of the multi-order model for the observed data.

        Orders are compared pairwise from order one upwards, so order zero -- no structure beyond the
        relative frequency of the nodes -- can be selected as well.

        Info: How an order is accepted
            With `method="likelihood_ratio"` an order is accepted when a
            [likelihood ratio test][pathpyG.core.multi_order_model.MultiOrderModel.likelihood_ratio_test] against the
            comparison order rejects the simpler model at `significance_threshold`. With
            `method="bayesian"` it is accepted when the
            [log Bayes factor][pathpyG.core.multi_order_model.MultiOrderModel.bayes_factor] exceeds
            `log_bayes_factor_threshold`.

        Info: Which order is returned
            - `"largest_significant"`: the largest order that beats the order just below it. This is
              the rule of Scholtes[^1] and the default for the likelihood ratio test.
            - `"dominates_all_lower_orders"`: the largest order that beats *every* lower order. This
              is the rule of Petrović and Scholtes[^2] and the default for the Bayesian selector.
            - `"stop_at_first_insignificant"`: stop at the first order that fails, and return the
              last accepted one. The most conservative rule.

        Args:
            max_order: The highest order to consider. Defaults to the model's maximum order.
            method: `"likelihood_ratio"` or `"bayesian"`.
            order_selection: Which order to return, see above. Defaults to the rule of the paper the
                chosen method comes from.
            significance_threshold: $p$-value threshold of the likelihood ratio test. Defaults to
                `0.01`. Only valid for `method="likelihood_ratio"`.
            normalization_constraints: Convention used to count parameters, see
                [`degrees_of_freedom`][pathpyG.core.multi_order_model.MultiOrderModel.degrees_of_freedom]. Defaults to
                `"per_history"`. Only valid for `method="likelihood_ratio"`.
            dirichlet_prior: Concentration of the flat Dirichlet prior. Defaults to `1.0`. Only valid
                for `method="bayesian"`.
            log_bayes_factor_threshold: How much larger a log evidence has to be for the higher order
                to be accepted. Defaults to `0.0`. Only valid for `method="bayesian"`.
            constraint_graph: First-order topology defining which transitions are possible. Defaults
                to the model's own order-1 layer.

        Returns:
            int: The estimated optimal maximum order, between zero and `max_order`.

        Raises:
            ValueError: For an unknown method or selection rule, a `max_order` above the model's, or
                an argument that does not belong to the chosen method.

        Examples:
            >>> import pathpyG as pp
            >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
            >>> paths.append_walk(("a", "c", "d"), weight=4)
            >>> paths.append_walk(("b", "c", "e"), weight=4)
            >>> model = pp.MultiOrderModel.from_path_data(paths, max_order=2)
            >>> print(model.estimate_order(), model.estimate_order(method="bayesian"))
            2 2
        """
        if method not in SELECTION_METHODS:
            logger.error("unknown method %s", method)
            raise ValueError(f"unknown method {method!r}. Expected one of {SELECTION_METHODS}.")

        frequentist_arguments = {
            "significance_threshold": significance_threshold,
            "normalization_constraints": normalization_constraints,
        }
        bayesian_arguments = {
            "dirichlet_prior": dirichlet_prior,
            "log_bayes_factor_threshold": log_bayes_factor_threshold,
        }
        wrong_method = bayesian_arguments if method == "likelihood_ratio" else frequentist_arguments
        misplaced = sorted(name for name, value in wrong_method.items() if value is not None)
        if misplaced:
            logger.error("arguments %s do not apply to method %s", misplaced, method)
            raise ValueError(f"the argument(s) {', '.join(misplaced)} do not apply to method={method!r}.")

        if order_selection is None:
            order_selection = DEFAULT_ORDER_SELECTION[method]
        if order_selection not in ORDER_SELECTION_RULES:
            logger.error("unknown order_selection %s", order_selection)
            raise ValueError(f"unknown order_selection {order_selection!r}. Expected one of {ORDER_SELECTION_RULES}.")

        if max_order is None:
            max_order = max(self.layers)
        if max_order > max(self.layers):
            logger.error("max_order %s exceeds the maximum order of the model", max_order)
            raise ValueError(
                f"max_order ({max_order}) cannot be larger than the maximum order of the "
                f"multi-order model ({max(self.layers)})"
            )
        if max_order < 1:
            logger.error("max_order must be at least one, got %s", max_order)
            raise ValueError(f"max_order must be at least one, got {max_order}")

        if method == "likelihood_ratio":
            threshold = 0.01 if significance_threshold is None else significance_threshold
            constraints = "per_history" if normalization_constraints is None else normalization_constraints

            def beats(simpler_order: int, richer_order: int) -> bool:
                return self.likelihood_ratio_test(
                    null_max_order=simpler_order,
                    max_order=richer_order,
                    significance_threshold=threshold,
                    normalization_constraints=constraints,
                    constraint_graph=constraint_graph,
                )[0]

        else:
            prior = 1.0 if dirichlet_prior is None else dirichlet_prior
            factor_threshold = 0.0 if log_bayes_factor_threshold is None else log_bayes_factor_threshold
            evidences = [
                self.log_evidence(order, dirichlet_prior=prior, constraint_graph=constraint_graph)
                for order in range(max_order + 1)
            ]

            def beats(simpler_order: int, richer_order: int) -> bool:
                return evidences[richer_order] - evidences[simpler_order] > factor_threshold

        estimated_order = 0
        for order in range(1, max_order + 1):
            if order_selection == "dominates_all_lower_orders":
                accepted = all(beats(lower_order, order) for lower_order in range(order))
            else:
                accepted = beats(order - 1, order)
            if accepted:
                estimated_order = order
            elif order_selection == "stop_at_first_insignificant":
                break

        return estimated_order

    # ------------------------------------------------------------------ conversion

    def to_dbgnn_data(self, max_order: int = 2, mapping: str = "last") -> Data:
        """Convert the MultiOrderModel to a De Bruijn graph for the given maximum order that can be used in the [DBGNN][pathpyG.nn.dbgnn.DBGNN]-model.

        The message-passing weights are the number of distinct walks each edge represents, i.e.
        `edge_num_walks` where a temporal model provides it and `edge_weight` otherwise.

        Args:
            max_order: The maximum order of the De Bruijn graph to be computed.
            mapping: The mapping to use for the bipartite edge index. One of "last", "first", or "both".

        Returns:
            Data: The De Bruijn graph data.
        """
        if max_order not in self.layers:
            logger.error("higher-order graph of order %s not found", max_order)
            raise ValueError(f"Higher-order graph of order {max_order} not found.")

        def message_passing_weight(layer: HigherOrderGraph) -> torch.Tensor:
            return layer.data.edge_num_walks if "edge_num_walks" in layer.data else layer.data.edge_weight

        first_order = self.layers[1]
        highest_order = self.layers[max_order]
        num_nodes = first_order.data.num_nodes
        num_higher_order_nodes = highest_order.data.num_nodes
        if first_order.data.x is not None:
            node_features = first_order.data.x
        else:
            node_features = torch.eye(num_nodes, num_nodes, device=first_order.data.edge_index.device)
        higher_order_features = torch.eye(
            num_higher_order_nodes, num_higher_order_nodes, device=highest_order.data.edge_index.device
        )

        return Data(
            num_nodes=num_nodes,
            num_ho_nodes=num_higher_order_nodes,
            x=node_features,
            x_h=higher_order_features,
            edge_index=first_order.data.edge_index,
            edge_index_higher_order=highest_order.data.edge_index,
            edge_weights=message_passing_weight(first_order).float(),
            edge_weights_higher_order=message_passing_weight(highest_order).float(),
            bipartite_edge_index=highest_order.bipartite_edge_index(
                first_order, mapping=mapping, device=first_order.data.edge_index.device
            ),
            y=first_order.data.y if first_order.data.y is not None else None,
        )
