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
    aggregate_node_attributes,
    lift_node_sequence,
    lift_order_edge_index,
    lift_order_step,
)
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.higher_order_graph import HigherOrderGraph
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")


class MultiOrderModel:
    """MultiOrderModel based on [torch_geometric.data.Data][].

    This class stores multiple higher-order De Bruijn graphs as layers in a dictionary.
    Each layer corresponds to a De Bruijn graph of order k, where k is the key in the dictionary.
    Each graph layer is represented as a
    [HigherOrderGraph][pathpyG.core.higher_order_graph.HigherOrderGraph] object, layer 1
    included. Each layer therefore knows its own order and the first-order nodes it was
    built from. Models built from path data also hold the memoryless order-0 layer, which
    is used by the likelihood computations.
    This class provides methods to search for the optimal order of the model based on likelihood ratio tests,
    as well as methods to compute the log-likelihood of observed paths given the model.

    Attributes:
        layers (dict[int, HigherOrderGraph]): A dictionary mapping the order k to the
            corresponding higher-order De Bruijn graph of order k.

    Examples:
        Example where the optimal order is 1:
        >>> import pathpyG as pp
        >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=3)
        >>> paths.append_walk(("b", "c", "e"), weight=3)
        >>> m = pp.MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(m.estimate_order(paths, max_order=2))
        1

        Example where the optimal order is 2:
        >>> paths = pp.PathData(pp.IndexMap(list("abcde")))
        >>> paths.append_walk(("a", "c", "d"), weight=4)
        >>> paths.append_walk(("b", "c", "e"), weight=4)
        >>> m = pp.MultiOrderModel.from_path_data(paths, max_order=2)
        >>> print(m.estimate_order(paths, max_order=2))
        2

        Each layer knows the first-order path that each of its nodes represents:
        >>> print(m.layers[2].order, m.layers[2].nodes)
        2 [('a', 'c'), ('b', 'c'), ('c', 'd'), ('c', 'e')]
    """

    def __init__(self) -> None:
        """Initialize an empty MultiOrderModel."""
        self.layers: dict[int, HigherOrderGraph] = {}

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

        Raises:
            ValueError: If `max_order` is smaller than 1. The order-0 layer is only defined for path data.
        """
        if max_order < 1:
            logger.error("max_order must be at least 1 for a temporal graph, got %s", max_order)
            raise ValueError(
                f"max_order must be at least 1 for a temporal graph, got {max_order}; "
                "the order-0 layer is only defined for path data"
            )
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

        # Each iteration lifts the *unaggregated* order-(k-1) data to order k and aggregates it
        # only if the layer is kept. The aggregated layers are never lifted themselves.
        for k in range(1, max_order + 1):
            if k == 2:
                # The first lift is temporal: an edge may only be continued within `delta`.
                node_sequence = lift_node_sequence(edge_index, node_sequence)
                edge_index = EventGraph.build_edge_index(g, delta) if event_graph is None else event_graph
                edge_weight = aggregate_node_attributes(edge_index, edge_weight, "src")
            elif k > 2:
                edge_index, node_sequence, edge_weight = lift_order_step(
                    edge_index, node_sequence, edge_weight=edge_weight, aggr="src"
                )

            if cached or k == max_order:
                m.layers[k] = HigherOrderGraph.aggregate(
                    edge_index=edge_index,
                    node_sequence=node_sequence,
                    edge_weight=edge_weight,
                    first_order_mapping=g.mapping,
                    n_first_order=g.n,
                )

        return m

    @classmethod
    def from_event_graph(
        cls,
        eg: EventGraph,
        max_order: int = 2,
        cached: bool = True,
    ) -> "MultiOrderModel":
        """Create a multi-order model from a pre-built event graph.

        Args:
            eg: The second-order temporal `EventGraph` to build the model from.
            max_order: The maximum order of the model to compute.
            cached: Whether to also keep the aggregated layers below `max_order`.

        Returns:
            MultiOrderModel: A multi-order model equivalent to
            `MultiOrderModel.from_temporal_graph(eg.to_temporal_graph(), delta=eg.delta, ...)`.
        """
        m = cls()
        m.layers = MultiOrderModel.from_temporal_graph(
            eg.to_temporal_graph(),
            delta=eg.delta,
            max_order=max_order,
            cached=cached,
            event_graph=eg.data.edge_index.as_tensor(),
        ).layers
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
                in the [MultiOrderModel][pathpyG.MultiOrderModel]. The layers of order 0 and 1
                are always kept, since the likelihood computations rely on them.

        Returns:
            MultiOrderModel: The MultiOrderModel.
        """
        if max_order < 0:
            logger.error("max_order must be at least 0, got %s", max_order)
            raise ValueError(f"max_order must be at least 0, got {max_order}")
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

        # First-order nodes that are not visited by any path are still part of the model.
        if path_data.mapping.has_ids:
            n_first_order = path_data.mapping.num_ids()
        else:
            n_first_order = int(node_sequence.max().item()) + 1 if node_sequence.numel() > 0 else 0

        # Order 0: every visit of a node, weighted by the frequency of its path.
        visit_weight = path_graph.dag_weight.repeat_interleave(path_graph.dag_num_nodes)
        node_weight = torch.zeros(n_first_order, dtype=visit_weight.dtype, device=visit_weight.device)
        node_weight.scatter_add_(0, node_sequence.squeeze(1), visit_weight)
        m.layers[0] = HigherOrderGraph.from_node_weights(node_weight, first_order_mapping=path_data.mapping)
        if max_order == 0:
            return m

        # The paths themselves are the unaggregated order-1 data (one node per visit). Each
        # iteration lifts the unaggregated order-(k-1) data to order k and aggregates it only
        # if the layer is kept. The aggregated layers are never lifted themselves.
        for k in range(1, max_order + 1):
            if k > 1:
                edge_index, node_sequence, edge_weight = lift_order_step(
                    edge_index, node_sequence, edge_weight=edge_weight, aggr=aggr
                )

            if k == 1 or cached or k == max_order:
                m.layers[k] = HigherOrderGraph.aggregate(
                    edge_index=edge_index,
                    node_sequence=node_sequence,
                    first_order_mapping=path_data.mapping,
                    edge_weight=edge_weight,
                    n_first_order=n_first_order,
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

        # Degrees of freedom for zeroth order: one probability per first-order node, minus normalisation
        if 0 in self.layers:
            dof = self.layers[0].m - 1
        else:
            dof = self.layers[1].n_first_order - 1

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

    def get_zeroth_order_log_likelihood(self, dag_graph: Data) -> float:
        """Compute the zeroth order log likelihood.

        Args:
            dag_graph (Data): Input DAG graph data.

        Returns:
            float: Zeroth order log likelihood.
        """
        # Get frequencies
        # getting the index of the last edge of each path (to be used to extract weights)
        frequencies = dag_graph.dag_weight

        # Get ixs starting nodes
        # Q: Is dag_graph.path_index[:-1] enough to get the start_ixs?
        mask = torch.ones(dag_graph.num_nodes, dtype=bool)  # type: ignore[call-overload]
        mask[dag_graph.edge_index[1]] = False
        start_ixs = dag_graph.node_sequence.squeeze(1)[mask]

        node_visit_probabilities = self._node_visit_probabilities()
        return torch.mul(frequencies, torch.log(node_visit_probabilities[start_ixs])).sum().item()

    def _zeroth_order_layer(self) -> HigherOrderGraph:
        """Return the order-0 layer.

        Raises:
            ValueError: If the model has no order-0 layer, i.e. was not built from path data.
        """
        if 0 not in self.layers:
            logger.error("MultiOrderModel has no order-0 layer")
            raise ValueError("the likelihood requires the order-0 layer, which is only built from path data")
        return self.layers[0]

    def _node_visit_probabilities(self) -> torch.Tensor:
        """Return the visitation probability of each first-order node under the order-0 layer.

        Returns:
            torch.Tensor: Tensor of shape `(n_first_order,)` indexed by first-order node index.
        """
        g0 = self._zeroth_order_layer()
        probabilities = torch.zeros(g0.n_first_order, device=g0.device)
        probabilities[g0.data.edge_first_order_node] = g0.transition_probabilities(edge_attr="edge_weight").float()
        return probabilities

    def get_intermediate_order_log_likelihood(self, dag_graph: Data, order: int) -> float:
        """Compute the intermediate order log likelihood.

        Args:
            m (MultiOrderModel): Multi-order model.
            dag_graph (Data): Input DAG graph data.
            order (int): Order of the intermediate log likelihood.

        Returns:
            float: Intermediate order log likelihood.
        """
        # Get frequencies
        frequencies = dag_graph.dag_weight
        path_lengths = dag_graph.dag_num_nodes
        # paths shrink by 'order' if we encode them using higher-order nodes
        paths_lenghts_ho = path_lengths - order
        # selecting only path that didn t shrink to zero due to higher-order transformation
        paths_lenghts_ho_filtered = paths_lenghts_ho[paths_lenghts_ho > 0]
        frequencies = frequencies[paths_lenghts_ho > 0]
        # start index of the path in the higher order space
        ixs_start_paths_ho = cumsum(paths_lenghts_ho_filtered)[:-1]

        transition_probabilities = self.layers[order].transition_probabilities()[
            self.layers[order + 1].data.inverse_idx[ixs_start_paths_ho]
        ]

        log_transition_probabilities = torch.log(transition_probabilities)
        llh_by_subpath = torch.mul(frequencies, log_transition_probabilities)
        return llh_by_subpath.sum().item()

    def get_mon_log_likelihood(self, dag_graph: Data, max_order: int = 1) -> float:
        """Compute the likelihood of the walks given a multi-order model.

        Args:
            m (MultiOrderModel): The multi-order model.
            dag_graph (Data): Dataset containing the walks.
            max_order (int, optional): The maximum order up to which model layers
                shall be taken into account. Defaults to 1.

        Returns:
            float: The log likelihood of the walks given the multi-order model.
        """
        if max_order == 0:
            # Under the memoryless model every visit, including the first of each path, is an
            # independent draw from the order-0 layer, whose loop weights count the visits.
            g0 = self._zeroth_order_layer()
            visit_probabilities = g0.transition_probabilities(edge_attr="edge_weight")
            return torch.xlogy(g0.data.edge_weight, visit_probabilities).sum().item()

        llh = 0.0

        # Adding likelihood of zeroth order
        llh += self.get_zeroth_order_log_likelihood(dag_graph)

        # Adding the likelihood for all the intermediate orders
        for order in range(1, max_order):
            llh += self.get_intermediate_order_log_likelihood(dag_graph, order)

        # Adding the likelihood of highest/stationary order
        transition_probabilities = self.layers[max_order].transition_probabilities(edge_attr="edge_weight")
        log_transition_probabilities = torch.log(transition_probabilities)
        llh_by_subpath = log_transition_probabilities * self.layers[max_order].data.edge_weight
        llh += llh_by_subpath.sum().item()

        return llh

    def likelihood_ratio_test(
        self,
        dag_graph: Data,
        max_order_null: int = 0,
        max_order: int = 1,
        assumption: str = "paths",
        significance_threshold: float = 0.01,
    ) -> tuple:
        """Perform a likelihood ratio test to compare two models of different order.

        Args:
            dag_graph (Data): The input DAG graph data.
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
            self.get_mon_log_likelihood(dag_graph, max_order=max_order_null)
            - self.get_mon_log_likelihood(dag_graph, max_order=max_order)
        )

        # we calculate the additional degrees of freedom in the alternative model
        dof_diff = self.get_mon_dof(max_order, assumption=assumption) - self.get_mon_dof(
            max_order_null, assumption=assumption
        )

        # if the p-value is *below* the significance threshold, we reject the null hypothesis
        p = 1 - chi2.cdf(x, dof_diff)
        return (p < significance_threshold), p

    def estimate_order(
        self, dag_data: PathData, max_order: Optional[int] = None, significance_threshold: float = 0.01
    ) -> int:
        """Estimate the optimal maximum order of the multi-order network model.

        Selects the optimal maximum order of a multi-order network model for the
        observed paths, based on a likelihood ratio test with p-value threshold of p
        By default, all orders up to the maximum order of the multi-order model will be tested.

        Args:
            dag_data: The path statistics data for which to estimate the optimal order.
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
        if set(dag_data.mapping.node_ids).intersection(set(self.layers[1].mapping.node_ids)) != set(  # type: ignore[arg-type]
            dag_data.mapping.node_ids  # type: ignore[arg-type]
        ):
            logger.error("Input paths do not have same set of nodes as multi-order network")
            raise ValueError("Input paths do not have same set of nodes as multi-order network")

        max_accepted_order = 1
        dag_graph = dag_data.data

        # Test for highest order that passes
        # likelihood ratio test against null model
        for k in range(2, max_order + 1):
            if self.likelihood_ratio_test(
                dag_graph, max_order_null=k - 1, max_order=k, significance_threshold=significance_threshold
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
        if max_order < 1:
            logger.error("max_order must be at least 1 for the DBGNN, got %s", max_order)
            raise ValueError(f"max_order must be at least 1 for the DBGNN, got {max_order}")
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
        bipartite_edge_index = g_max_order.bipartite_edge_index(g, mapping=mapping, device=edge_index.device)

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
