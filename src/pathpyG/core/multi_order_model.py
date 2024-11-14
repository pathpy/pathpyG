from __future__ import annotations

from scipy.stats import chi2
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, cumsum

from pathpyG.core.graph import Graph
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.utils.dbgnn import generate_bipartite_edge_index
from pathpyG.algorithms.temporal import lift_order_temporal
from pathpyG.algorithms.lift_order import (
    aggregate_node_attributes,
    lift_order_edge_index,
    lift_order_edge_index_weighted,
    aggregate_edge_index,
)


class MultiOrderModel:
    """MultiOrderModel based on torch_geometric.Data."""

    def __init__(self) -> None:
        self.layers: dict[int, Graph] = {}

    def __str__(self) -> str:
        """Return a string representation of the higher-order graph."""
        max_order = max(list(self.layers.keys())) if self.layers else 0
        s = f"MultiOrderModel with max. order {max_order}"
        return s

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
    def from_temporal_graph(
        g: TemporalGraph, delta: float | int = 1, max_order: int = 1, weight: str = "edge_weight", cached: bool = True
    ) -> MultiOrderModel:
        """Creates multiple higher-order De Bruijn graph models for paths in a temporal graph.

        Args:
            g: The temporal graph.
            delta: The maximum time difference between two consecutive edges in a path.
            max_order: The maximum order of the MultiOrderModel that should be computed.
            weight: The edge attribute to use as edge weight.
            cached: Whether to save the aggregated higher-order graphs smaller than max order in the MultiOrderModel.

        Returns:
            MultiOrderModel: The MultiOrderModel.
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
            edge_index = lift_order_temporal(g, delta)
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
                    m.layers[k] = gk

        return m

    @staticmethod
    def from_PathData(
        path_data: PathData, max_order: int = 1, mode: str = "propagation", cached: bool = True
    ) -> MultiOrderModel:
        """
        Creates multiple higher-order De Bruijn graphs modelling paths in PathData.

        Args:
            path_data: `PathData` object containing paths as list of PyG Data objects
                with sorted edge indices, node sequences and num_nodes.
            max_order: The maximum order of the MultiOrderModel that should be computed
            mode: The process that we assume. Can be "diffusion" or "propagation".
            cached: Whether to save the aggregated higher-order graphs smaller than max order
                in the MultiOrderModel.

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
                m.layers[k] = gk

        return m

    def get_mon_dof(self, max_order: int = None, assumption: str = "paths") -> int:
        """
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
            AssertionError: If max_order is larger than the maximum order of
                the multi-order network.
            ValueError: If the assumption is not 'paths' or 'ngrams'.
        """
        if max_order is None:
            max_order = max(self.layers)

        assert max_order <= max(
            self.layers
        ), "Error: max_order cannot be larger than maximum order of multi-order network"

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
            raise ValueError(
                f"Unknown assumption {assumption} in input. The only accepted values are 'path' and 'ngram'"
            )

        return int(dof)

    def get_zeroth_order_log_likelihood(self, dag_graph: Data) -> float:
        """
        Compute the zeroth order log likelihood.

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
        mask = torch.ones(dag_graph.num_nodes, dtype=bool)
        mask[dag_graph.edge_index[1]] = False
        start_ixs = dag_graph.node_sequence.squeeze()[mask]

        # Compute node emission probabilities
        # TODOL modify once we have zeroth order in mon
        _, counts = torch.unique(dag_graph.node_sequence, return_counts=True)
        # WARNING: Only works if all nodes in the first-order graph are also in `node_sequence`
        # Otherwise the missing nodes will not be included in `counts` which can lead to elements at the wrong index.
        node_emission_probabilities = counts / counts.sum()
        return torch.mul(frequencies, torch.log(node_emission_probabilities[start_ixs])).sum().item()

    def get_intermediate_order_log_likelihood(self, dag_graph: Data, order: int) -> float:
        """
        Compute the intermediate order log likelihood.

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
        """
        Compute the likelihood of the walks given a multi-order model.

        Args:
            m (MultiOrderModel): The multi-order model.
            dag_graph (Data): Dataset containing the walks.
            max_order (int, optional): The maximum order up to which model layers
                shall be taken into account. Defaults to 1.

        Returns:
            float: The log likelihood of the walks given the multi-order model.
        """
        llh = 0

        # Adding likelihood of zeroth order
        llh += self.get_zeroth_order_log_likelihood(dag_graph)

        # Adding the likelihood for all the intermediate orders
        for order in range(1, max_order):
            llh += self.get_intermediate_order_log_likelihood(dag_graph, order)

        # Adding the likelihood of highest/stationary order
        if max_order > 0:
            transition_probabilities = self.layers[max_order].transition_probabilities()
            log_transition_probabilities = torch.log(transition_probabilities)
            llh_by_subpath = log_transition_probabilities * self.layers[max_order].data.edge_weight
            llh += llh_by_subpath.sum().item()
        else:
            # Compute likelihood for zeroth order (to be modified)
            # TODO: modify once we have zeroth order in mon
            # (then won t need to compute emission probs from dag_graph -- which also hinders us from computing the lh that a new set of paths was generated by the model)
            frequencies = dag_graph.dag_weight
            counts = torch.bincount(
                dag_graph.node_sequence.squeeze(), frequencies.repeat_interleave(dag_graph.dag_num_nodes)
            )
            node_emission_probabilities = counts / counts.sum()
            llh = torch.mul(torch.log(node_emission_probabilities), counts).sum().item()

        return llh

    def likelihood_ratio_test(
        self,
        dag_graph: Data,
        max_order_null: int = 0,
        max_order: int = 1,
        assumption: str = "paths",
        significance_threshold: float = 0.01,
    ) -> tuple:
        """
        Perform a likelihood ratio test to compare two models of different order.
        
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
        assert (
            max_order_null < max_order
        ), "Error: order of null hypothesis must be smaller than order of alternative hypothesis"
        assert max_order <= max(
            self.layers
        ), f"Error: order of hypotheses ({max_order_null} and {max_order}) must be smaller than the maximum order of the MultiOrderModel {max(self.layers)}"
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

    def estimate_order(self, dag_data: PathData, max_order: int = None, significance_threshold: float = 0.01) -> int:
        """
        Selects the optimal maximum order of a multi-order network model for the
        observed paths, based on a likelihood ratio test with p-value threshold of p
        By default, all orders up to the maximum order of the multi-order model will be tested.

        Args:
            dag_data (DAGData): The path statistics data for which to estimate the optimal order.
            max_order (int, optional): The maximum order to consider during the estimation process.
                If not provided, the maximum order of the multi-order model is used.
            significance_threshold (float, optional): The p-value threshold for the likelihood ratio test.
                An order is accepted if the improvement in likelihood is significant at this threshold.

        Returns:
            int: The estimated optimal maximum order for the multi-order network model.

        Raises:
            AssertionError: If the provided max_order is larger than the maximum order of the multi-order model
                or if the input DAGData does not have the same set of nodes as the multi-order network
        """
        if max_order is None:
            max_order = max(self.layers)  # THIS
        assert max_order <= max(
            self.layers
        ), "Error: maxOrder cannot be larger than maximum order of multi-order network"
        assert max_order > 1, "Error: max_order must be larger than one"

        assert set(dag_data.mapping.node_ids).intersection(set(self.layers[1].mapping.node_ids)) == set(
            dag_data.mapping.node_ids
        ), "Input DAGData doesn t have the same set of nodes as those of the multi-order network"

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
        """
        Convert the MultiOrderModel to a De Bruijn graph for the given maximum order
        that can be used in `pathpyG.nn.dbgnn.DBGNN`.

        Args:
            max_order: The maximum order of the De Bruijn graph to be computed.
            mapping: The mapping to use for the bipartite edge index. One of "last", "first", or "both".

        Returns:
            Data: The De Bruijn graph data.
        """
        if max_order not in self.layers:
            raise ValueError(f"Higher-order graph of order {max_order} not found.")

        g = self.layers[1]
        g_max_order = self.layers[max_order]
        num_nodes = g.data.num_nodes
        num_ho_nodes = g_max_order.data.num_nodes
        if g.data.x is not None:
            x = g.data.x
        else:
            x = torch.eye(num_nodes, num_nodes)
        x_max_order = torch.eye(num_ho_nodes, num_ho_nodes)
        edge_index = g.data.edge_index
        edge_index_max_order = g_max_order.data.edge_index
        edge_weight = g.data.edge_weight
        edge_weight_max_order = g_max_order.data.edge_weight
        bipartite_edge_index = generate_bipartite_edge_index(g, g_max_order, mapping=mapping)

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
