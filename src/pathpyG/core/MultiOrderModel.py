from __future__ import annotations

from scipy.stats import chi2
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import cumsum, coalesce, degree, sort_edge_index, scatter

from pathpyG.utils.config import config
from pathpyG.core.Graph import Graph
from pathpyG.core.DAGData import DAGData
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.IndexMap import IndexMap


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
    def aggregate_edge_weight(ho_index: torch.Tensor, edge_weight: torch.Tensor, aggr: str = "src") -> torch.Tensor:
        """
        Aggregate edge weights of a (k-1)-th order graph for a kth-order graph.

        Args:
            ho_index: The higher-order edge index of the higher-order graph.
            edge_weight: The edge weights of the (k-1)th order graph.
            aggr: The aggregation method to use. One of "src", "dst", "max", "mul".
        """
        if aggr == "src":
            ho_edge_weight = edge_weight[ho_index[0]]
        elif aggr == "dst":
            ho_edge_weight = edge_weight[ho_index[1]]
        elif aggr == "max":
            ho_edge_weight = torch.maximum(edge_weight[ho_index[0]], edge_weight[ho_index[1]])
        elif aggr == "mul":
            ho_edge_weight = edge_weight[ho_index[0]] * edge_weight[ho_index[1]]
        else:
            raise ValueError(f"Unknown aggregation method {aggr}")
        return ho_edge_weight

    @staticmethod
    def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Do a line graph transformation on the edge index to lift the order of the graph by one.
        Assumes that the edge index is sorted.

        Args:
            edge_index: A **sorted** edge index tensor of shape (2, num_edges).
            num_nodes: The number of nodes in the graph.
        """
        outdegree = degree(edge_index[0], dtype=torch.long, num_nodes=num_nodes)
        # Map outdegree to each destination node to create an edge for each combination
        # of incoming and outgoing edges for each destination node
        outdegree_per_dst = outdegree[edge_index[1]]
        num_new_edges = outdegree_per_dst.sum()
        # Create sources of the new higher-order edges
        ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)

        # Create destination nodes that start the indexing after the cumulative sum of the outdegree
        # of all previous nodes in the ordered sequence of nodes
        ptrs = cumsum(outdegree, dim=0)[:-1]
        ho_edge_dsts = torch.repeat_interleave(ptrs[edge_index[1]], outdegree_per_dst)
        idx_correction = torch.arange(num_new_edges, dtype=torch.long, device=edge_index.device)
        idx_correction -= cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
        ho_edge_dsts += idx_correction
        return torch.stack([ho_edge_srcs, ho_edge_dsts], dim=0)

    @staticmethod
    def lift_order_edge_index_weighted(
        edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, aggr: str = "src"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Do a line graph transformation on the edge index to lift the order of the graph by one.
        Additionally, aggregate the edge weights of the (k-1)-th order graph to the (k)-th order graph.
        Assumes that the edge index is sorted.

        Args:
            edge_index: A **sorted** edge index tensor of shape (2, num_edges).
            edge_weight: The edge weights of the (k-1)th order graph.
            num_nodes: The number of nodes in the graph.
            aggr: The aggregation method to use. One of "src", "dst", "max", "mul".
        """
        ho_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes)
        ho_edge_weight = MultiOrderModel.aggregate_edge_weight(ho_index, edge_weight, aggr)

        return ho_index, ho_edge_weight

    @staticmethod
    def aggregate_edge_index(
        edge_index: torch.Tensor, node_sequence: torch.Tensor, edge_weight: torch.Tensor | None = None
    ) -> Graph:
        """
        Aggregate the possibly duplicated edges in the (higher-order) edge index and return a graph object
        containing the (higher-order) edge index without duplicates and the node sequences.
        The edge weights of duplicated edges are summed up.

        Args:
            edge_index: The edge index of a (higher-order) graph where each source and destination node
                corresponds to a node which is an edge in the (k-1)-th order graph.
            node_sequence: The node sequences of first order nodes that each node in the edge index corresponds to.
            edge_weight: The edge weights corresponding to the edge index.
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # If first order, then the indices in the node sequence are the inverse idx we would need already
        if node_sequence.size(1) == 1:
            unique_nodes = torch.arange(node_sequence.max().item() + 1, device=node_sequence.device).unsqueeze(1)
            mapped_edge_index = node_sequence.squeeze()[edge_index]
            unique_nodes, inverse_idx = torch.unique(node_sequence, dim=0, return_inverse=True)
        else:
            unique_nodes, inverse_idx = torch.unique(node_sequence, dim=0, return_inverse=True)
            mapped_edge_index = inverse_idx[edge_index]
        aggregated_edge_index, edge_weight = coalesce(
            mapped_edge_index,
            edge_attr=edge_weight,
            num_nodes=unique_nodes.size(0),
            reduce="sum",
        )
        data = Data(
            edge_index=aggregated_edge_index,
            num_nodes=unique_nodes.size(0),
            node_sequence=unique_nodes,
            edge_weight=edge_weight,
            inverse_idx=inverse_idx,
        )
        return Graph(data)

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
        in the from_temporal_graph (filtering non-time-respecting paths of order 2)
        or from_DAGs (reindexing with dataloader) functions.

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
            ho_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
        else:
            ho_index, edge_weight = MultiOrderModel.lift_order_edge_index_weighted(
                edge_index, edge_weight=edge_weight, num_nodes=node_sequence.size(0), aggr=aggr
            )
        node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)

        # Aggregate
        if save:
            gk = MultiOrderModel.aggregate_edge_index(ho_index, node_sequence, edge_weight)
            gk.mapping = IndexMap([tuple(mapping.to_ids(v.cpu())) for v in gk.data.node_sequence])
        else:
            gk = None
        return ho_index, node_sequence, edge_weight, gk

    @staticmethod
    def from_temporal_graph(
        g: TemporalGraph, delta: float | int = 1, max_order: int = 1, weight: str = "edge_weight", cached: bool = True
    ) -> MultiOrderModel:
        """Creates multiple higher-order De Bruijn graph models for paths in a temporal graph."""
        m = MultiOrderModel()
        edge_index, timestamps = sort_edge_index(g.data.edge_index, g.data.t)
        node_sequence = torch.arange(g.data.num_nodes, device=edge_index.device).unsqueeze(1)
        if weight in g.data:
            edge_weight = g.data[weight]
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        if cached or max_order == 1:
            m.layers[1] = MultiOrderModel.aggregate_edge_index(
                edge_index=edge_index, node_sequence=node_sequence, edge_weight=edge_weight
            )
            m.layers[1].mapping = g.mapping

        if max_order > 1:
            # Compute null model
            null_model_edge_index, null_model_edge_weight = MultiOrderModel.lift_order_edge_index_weighted(
                edge_index, edge_weight=edge_weight, num_nodes=node_sequence.size(0), aggr="src"
            )
            # Update node sequences
            node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)
            # Remove non-time-respecting higher-order edges
            time_diff = timestamps[null_model_edge_index[1]] - timestamps[null_model_edge_index[0]]
            non_negative_mask = time_diff > 0
            delta_mask = time_diff <= delta
            time_respecting_mask = non_negative_mask & delta_mask
            edge_index = null_model_edge_index[:, time_respecting_mask]
            edge_weight = null_model_edge_weight[time_respecting_mask]
            # Aggregate
            if cached or max_order == 2:
                m.layers[2] = MultiOrderModel.aggregate_edge_index(
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
    def from_DAGs(
        dag_data: DAGData, max_order: int = 1, mode: str = "propagation", cached: bool = True
    ) -> MultiOrderModel:
        """
        Creates multiple higher-order De Bruijn graphs for paths in DAGData.

        Args:
            dag_data: The DAGData object containing the DAGs as list of PyG Data objects
                with sorted edge indices, node sequences and num_nodes.
            max_order: The maximum order of the MultiOrderModel that should be computed
            mode: The process that we assume. Can be "diffusion" or "propagation".
            cached: Whether to save the aggregated higher-order graphs smaller than max order
                in the MultiOrderModel.
        """
        m = MultiOrderModel()

        # We assume that the DAGs are sorted and that walks are remapped to a DAG
        dag_graph = next(iter(DataLoader(dag_data.dags, batch_size=len(dag_data.dags)))).to(config["torch"]["device"])
        edge_index = dag_graph.edge_index
        node_sequence = dag_graph.node_sequence
        if dag_graph.edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        else:
            edge_weight = dag_graph.edge_weight
        if mode == "diffusion":
            edge_weight = (
                edge_weight / degree(edge_index[0], dtype=torch.long, num_nodes=node_sequence.size(0))[edge_index[0]]
            )
            aggr = "mul"
        elif mode == "propagation":
            aggr = "src"

        m.layers[1] = MultiOrderModel.aggregate_edge_index(
            edge_index=edge_index, node_sequence=node_sequence, edge_weight=edge_weight
        )
        m.layers[1].mapping = dag_data.mapping

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
            # COMPUTING CONTRIBUTION FROM NUM PATHS AND NONERO OUTDEGREES SEPARATELY
            # TODO: CAN IT BE DONE TOGETHER?

            edge_index = self.layers[1].data.edge_index
            # Adding dof from Number of paths of length k
            for k in range(1, max_order + 1):
                if k > 1:
                    num_nodes = 0 if edge_index.numel() == 0 else edge_index.max().item() + 1
                    edge_index = self.lift_order_edge_index(edge_index, num_nodes)
                # counting number of len k paths
                num_len_k_paths = edge_index.shape[1]  # edge_index.max().item() +1  # Number of paths of length k
                dof += num_len_k_paths

            # removing dof from total probability of nonzero degree nodes
            for k in range(1, max_order + 1):
                if k == 1:
                    edge_index_adj = self.layers[1].data.edge_index
                    edge_index = edge_index_adj
                else:
                    edge_index, _ = edge_index @ edge_index_adj
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
        # TODO: put this tensor directly in dag_graph (intead of edge_weight) and remove the following line
        # getting the index of the last edge of each path (to be used to extract weights)
        last_edge_ixs = dag_graph.ptr[1:] - torch.arange(2, dag_graph.ptr.shape[0] + 1)
        frequencies = dag_graph.edge_weight[last_edge_ixs]

        # Get ixs starting nodes
        # Q: Is dag_graph.ptr[:-1] enough to get the start_ixs?
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
        # TODO: put this tensor directly in dag_graph (intead of edge_weight) and remove the following line
        # getting the index of the last edge of each path (to be used to extract weights)
        last_edge_ixs = dag_graph.ptr[1:] - torch.arange(2, dag_graph.ptr.shape[0] + 1)
        frequencies = dag_graph.edge_weight[last_edge_ixs]

        # Get intermediate HO nodes ixs
        mask = torch.ones(dag_graph.num_nodes, dtype=bool)
        mask[dag_graph.edge_index[1]] = False
        ixs = torch.where(mask)[0]
        num_ixs = ixs.shape[0]
        ho_intermediate_ixs = ixs - torch.arange(num_ixs) * order

        # computing loglikelihood of subpaths
        transition_probabilities = compute_transition_probabilities(self.layers[order])[
            self.layers[order + 1].data.inverse_idx[ho_intermediate_ixs]
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
            transition_probabilities = compute_transition_probabilities(self.layers[max_order])
            log_transition_probabilities = torch.log(transition_probabilities)
            llh_by_subpath = (
                log_transition_probabilities * self.layers[max_order].data.edge_weight
            )
            llh += llh_by_subpath.sum().item()
        else:
            # Compute likelihood for zeroth order (to be modified)
            # TODO: modify once we have zeroth order in mon
            # (then won t need to compute emission probs from dag_graph -- which also hinders us from computing the lh that a new set of path swas generated by the model)
            # getting the index of the last edge of each path (to be used to extract weights)
            last_edge_ixs = dag_graph.ptr[1:] - torch.arange(2, dag_graph.ptr.shape[0] + 1)
            frequencies = dag_graph.edge_weight[last_edge_ixs]
            counts = torch.bincount(
                dag_graph.node_sequence.T[0], frequencies.repeat_interleave(dag_graph.ptr[1:] - dag_graph.ptr[:-1])
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

    def estimate_order(self, dag_data: DAGData, max_order: int = None, significance_threshold: float = 0.01) -> int:
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

        dag_graph = next(iter(DataLoader(dag_data.dags, batch_size=len(dag_data.dags)))).to(config["torch"]["device"])
        assert set(dag_data.mapping.node_ids).intersection(set(self.layers[1].mapping.node_ids)) == set(
            dag_data.mapping.node_ids
        ), "Input DAGData doesn t have the same set of nodes as those of the multi-order network"

        max_accepted_order = 1

        # Test for highest order that passes
        # likelihood ratio test against null model
        for k in range(2, max_order + 1):
            if self.likelihood_ratio_test(
                dag_graph, max_order_null=k - 1, max_order=k, significance_threshold=significance_threshold
            )[0]:
                max_accepted_order = k

        return max_accepted_order


def compute_weighted_outdegrees(graph: Graph) -> torch.Tensor:
    """
    Compute the weighted outdegrees of each node in the graph.

    Args:
        graph (Graph): pathpy graph object.

    Returns:
        tensor: Weighted outdegrees of nodes.
    """
    weighted_outdegree = scatter(
        graph.data.edge_weight, graph.data.edge_index[0], dim=0, dim_size=graph.data.num_nodes, reduce="sum"
    )
    return weighted_outdegree


def compute_transition_probabilities(graph: Graph) -> torch.Tensor:
    """
    Compute transition probabilities based on weighted outdegrees.

    Args:
        graph (Graph): pathpy graph object.

    Returns:
        tensor: Transition probabilities.
    """
    weighted_outdegree = compute_weighted_outdegrees(graph)
    source_ids = graph.data.edge_index[0]
    return graph.data.edge_weight / weighted_outdegree[source_ids]
