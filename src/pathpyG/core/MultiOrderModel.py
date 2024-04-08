from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import cumsum, coalesce, degree, sort_edge_index

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
            edge_weight = dag_graph.edge_attr
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
