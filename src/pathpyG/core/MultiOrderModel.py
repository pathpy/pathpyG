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
    def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Do a line graph transformation on the edge index to lift the order of the graph by one.

        Args:
            edge_index: A **sorted** edge index tensor of shape (2, num_edges).
            num_nodes: The number of nodes in the graph.
        """

        # Since this is a complicated function, we will use the following example to explain the steps:
        # Example:
        #   edge_index = [[0, 0, 1, 1, 1, 3, 4, 5, 6],
        #                 [1, 3, 2, 3, 6, 4, 5, 7, 5]]

        # Compute the outdegree of each node used to get all the edge combinations leading to a higher-order edge
        # Example:
        #   outdegree = [2, 3, 0, 1, 1, 1, 1, 0]
        outdegree = degree(edge_index[0], dtype=torch.long, num_nodes=num_nodes)

        # For each center node, we need to combine each outgoing edge with each incoming edge
        # We achieve this by creating `outdegree` number of edges for each destination node of the old edge index
        # Example:
        #   outdegree_per_dst = [3, 1, 0, 1, 1, 1, 1, 0, 1]
        #   num_new_edges = 9
        outdegree_per_dst = outdegree[edge_index[1]]
        num_new_edges = outdegree_per_dst.sum()

        # Use each edge from the edge index as node and assign the new indices in the order of the original edge index
        # Each higher order node has one outgoing edge for each outgoing edge of the original destination node
        # Since we keep the ordering, we can just repeat each node using the outdegree_per_dst tensor
        # Example:
        #   ho_edge_srcs = [0, 0, 0, 1, 3, 4, 5, 6, 8]
        ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)

        # For each node, we calculate pointers of shape (num_nodes,) that indicate the start of the original edges
        # (new higher-order nodes) that have the node as source node
        # (Note we use PyG's cumsum function because it adds a 0 at the beginning of the tensor and
        # we want the `left` boundaries of the intervals, so we also remove the last element of the result with [:-1])
        # Example:
        #   ptrs = [0, 2, 5, 5, 6, 7, 8, 9]
        ptrs = cumsum(outdegree, dim=0)[:-1]

        # Use these pointers to get the start of the edges for each higher-order src and repeat it `outdegree` times
        # Since we keep the ordering, all new higher-order edges that have the same src are indexed consecutively
        # Example:
        #   ho_edge_dsts = [2, 2, 2, 5, 5, 8, 6, 7, 7]
        ho_edge_dsts = torch.repeat_interleave(ptrs[edge_index[1]], outdegree_per_dst)

        # Since the above only repeats the start of the edges, we need to add (0, 1, 2, 3, ...)
        # for all `outdegree` number of edges consecutively to get the correct destination nodes
        # We can achieve this by starting with a range from (0, 1, ..., num_new_edges)
        # Example:
        #   idx_correction    = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        idx_correction = torch.arange(num_new_edges, dtype=torch.long, device=edge_index.device)
        # Then, we subtract the cumulative sum of the outdegree for each destination node to get a tensor.
        # Example:
        #   idx_correction    = [0, 1, 2, 0, 0, 0, 0, 0, 0]
        idx_correction -= cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
        # Add this tensor to the destination nodes to get the correct destination nodes for each higher-order edge
        # Example:
        #   ho_edge_dsts = [2, 3, 4, 5, 5, 8, 6, 7, 7]
        ho_edge_dsts += idx_correction
        # tensor([[0, 0, 0, 1, 3, 4, 5, 6, 8],
        #         [2, 3, 4, 5, 5, 8, 6, 7, 7]])
        return torch.stack([ho_edge_srcs, ho_edge_dsts], dim=0)

    def aggregate_edge_index(self, edge_index: torch.Tensor, node_sequences: torch.Tensor) -> Graph:
        """
        Aggregate the possibly duplicated edges in the (higher-order) edge index and return a graph object
        containing the (higher-order) edge index without duplicates and the node sequences.

        Args:
            edge_index: The edge index of a (higher-order) graph where each source and destination node
                corresponds to a node which is an edge in the (k-1)-th order graph.
            node_sequences: The node sequences of first order nodes that each node in the edge index corresponds to.
        """
        unique_nodes, inverse_idx = torch.unique(node_sequences, dim=0, return_inverse=True)
        mapped_edge_index = inverse_idx[edge_index]
        aggregated_edge_index, edge_weight = coalesce(
            mapped_edge_index,
            edge_attr=torch.ones(edge_index.size(1), device=edge_index.device),
            num_nodes=unique_nodes.size(0),
        )
        data = Data(
            edge_index=aggregated_edge_index,
            num_nodes=unique_nodes.size(0),
            node_sequences=unique_nodes,
            edge_weight=edge_weight,
        )
        return Graph(data)

    def _iterate_lift_order(
        self, edge_index: torch.Tensor, node_sequences: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift order by one and save the result in the layers dictionary of the object.
        This is a helper function that should not be called directly.
        Only use for edge_indices after the special cases have been handled
        in the from_temporal_graph (filtering non-time-respecting paths of order 2)
        or from_DAGs (reindexing with dataloader) functions.

        Args:
            edge_index: The edge index of the (k-1)-th order graph.
            node_sequences: The node sequences of the (k-1)-th order graph.
            k: The order of the graph that should be computed.
        """
        # Lift order
        ho_index = self.lift_order_edge_index(edge_index, num_nodes=node_sequences.size(0))
        node_sequences = torch.cat([node_sequences[edge_index[0]], node_sequences[edge_index[1]][:, -1:]], dim=1)

        # Aggregate
        self.layers[k] = self.aggregate_edge_index(ho_index, node_sequences)
        self.layers[k].mapping = IndexMap(
            [tuple([self.layers[1].mapping.to_id(x) for x in v.tolist()]) for v in self.layers[k].data.node_sequences]
        )
        return ho_index, node_sequences

    @staticmethod
    def from_temporal_graph(g: TemporalGraph, delta: float | int = 1, max_order: int = 1) -> MultiOrderModel:
        """Creates multiple higher-order De Bruijn graph models for paths in a temporal graph."""
        m = MultiOrderModel()
        edge_index, timestamps = sort_edge_index(g.data.edge_index, g.data.t)
        node_sequences = torch.arange(g.data.num_nodes, device=edge_index.device).unsqueeze(1)
        m.layers[1] = m.aggregate_edge_index(edge_index, node_sequences)

        if max_order > 1:
            # Compute null model
            null_model_edge_index = m.lift_order_edge_index(edge_index, num_nodes=node_sequences.size(0))
            # Update node sequences
            node_sequences = torch.cat([node_sequences[edge_index[0]], node_sequences[edge_index[1]][:, -1:]], dim=1)
            # Remove non-time-respecting higher-order edges
            time_diff = timestamps[edge_index[1]] - timestamps[edge_index[0]]
            non_negative_mask = time_diff > 0
            delta_mask = time_diff <= delta
            time_respecting_mask = non_negative_mask & delta_mask
            edge_index = null_model_edge_index[:, time_respecting_mask]
            # Aggregate
            m.layers[2] = m.aggregate_edge_index(edge_index, node_sequences)

            for k in range(3, max_order + 1):
                edge_index, node_sequences = m._iterate_lift_order(edge_index, node_sequences, k)

        return m

    @staticmethod
    def from_DAGs(dag_data: DAGData, max_order: int = 1) -> MultiOrderModel:
        """
        Creates multiple higher-order De Bruijn graphs for paths in DAGData.

        Args:
            dag_data: The DAGData object containing the DAGs as list of PyG Data objects
                with sorted edge indices, node sequences and num_nodes.
            max_order: The maximum order of the MultiOrderModel that should be computed
        """
        m = MultiOrderModel()

        # We assume that the DAGs are sorted and that walks are remapped to a DAG
        dag_graph = next(iter(DataLoader(dag_data.dags, batch_size=len(dag_data.dags)))).to(config["torch"]["device"])
        edge_index = dag_graph.edge_index
        node_sequences = dag_graph.node_sequences

        m.layers[1] = m.aggregate_edge_index(edge_index, node_sequences)
        m.layers[1].mapping = dag_data.mapping

        for k in range(2, max_order + 1):
            edge_index, node_sequences = m._iterate_lift_order(edge_index, node_sequences, k)

        return m
