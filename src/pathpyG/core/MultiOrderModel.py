from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data, DataLoader
from torch_geometric import EdgeIndex

from pathpyG import Graph
from pathpyG import DAGData
from pathpyG import TemporalGraph

from pathpyG.utils.config import config
from pathpyG.algorithms.temporal import temporal_graph_to_event_dag
from pathpyG.core.IndexMap import IndexMap

from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.utils import cumsum, coalesce, degree

# TODO: Add description for arguments
class MultiOrderModel:
    """MultiOrderModel based on torch_geometric.Data."""

    def __init__(self):
        self.layers = {}
    
    def __str__(self) -> str:
        """Return a string representation of the higher-order graph."""
        max_order = len(self.layers)
        s = f"MultiOrderModel with max. order {max_order}"
        return s

    @staticmethod
    def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
        # Since this is a complicated function, we will use the following example to explain the steps:
        # Example:
        #   edge_index = [[0, 0, 1, 1, 1, 3, 4, 5, 6],
        #                 [1, 3, 2, 3, 6, 4, 5, 7, 5]]

        # Compute the outdegree of each node which we will use to get all the edge combinations that lead to a higher order edge
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

        # We use each edge from the edge index as new node and assign the new indices in the order of the original edge index
        # Each higher order node has one outgoing edge for each outgoing edge of the original destination node
        # Since we keep the ordering, we can just repeat each node using the outdegree_per_dst tensor
        # Example:
        #   ho_edge_srcs = [0, 0, 0, 1, 3, 4, 5, 6, 8]
        ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)

        # For each node, we calculate pointers of shape (num_nodes,) that indicate the start of the original edges (new higher order nodes) that have the node as source node
        # (Note we use PyG's cumsum function because it adds a 0 at the beginning of the tensor and we want the `left` boundaries of the intervals, so we also remove the last element of the result with [:-1])
        # Example:
        #   ptrs = [0, 2, 5, 5, 6, 7, 8, 9]
        ptrs = cumsum(outdegree, dim=0)[:-1]

        # Use these pointers to get the start of the edges for each higher order source node and repeat it `outdegree` times
        # Since we keep the ordering, all new higher order edges that have the same source node are indexed consecutively
        # Example:
        #   ho_edge_dsts = [2, 2, 2, 5, 5, 8, 6, 7, 7]
        ho_edge_dsts = torch.repeat_interleave(ptrs[edge_index[1]], outdegree_per_dst)

        # Since the above only repeats the start of the edges, we need to add (0, 1, 2, 3, ...) for all `outdegree` number of edges consecutively to get the correct destination nodes
        # We can achieve this by starting with a range from (0, 1, ..., num_new_edges)
        # Example: 
        #   idx_correction    = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        idx_correction = torch.arange(num_new_edges, dtype=torch.long, device=edge_index.device)
        # Then, we subtract the cumulative sum of the outdegree for each destination node to get a tensor.
        # Example:
        #   idx_correction    = [0, 1, 2, 0, 0, 0, 0, 0, 0]
        idx_correction -= cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
        # Finally, we add this tensor to the destination nodes to get the correct destination nodes for each higher order edge
        # Example:
        #   ho_edge_dsts = [2, 3, 4, 5, 5, 8, 6, 7, 7]
        ho_edge_dsts += idx_correction
        # tensor([[0, 0, 0, 1, 3, 4, 5, 6, 8],
        #         [2, 3, 4, 5, 5, 8, 6, 7, 7]])
        return torch.stack([ho_edge_srcs, ho_edge_dsts], dim=0)

    @staticmethod
    def from_temporal_graph(g: TemporalGraph, delta: float | int = 1, max_order: int = 1) -> MultiOrderModel:
        """Creates multiple higher-order De Bruijn graph models for paths in a temporal graph."""
        m = MultiOrderModel()

        # TODO: add higher-order layers
        # m.layers.append(...)
        return m

    @staticmethod
    def from_DAGs(data: pp.DAGData, max_order: int = 1) -> pp.MultiOrderModel:
        """Creates multiple higher-order De Bruijn graphs for paths in DAGData."""
        m = pp.MultiOrderModel()

        # Outsource to DAGData
        data_list = []
        for dag in data.dags:
            edge_index = coalesce(dag.long())
            unique_nodes = torch.unique(edge_index)
            num_nodes = unique_nodes.size(0)
            data_list.append(Data(edge_index=edge_index, node_sequences=unique_nodes.unsqueeze(1), num_nodes=num_nodes))
        dag_graph = next(iter(DataLoader(data_list, batch_size=len(data.dags))))
        edge_index = dag_graph.edge_index
        old_node_sequences = dag_graph.node_sequences

        m.layers[1] = pp.Graph(dag_graph)
        
        for k in range(2, max_order+1):
            # Lift order
            node_sequences = torch.cat([old_node_sequences[edge_index[0]], old_node_sequences[edge_index[1]][:, -1:]], dim=1)
            ho_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=old_node_sequences.size(0))
            unique_nodes, inverse_idx = torch.unique(node_sequences, dim=0, return_inverse=True)

            # Save for the next iteration
            edge_index = ho_index
            old_node_sequences = node_sequences

            # Save aggregated higher-order graph
            mapped_ho_index = inverse_idx[ho_index]
            aggregated_ho_index, edge_weights = coalesce(mapped_ho_index, edge_attr=torch.ones(ho_index.size(1)), num_nodes=unique_nodes.size(0))
            m.layers[k] = pp.Graph(Data(edge_index=aggregated_ho_index, num_nodes=unique_nodes.size(0), node_sequences=unique_nodes, edge_weights=edge_weights))
        
        return m