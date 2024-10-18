"""Utility functions for lifting the order of a graph (line-graph transformation)."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, cumsum, coalesce

from pathpyG.core.Graph import Graph


def aggregate_node_attributes(
    edge_index: torch.Tensor, node_attribute: torch.Tensor, aggr: str = "src"
) -> torch.Tensor:
    """
    Aggregate the node attributes of each pair of nodes in the edge index

    This method aggregates the node attributes of each pair of nodes in the edge index
    using the aggregation method specified. The method returns an attribute for each edge.
    The aggregation methods are:
    - "src": Use the attribute of the source node for each edge.
    - "dst": Use the attribute of the destination node for each edge.
    - "max": Use the maximum of the attributes of the source and destination nodes for each edge.
    - "mul": Use the product of the attributes of the source and destination nodes for each edge.
    - "add": Use the sum of the attributes of the source and destination nodes for each edge.

    Args:
        edge_index: The edge index of the graph.
        node_attribute: The node attribute tensor.
        aggr: The aggregation method to use. One of "src", "dst", "max", "mul" or "add".

    Returns:
        The aggregated node attributes for each edge.
    """
    if aggr == "src":
        aggr_attributes = node_attribute[edge_index[0]]
    elif aggr == "dst":
        aggr_attributes = node_attribute[edge_index[1]]
    elif aggr == "max":
        aggr_attributes = torch.maximum(node_attribute[edge_index[0]], node_attribute[edge_index[1]])
    elif aggr == "mul":
        aggr_attributes = node_attribute[edge_index[0]] * node_attribute[edge_index[1]]
    elif aggr == "add":
        aggr_attributes = node_attribute[edge_index[0]] + node_attribute[edge_index[1]]
    else:
        raise ValueError(f"Unknown aggregation method {aggr}")
    return aggr_attributes


def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Line graph transformation.

    Do a line graph transformation on the edge index to lift the order of the graph by one.
    Assumes that the edge index is sorted.

    Args:
        edge_index: A **sorted** edge index tensor of shape (2, num_edges).
        num_nodes: The number of nodes in the graph.

    Returns:
        The edge index of the lifted (line) graph.
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


def lift_order_edge_index_weighted(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, aggr: str = "src"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted line graph transformation.

    Do a line graph transformation on the edge index to lift the order of the graph by one.
    Additionally, aggregate the edge weights of the (k-1)-th order graph to the (k)-th order graph.
    Assumes that the edge index is sorted.

    Args:
        edge_index: A **sorted** edge index tensor of shape (2, num_edges).
        edge_weight: The edge weights of the (k-1)th order graph.
        num_nodes: The number of nodes in the graph.
        aggr: The aggregation method to use. One of "src", "dst", "max", "mul" or "add".

    Returns:
        A tuple containing the edge index of the lifted (line) graph and the aggregated edge weights.
    """
    ho_index = lift_order_edge_index(edge_index, num_nodes)
    ho_edge_weight = aggregate_node_attributes(ho_index, edge_weight, aggr)

    return ho_index, ho_edge_weight


def aggregate_edge_index(
    edge_index: torch.Tensor, node_sequence: torch.Tensor, edge_weight: torch.Tensor | None = None, aggr: str = "sum"
) -> Graph:
    """
    Aggregate the possibly duplicated edges in the (higher-order) edge index and return a graph object
    containing the (higher-order) edge index without duplicates and the node sequences.

    This method can be seen as a higher-order generalization of the `torch_geometric.utils.coalesce` method.
    It is used for example to generate the DeBruijn graph of a given order from the corresponding line graph.

    Args:
        edge_index: The edge index of a (higher-order) graph where each source and destination node
            corresponds to a node which is an edge in the (k-1)-th order graph.
        node_sequence: The node sequences of first order nodes that each node in the edge index corresponds to.
        edge_weight: The edge weights corresponding to the edge index.

    Returns:
        A graph object containing the aggregated edge index, the node sequences, the edge weights and the inverse index.
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    unique_nodes, inverse_idx = torch.unique(node_sequence, dim=0, return_inverse=True)
    # If first order, then the indices in the node sequence are the inverse idx we would need already
    if node_sequence.size(1) == 1:
        mapped_edge_index = node_sequence.squeeze()[edge_index]
    else:
        mapped_edge_index = inverse_idx[edge_index]
    aggregated_edge_index, edge_weight = coalesce(
        mapped_edge_index,
        edge_attr=edge_weight,
        num_nodes=unique_nodes.size(0),
        reduce=aggr,
    )
    data = Data(
        edge_index=aggregated_edge_index,
        num_nodes=unique_nodes.size(0),
        node_sequence=unique_nodes,
        edge_weight=edge_weight,
        inverse_idx=inverse_idx,
    )
    return Graph(data)
