"""Utility functions for lifting the order of a graph (line-graph transformation)."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, cumsum, degree


def aggregate_node_attributes(
    edge_index: torch.Tensor, node_attribute: torch.Tensor, aggr: str = "src"
) -> torch.Tensor:
    """Aggregate the node attributes of each pair of nodes in the edge index.

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


def lift_order_edge_index(edge_index: torch.Tensor, num_nodes: int | None = None) -> torch.Tensor:
    """Line graph transformation.

    Do a line graph transformation on the edge index to lift the order of the graph by one.
    Assumes that the edge index is sorted.

    Args:
        edge_index: A **sorted** edge index tensor of shape (2, num_edges).
        num_nodes: The number of nodes in the graph. If not given,
            it will be inferred from the edge index (maximum node index + 1).

    Returns:
        The edge index of the lifted (line) graph.
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    outdegree = degree(edge_index[0], dtype=torch.long, num_nodes=num_nodes)
    # Map outdegree to each destination node to create an edge for each combination
    # of incoming and outgoing edges for each destination node
    outdegree_per_dst = outdegree[edge_index[1]]
    # Create sources of the new higher-order edges
    ho_edge_srcs = torch.repeat_interleave(outdegree_per_dst)

    # Create destination nodes that start the indexing after the cumulative sum of the outdegree
    # of all previous nodes in the ordered sequence of nodes
    ptrs = cumsum(outdegree, dim=0)[:-1]
    ho_edge_dsts = torch.repeat_interleave(ptrs[edge_index[1]], outdegree_per_dst)
    idx_correction = torch.arange(ho_edge_srcs.size(0), dtype=torch.long, device=edge_index.device)
    idx_correction -= cumsum(outdegree_per_dst, dim=0)[ho_edge_srcs]
    ho_edge_dsts += idx_correction
    return torch.stack([ho_edge_srcs, ho_edge_dsts], dim=0)


def lift_order_edge_index_weighted(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int | None = None, aggr: str = "src"
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
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    ho_index = lift_order_edge_index(edge_index, num_nodes)
    ho_edge_weight = aggregate_node_attributes(ho_index, edge_weight, aggr)

    return ho_index, ho_edge_weight


def lift_node_sequence(edge_index: torch.Tensor, node_sequence: torch.Tensor) -> torch.Tensor:
    """Extend node sequences by one order along an edge index.

    Each edge `(u, v)` of the (k-1)-th order graph becomes a node of the k-th order graph,
    representing the path of `u` followed by the last first-order node of `v`.

    Args:
        edge_index: A **sorted** edge index tensor of shape (2, num_edges).
        node_sequence: The node sequences of the (k-1)-th order graph, of shape (num_nodes, k-1).

    Returns:
        The node sequences of the k-th order graph, of shape (num_edges, k).
    """
    return torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)


def lift_order_step(
    edge_index: torch.Tensor,
    node_sequence: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    aggr: str = "src",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Lift an edge index together with its node sequences by one order.

    Combines the line-graph transformation of the edge index with the corresponding
    extension of the node sequences, so that the result again describes a graph whose
    nodes are paths of first-order nodes. The result is **not** aggregated: duplicate
    node sequences are left for [`aggregate_edge_index`][pathpyG.algorithms.lift_order.aggregate_edge_index]
    (or [`HigherOrderGraph.aggregate`][pathpyG.core.higher_order_graph.HigherOrderGraph.aggregate]) to merge.

    Args:
        edge_index: A **sorted** edge index tensor of shape (2, num_edges).
        node_sequence: The node sequences of the (k-1)-th order graph.
        edge_weight: The edge weights of the (k-1)-th order graph. If None, the lifted
            graph is returned without weights.
        aggr: The aggregation method for the edge weights. One of "src", "dst", "max",
            "mul" or "add". Ignored if `edge_weight` is None.

    Returns:
        A tuple of the lifted edge index, the lifted node sequences and the aggregated
        edge weights (None if `edge_weight` was None).
    """
    if edge_weight is None:
        ho_index = lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
    else:
        ho_index, edge_weight = lift_order_edge_index_weighted(
            edge_index, edge_weight=edge_weight, num_nodes=node_sequence.size(0), aggr=aggr
        )
    return ho_index, lift_node_sequence(edge_index, node_sequence), edge_weight


def aggregate_edge_index(
    edge_index: torch.Tensor,
    node_sequence: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    node_weight: torch.Tensor | None = None,
    edge_attrs: dict[str, torch.Tensor] | None = None,
    min_num_nodes: int | None = None,
    aggr: str = "sum",
) -> Data:
    r"""Aggregate the possibly duplicated edges in the (higher-order) edge index.

    Aggregate the possibly duplicated edges in the (higher-order) edge index and return the
    (higher-order) edge index without duplicates, together with the node sequences.

    Like the rest of this module this works at the level of tensors and does not build a graph
    object. Use [`HigherOrderGraph.aggregate`][pathpyG.core.higher_order_graph.HigherOrderGraph.aggregate]
    to obtain a De Bruijn graph with its node ID mapping.

    This method can be seen as a higher-order generalization of the `torch_geometric.utils.coalesce` method.
    It is used for example to generate the DeBruijn graph of a given order from the corresponding line graph.

    Rows of `node_sequence` are *instances*: one entry per occurrence of a higher-order node in the
    data, before identical occurrences are merged. `edge_weight` and `node_weight` are given per
    instance and are summed onto the merged edges and nodes respectively.

    Info: Node and edge weights
        For a De Bruijn graph of order $k$ built from observed paths, the aggregated weights are the
        occurrence counts of $k$-grams and $(k+1)$-grams. They satisfy

        $$
        \mathrm{node\_weight}_k(w) = \mathrm{start\_weight}_k(w)
            + \sum_{u \to w \in E_k} \mathrm{edge\_weight}_k(u \to w),
        $$

        because every occurrence of $w$ either starts an observation or is preceded by exactly one
        instance of an in-edge of $w$. See
        [`HigherOrderGraph`][pathpyG.core.higher_order_graph.HigherOrderGraph].

    Args:
        edge_index: The edge index of a (higher-order) graph where each source and destination node
            corresponds to a node which is an edge in the (k-1)-th order graph.
        node_sequence: The node sequences of first order nodes that each node in the edge index corresponds to.
        edge_weight: The edge weights corresponding to the edge index. Defaults to ones.
        node_weight: Per-instance node weights, aligned with the rows of `node_sequence`. If given,
            they are summed onto the aggregated nodes and stored as `node_weight`.
        edge_attrs: Further per-instance edge attributes to carry along, keyed by the name to store
            them under. They are reduced with `aggr` just like `edge_weight`.
        min_num_nodes: Lower bound on the number of nodes of the aggregated graph. Only meaningful
            for order 1, where first-order nodes that no path traverses are kept as isolated nodes
            so that node-indexed attributes cover every first-order node.
        aggr: The aggregation method to use for the edge weights. One of "sum", "mean", "min", "max".

    Returns:
        A [Data][torch_geometric.data.Data] object holding the aggregated edge index, the node
        sequences and the edge weights, as well as the aggregated node weights if `node_weight`
        was given.
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    order = node_sequence.size(1)
    if order == 1:
        # Order-1 node indices already *are* first-order node indices, so no remapping is
        # needed and untraversed first-order nodes remain as isolated nodes.
        instance_to_node = node_sequence.squeeze(1)
        num_nodes = int(instance_to_node.max().item()) + 1 if instance_to_node.numel() > 0 else 0
        if min_num_nodes is not None:
            num_nodes = max(num_nodes, min_num_nodes)
        unique_nodes = torch.arange(num_nodes, device=edge_index.device).unsqueeze(1)
        mapped_edge_index = instance_to_node[edge_index]
    else:
        unique_nodes, instance_to_node = torch.unique(node_sequence, dim=0, return_inverse=True)
        num_nodes = unique_nodes.size(0)
        mapped_edge_index = instance_to_node[edge_index]

    extra_attributes = list((edge_attrs or {}).items())
    aggregated_edge_index, aggregated_attributes = coalesce(
        mapped_edge_index,
        edge_attr=[edge_weight] + [value for _, value in extra_attributes],
        num_nodes=num_nodes,
        reduce=aggr,
    )
    data = Data(
        edge_index=aggregated_edge_index,
        num_nodes=num_nodes,
        node_sequence=unique_nodes,
        edge_weight=aggregated_attributes[0],
    )
    for (name, _), value in zip(extra_attributes, aggregated_attributes[1:]):
        data[name] = value
    if node_weight is not None:
        aggregated_node_weight = torch.zeros(num_nodes, dtype=node_weight.dtype, device=node_weight.device)
        aggregated_node_weight.scatter_add_(0, instance_to_node, node_weight)
        data.node_weight = aggregated_node_weight
    return data
