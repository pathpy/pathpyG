from __future__ import annotations

import torch

from torch_geometric.utils import coalesce
from torch_geometric.data import Data

from pathpyG.algorithms.lift_order import aggregate_edge_index
from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.temporal_graph import TemporalGraph


def generate_bipartite_edge_index(g: Graph, g2: Graph, mapping: str = "last") -> torch.Tensor:
    """Generate edge_index for bipartite graph connecting nodes of a second-order graph to first-order nodes."""

    if mapping == "last":
        bipartide_edge_index = torch.tensor([list(range(g2.n)), [v[1] for v in g2.data.node_sequence]])

    elif mapping == "first":
        bipartide_edge_index = torch.tensor([list(range(g2.n)), [v[0] for v in g2.data.node_sequence]])
    else:
        bipartide_edge_index = torch.tensor(
            [
                list(range(g2.n)) + list(range(g2.n)),
                [v[0] for v in g2.data.node_sequence] + [v[1] for v in g2.data.node_sequence],
            ]
        )

    return bipartide_edge_index


def generate_second_order_model(g: TemporalGraph, delta: float | int = 1, weight: str = "edge_weight") -> MultiOrderModel:
    """
    Generate a multi-order model with first- and second-order layer from a temporal graph.
    This method is optimized for the memory footprint of large graphs and may be slower than creating small models with the default apporach.
    """
    data = g.data.sort_by_time()
    edge_index1, timestamps1 = data.edge_index, data.time

    node_sequence1 = torch.arange(data.num_nodes, device=edge_index1.device).unsqueeze(1)
    if weight in data:
        edge_weight = data[weight]
    else:
        edge_weight = torch.ones(edge_index1.size(1), device=edge_index1.device)

    layer1 = aggregate_edge_index(
        edge_index=edge_index1, node_sequence=node_sequence1, edge_weight=edge_weight
    )
    layer1.mapping = g.mapping
    
    node_sequence2 = torch.cat([node_sequence1[edge_index1[0]], node_sequence1[edge_index1[1]][:, -1:]], dim=1)
    node_sequence2, edge1_to_node2 = torch.unique(node_sequence2, dim=0, return_inverse=True)

    edge_index2 = []
    edge_weight2 = []
    for timestamp in timestamps1.unique():
        src_nodes2, src_nodes2_counts = edge1_to_node2[timestamps1 == timestamp].unique(return_counts=True)
        dst_nodes2, dst_nodes2_counts = edge1_to_node2[(timestamps1 > timestamp) & (timestamps1 <= timestamp + delta)].unique(return_counts=True)
        for src_node2, src_node2_count in zip(src_nodes2, src_nodes2_counts):
            dst_node2 = dst_nodes2[node_sequence2[dst_nodes2, 0] == node_sequence2[src_node2, -1]]
            dst_node2_count = dst_nodes2_counts[node_sequence2[dst_nodes2, 0] == node_sequence2[src_node2, -1]]
        
            edge_index2.append(torch.stack([src_node2.expand(dst_node2.size(0)), dst_node2], dim=0))
            edge_weight2.append(src_node2_count.expand(dst_node2.size(0)) * dst_node2_count)

    edge_index2 = torch.cat(edge_index2, dim=1)
    edge_weight2 = torch.cat(edge_weight2, dim=0)

    edge_index2, edge_weight2 = coalesce(edge_index2, edge_attr=edge_weight2, num_nodes=node_sequence2.size(0), reduce="sum")

    data2 = Data(
        edge_index=edge_index2,
        num_nodes=node_sequence2.size(0),
        node_sequence=node_sequence2,
        edge_weight=edge_weight2,
        inverse_idx=edge1_to_node2,
    )
    layer2 = Graph(data2)
    layer2.mapping = IndexMap(
        [tuple(layer1.mapping.to_ids(v.cpu())) for v in node_sequence2]
    )
    

    m = MultiOrderModel()
    m.layers[1] = layer1
    m.layers[2] = layer2
    return m