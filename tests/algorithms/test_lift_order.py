import torch

from pathpyG.algorithms.lift_order import (
    aggregate_node_attributes,
    lift_order_edge_index,
    lift_order_edge_index_weighted,
    aggregate_edge_index,
)


def test_aggregate_node_attributes():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [1, 2, 0, 3, 0],
        ]
    )
    node_attribute = torch.tensor([1, 2, 3, 4])
    aggr_attributes = aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="src")
    assert aggr_attributes.tolist() == [1, 2, 3, 3, 4]
    aggr_attributes = aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="dst")
    assert aggr_attributes.tolist() == [2, 3, 1, 4, 1]
    aggr_attributes = aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="max")
    assert aggr_attributes.tolist() == [2, 3, 3, 4, 4]
    aggr_attributes = aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="mul")
    assert aggr_attributes.tolist() == [2, 6, 3, 12, 4]
    aggr_attributes = aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="add")
    assert aggr_attributes.tolist() == [3, 5, 4, 7, 5]


def test_lift_order_edge_index():
    # Inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/test/transforms/test_line_graph.py
    # Directed.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [1, 2, 0, 3, 0],
        ]
    )
    ho_index = lift_order_edge_index(edge_index=edge_index, num_nodes=4)
    assert ho_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0]]


def test_lift_order_edge_index_weighted():
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [1, 2, 0, 3, 0],
        ]
    )
    edge_weight = torch.tensor([1, 2, 3, 4, 5])
    ho_index, ho_weight = lift_order_edge_index_weighted(edge_index=edge_index, edge_weight=edge_weight, num_nodes=4)
    assert ho_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0]]
    assert ho_weight.tolist() == [1, 2, 2, 3, 4, 5]


def test_aggregate_edge_index():
    edge_index = torch.tensor(
        [
            [0, 2, 2, 1],
            [1, 1, 3, 0],
        ]
    )
    edge_weight = torch.tensor([1, 2, 3, 4])
    node_sequence = torch.tensor(
        [
            [1, 2],  # Node 0
            [2, 3],  # Node 1
            [1, 2],  # Node 2
            [4, 5],  # Node 3
        ]
    )
    g = aggregate_edge_index(
        edge_index=edge_index, edge_weight=edge_weight, node_sequence=node_sequence
    )
    assert g.data.edge_index.as_tensor().tolist() == [[0, 0, 1], [1, 2, 0]]
    assert g.data.edge_weight.tolist() == [3, 3, 4]
    assert g.data.node_sequence.tolist() == [
        [1, 2],
        [2, 3],
        [4, 5]
    ]
