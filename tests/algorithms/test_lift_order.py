import pytest
import torch
from torch_geometric.data import Data

from pathpyG.algorithms.lift_order import (
    aggregate_edge_index,
    aggregate_node_attributes,
    lift_order_edge_index,
    lift_order_edge_index_weighted,
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
    with pytest.raises(ValueError):
        aggregate_node_attributes(edge_index=edge_index, node_attribute=node_attribute, aggr="unknown")


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
    data = aggregate_edge_index(edge_index=edge_index, edge_weight=edge_weight, node_sequence=node_sequence)
    assert data.edge_index.tolist() == [[0, 0, 1], [1, 2, 0]]
    assert data.edge_weight.tolist() == [3, 3, 4]
    assert data.node_sequence.tolist() == [[1, 2], [2, 3], [4, 5]]
    # A tensor-level helper: it returns the aggregated data, not a graph object.
    assert isinstance(data, Data)
    assert not hasattr(data, "mapping")
    assert "inverse_idx" not in data
    assert "node_weight" not in data


def test_aggregate_edge_index_aggregates_edge_attributes():
    """Further per-instance edge attributes are reduced onto the merged edges like the weights."""
    data = aggregate_edge_index(
        edge_index=torch.tensor([[0, 2], [1, 1]]),
        node_sequence=torch.tensor([[1, 2], [2, 3], [1, 2]]),
        edge_weight=torch.tensor([1.0, 2.0]),
        edge_attrs={"edge_num_walks": torch.tensor([2.0, 3.0])},
    )
    # Instances 0 and 2 are both the path (1, 2), so both edges become the edge (1, 2) -> (2, 3).
    assert data.edge_weight.tolist() == [3.0]
    assert data.edge_num_walks.tolist() == [5.0]


def test_aggregate_edge_index_first_order_with_unvisited_node():
    """First-order indices are kept as node indices, even if a lower index is never visited."""
    # first-order nodes 0..4; node 1 is never visited; the steps are 0->4 and 2->0
    data = aggregate_edge_index(
        edge_index=torch.tensor([[0, 2], [1, 3]]), node_sequence=torch.tensor([[0], [4], [2], [0]])
    )
    assert data.num_nodes == 5
    assert data.edge_index.tolist() == [[0, 2], [4, 0]]
    assert data.edge_weight.tolist() == [1.0, 1.0]
    assert data.node_sequence.tolist() == [[0], [1], [2], [3], [4]]


def test_aggregate_edge_index_first_order_min_num_nodes():
    """`min_num_nodes` keeps trailing first-order nodes that no path visits."""
    data = aggregate_edge_index(
        edge_index=torch.tensor([[0], [1]]), node_sequence=torch.tensor([[0], [1]]), min_num_nodes=4
    )
    assert data.num_nodes == 4
    assert data.node_sequence.tolist() == [[0], [1], [2], [3]]


def test_aggregate_edge_index_rejects_order_zero():
    with pytest.raises(ValueError):
        aggregate_edge_index(edge_index=torch.tensor([[0], [0]]), node_sequence=torch.empty((1, 0), dtype=torch.long))
