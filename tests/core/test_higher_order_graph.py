"""Tests for the De Bruijn graph layer type."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from pathpyG.core.graph import Graph
from pathpyG.core.higher_order_graph import HigherOrderGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph


def test_aggregate_sums_node_weights_over_instances():
    """`node_weight` is given per instance and summed onto the merged higher-order nodes."""
    edge_index = torch.tensor([[0, 2, 2, 1], [1, 1, 3, 0]])
    node_sequence = torch.tensor([[1, 2], [2, 3], [1, 2], [4, 5]])
    graph = HigherOrderGraph.aggregate(
        edge_index,
        node_sequence,
        edge_weight=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        node_weight=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        n_first_order=6,
    )
    # Instances 0 and 2 are both the path (1, 2) and are merged.
    assert graph.data.node_sequence.tolist() == [[1, 2], [2, 3], [4, 5]]
    assert graph.data.node_weight.tolist() == [2.0, 1.0, 1.0]
    assert graph.data.edge_weight.tolist() == [3.0, 3.0, 4.0]


def test_aggregate_without_node_weights():
    """A graph that only describes a topology carries no node weight at all."""
    graph = HigherOrderGraph.aggregate(torch.tensor([[0], [1]]), torch.tensor([[1, 2], [2, 3]]), n_first_order=4)
    assert "node_weight" not in graph.data


def test_layers_do_not_store_an_instance_to_node_map():
    """`inverse_idx` used to be kept on every layer; `node_weight` replaced its only consumer."""
    paths = PathData(IndexMap(list("abc")))
    paths.append_walk(("a", "b", "c"))
    layer = MultiOrderModel.from_path_data(paths, max_order=2).layers[2]
    assert "inverse_idx" not in layer.data


def test_order_one_keeps_untraversed_nodes_as_isolated_nodes():
    edge_index = torch.tensor([[0], [1]])
    node_sequence = torch.tensor([[0], [1]])
    graph = HigherOrderGraph.aggregate(edge_index, node_sequence, node_weight=torch.tensor([1.0, 1.0]), n_first_order=5)
    assert graph.n == 5
    assert graph.data.node_weight.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert graph.data.node_sequence.tolist() == [[0], [1], [2], [3], [4]]


def test_aggregate_carries_extra_edge_attributes():
    graph = HigherOrderGraph.aggregate(
        torch.tensor([[0, 2], [1, 1]]),
        torch.tensor([[1, 2], [2, 3], [1, 2]]),
        edge_weight=torch.tensor([1.0, 2.0]),
        edge_attrs={"edge_num_walks": torch.tensor([5.0, 7.0])},
        n_first_order=4,
    )
    assert graph.data.edge_weight.tolist() == [3.0]
    assert graph.data.edge_num_walks.tolist() == [12.0]


def test_lift_turns_edge_weights_into_node_weights():
    """A node of the lifted graph is an edge of this graph, so the weights carry over."""
    paths = PathData(IndexMap(list("abcd")))
    paths.append_walks([("a", "b", "c"), ("a", "b", "d")], [2.0, 3.0])
    first_order = MultiOrderModel.from_path_data(paths, max_order=1).layers[1]

    second_order = first_order.lift()
    assert second_order.order == 2
    assert second_order.nodes == [("a", "b"), ("b", "c"), ("b", "d")]
    assert second_order.data.node_weight.tolist() == [5.0, 2.0, 3.0]


def test_from_graph_builds_an_order_one_graph():
    graph = Graph.from_edge_list([("a", "c"), ("c", "d"), ("a", "c")])
    higher_order = HigherOrderGraph.from_graph(graph)
    assert higher_order.order == 1
    assert higher_order.nodes == ["a", "c", "d"]
    assert higher_order.data.edge_weight.tolist() == [2.0, 1.0]


def test_from_path_data_and_from_temporal_graph_agree_with_the_model():
    paths = PathData(IndexMap(list("acd")))
    paths.append_walk(("a", "c", "d"), weight=2.0)
    assert HigherOrderGraph.from_path_data(paths, order=2).nodes == [("a", "c"), ("c", "d")]

    temporal = TemporalGraph.from_edge_list([("a", "c", 1), ("c", "d", 2)])
    from_graph = HigherOrderGraph.from_temporal_graph(temporal, order=2, delta=1)
    from_model = MultiOrderModel.from_temporal_graph(temporal, delta=1, max_order=2).layers[2]
    assert from_graph.nodes == from_model.nodes == [("a", "c"), ("c", "d")]


def test_order_must_be_at_least_one():
    paths = PathData(IndexMap(list("acd")))
    paths.append_walk(("a", "c", "d"))
    with pytest.raises(ValueError, match="order must be at least 1"):
        HigherOrderGraph.from_path_data(paths, order=0)


def test_mapping_must_use_tuples_above_order_one():
    from torch_geometric.data import Data

    data = Data(edge_index=torch.tensor([[0], [1]]), node_sequence=torch.tensor([[0, 1], [1, 2]]), num_nodes=2)
    with pytest.raises(ValueError, match="must use tuple IDs"):
        HigherOrderGraph(data, mapping=IndexMap(["x", "y"]), n_first_order=3)


def test_to_first_order_projects_and_sums():
    paths = PathData(IndexMap(list("abcd")))
    paths.append_walks([("a", "b", "c"), ("a", "b", "d")], [2.0, 3.0])
    second_order = MultiOrderModel.from_path_data(paths, max_order=2).layers[2]

    projected = second_order.to_first_order(mode="last")
    assert projected.n == 4
    assert projected.data.edge_weight.tolist() == [2.0, 3.0]


def test_node_id_returns_the_first_order_path():
    paths = PathData(IndexMap(list("abc")))
    paths.append_walk(("a", "b", "c"))
    model = MultiOrderModel.from_path_data(paths, max_order=2)
    assert model.layers[1].node_id(0) == "a"
    assert model.layers[2].node_id(0) == ("a", "b")
