# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest
import torch
from torch_geometric import EdgeIndex

from pathpyG.algorithms.temporal import extract_time_respecting_walks
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph


def test_multi_order_model_init():
    model = MultiOrderModel()
    assert model.layers == {}
    assert model.walk_length is None
    assert model.num_observations == 0.0


def test_multi_order_model_str():
    model = MultiOrderModel()
    assert str(model) == "MultiOrderModel with max. order 0"

    model.layers[1] = "foo"
    assert str(model) == "MultiOrderModel with max. order 1"

    model.layers[5] = "bar"
    assert str(model) == "MultiOrderModel with max. order 5"

    model.walk_length = 7
    assert str(model) == "MultiOrderModel with max. order 5 of time-respecting walks of 7 events"


def test_iterate_lift_order(simple_graph_multi_edges):
    ho_index, node_sequence, edge_weight, gk = MultiOrderModel.iterate_lift_order(
        edge_index=simple_graph_multi_edges.data.edge_index,
        node_sequence=torch.arange(simple_graph_multi_edges.n).unsqueeze(1),
        mapping=simple_graph_multi_edges.mapping,
        save=True,
    )
    assert ho_index.tolist() == [[0, 2], [3, 3]]
    assert node_sequence.tolist() == [[0, 1], [0, 2], [0, 1], [1, 2]]
    assert edge_weight is None
    assert gk.data.edge_index.as_tensor().tolist() == [[0], [2]]
    assert gk.data.node_sequence.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert gk.data.edge_weight.tolist() == [2.0]
    assert gk.order == 2


def test_iterate_lift_order_carries_node_weights(simple_graph_multi_edges):
    """A node of the lifted graph is an edge of the input, so the input's edge weights become its node weights."""
    edge_index = simple_graph_multi_edges.data.edge_index
    _, _, _, gk = MultiOrderModel.iterate_lift_order(
        edge_index=edge_index,
        node_sequence=torch.arange(simple_graph_multi_edges.n).unsqueeze(1),
        mapping=simple_graph_multi_edges.mapping,
        edge_weight=torch.tensor([3.0, 5.0, 7.0, 11.0]),
        save=True,
    )
    # The three distinct edges (a,b), (a,c) and (b,c) carry 3 + 7, 5 and 11.
    assert gk.data.node_sequence.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert gk.data.node_weight.tolist() == [10.0, 5.0, 11.0]


def test_multi_order_model_from_paths(simple_walks_2):
    m = MultiOrderModel.from_path_data(simple_walks_2, max_order=2)
    g1 = m.layers[1]
    g2 = m.layers[2]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [2, 2, 3, 4]]))
    assert torch.equal(g1.data.edge_weight, torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert torch.equal(g1.data.node_weight, torch.tensor([2.0, 2.0, 4.0, 2.0, 2.0]))

    assert torch.equal(g2.data.edge_index, EdgeIndex([[0, 1], [2, 3]]))
    assert torch.equal(g2.data.edge_weight, torch.tensor([2.0, 2.0]))
    assert torch.equal(g2.data.node_weight, torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert m.num_observations == 4.0
    assert m.walk_length is None


def test_from_path_data_keeps_untraversed_nodes():
    """First-order nodes no path traverses stay in the order-1 layer, with weight zero.

    Regression test: they used to be missing, which shifted the indices of the zeroth-order
    node distribution and silently mis-assigned the emission probabilities.
    """
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walk(("a", "b", "c"))
    model = MultiOrderModel.from_path_data(paths, max_order=2)

    assert model.layers[1].n == 5
    assert model.layers[1].data.node_weight.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_from_path_data_rejects_invalid_order(simple_walks_2):
    with pytest.raises(ValueError, match="max_order must be at least one"):
        MultiOrderModel.from_path_data(simple_walks_2, max_order=0)


def test_multi_order_from_temporal_graph(simple_temporal_graph):
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4)
    g1 = m.layers[1]
    g2 = m.layers[2]
    g3 = m.layers[3]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [1, 2, 3, 4]]))
    assert torch.equal(g2.data.edge_index, EdgeIndex([[0, 1, 1], [1, 2, 3]]))
    assert torch.equal(g3.data.edge_index, EdgeIndex([[0, 0], [1, 2]]))
    assert m.walk_length == 3


def test_temporal_graph_walk_length_validation(simple_temporal_graph):
    with pytest.raises(ValueError, match="must be at least max_order"):
        MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4, walk_length=2)
    with pytest.raises(ValueError, match="max_order must be at least one"):
        MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=0, delta=4)


def test_temporal_graph_without_walks_of_the_requested_length(caplog):
    """The De Bruijn layers still describe the topology; only the statistics are empty."""
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("c", "d", 99)])
    with caplog.at_level("WARNING"):
        model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=2)
    assert "no time-respecting walk" in caplog.text
    assert model.num_observations == 0.0
    assert model.layers[1].data.node_weight.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert model.layers[1].data.edge_num_walks.tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="no observations at all"):
        model.log_likelihood(max_order=1)


def test_temporal_layers_keep_unrealizable_parts(simple_temporal_graph):
    """Walks that no observation can contain stay in the topology but carry weight zero."""
    # With delta=1 only (a->b)@1 has no delta-respecting neighbour at all.
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 9), ("c", "d", 10)])
    model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=2)

    assert model.layers[2].nodes == [("a", "b"), ("b", "c"), ("c", "d")]
    assert model.layers[2].data.node_weight.tolist() == [0.0, 1.0, 1.0]
    assert model.layers[1].data.edge_weight.tolist() == [0.0, 1.0, 1.0]
    # The De Bruijn weights still count every event once.
    assert model.layers[1].data.edge_num_walks.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("delta", [1, 4, 10])
@pytest.mark.parametrize("walk_length", [1, 2, 3])
def test_temporal_statistics_match_enumerated_walks(long_temporal_graph, delta, walk_length):
    """The dynamic program must reproduce what enumerating the walks gives, weight for weight."""
    try:
        temporal = MultiOrderModel.from_temporal_graph(
            long_temporal_graph, delta=delta, max_order=walk_length, walk_length=walk_length
        )
    except ValueError:
        pytest.skip("no walk of this length exists for this delta")

    walks = extract_time_respecting_walks(long_temporal_graph, delta=delta, length=walk_length)
    enumerated = MultiOrderModel.from_path_data(walks, max_order=walk_length)

    assert temporal.num_observations == pytest.approx(float(walks.data.dag_weight.sum()))
    for order in range(1, walk_length + 1):
        temporal_layer, enumerated_layer = temporal.layers[order], enumerated.layers[order]
        assert _nonzero_node_weights(temporal_layer) == _nonzero_node_weights(enumerated_layer)
        assert _nonzero_edge_weights(temporal_layer) == _nonzero_edge_weights(enumerated_layer)


def _nonzero_node_weights(layer) -> dict:
    weights = layer.data.node_weight
    return {layer.node_id(i): pytest.approx(float(weights[i])) for i in range(layer.n) if weights[i] != 0}


def _nonzero_edge_weights(layer) -> dict:
    edge_index = layer.data.edge_index.as_tensor()
    weights = layer.data.edge_weight
    return {
        (layer.node_id(int(source)), layer.node_id(int(destination))): pytest.approx(float(weight))
        for source, destination, weight in zip(edge_index[0], edge_index[1], weights)
        if weight != 0
    }


def test_to_dbgnn_data(simple_temporal_graph):
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4)
    data = m.to_dbgnn_data(max_order=3)
    assert torch.equal(data.edge_index, EdgeIndex([[0, 1, 2, 2], [1, 2, 3, 4]]))
    assert torch.equal(data.edge_index_higher_order, EdgeIndex([[0, 0], [1, 2]]))


def test_to_dbgnn_data_uses_distinct_walk_counts(simple_temporal_graph):
    """DBGNN is fed the De Bruijn weights, which do not depend on the observation length."""
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=2, delta=4)
    data = m.to_dbgnn_data(max_order=2)
    assert torch.equal(data.edge_weights, m.layers[1].data.edge_num_walks.float())
    assert torch.equal(data.edge_weights_higher_order, m.layers[2].data.edge_num_walks.float())


def test_to_dbgnn_data_missing_order(simple_temporal_graph):
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=2, delta=4)
    with pytest.raises(ValueError, match="not found"):
        m.to_dbgnn_data(max_order=5)
