# pylint: disable=missing-function-docstring,missing-module-docstring

import numpy as np
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


def test_multi_order_model_from_paths(simple_walks_2):
    m = MultiOrderModel.from_path_data(simple_walks_2, max_order=2)
    g1 = m.layers[1]
    g2 = m.layers[2]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [2, 2, 3, 4]]))
    assert torch.equal(g1.data.edge_weight, torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert torch.equal(m.layers[0].data.edge_weight, torch.tensor([2.0, 2.0, 4.0, 2.0, 2.0]))

    assert torch.equal(g2.data.edge_index, EdgeIndex([[0, 1], [2, 3]]))
    assert torch.equal(g2.data.edge_weight, torch.tensor([2.0, 2.0]))
    assert m.num_observations == 4.0
    assert m.walk_length is None


def test_from_path_data_keeps_untraversed_nodes():
    """First-order nodes no path traverses stay in the layers of order 0 and 1, with weight zero.

    Regression test: they used to be missing, which shifted the indices of the zeroth-order
    node distribution and silently mis-assigned the emission probabilities.
    """
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walk(("a", "b", "c"))
    model = MultiOrderModel.from_path_data(paths, max_order=2)

    assert model.layers[1].n == 5
    assert model.layers[0].data.edge_weight.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_from_path_data_rejects_invalid_order(simple_walks_2):
    with pytest.raises(ValueError, match="max_order must not be negative"):
        MultiOrderModel.from_path_data(simple_walks_2, max_order=-1)


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
    with pytest.raises(ValueError, match="and at least one"):
        MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=0, delta=4, walk_length=0)
    with pytest.raises(ValueError, match="max_order must not be negative"):
        MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=-1, delta=4)


def test_temporal_graph_without_walks_of_the_requested_length(caplog):
    """The De Bruijn layers still describe the topology; only the statistics are empty."""
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("c", "d", 99)])
    with caplog.at_level("WARNING"):
        model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=2)
    assert "no time-respecting walk" in caplog.text
    assert model.num_observations == 0.0
    assert model.layers[0].data.edge_weight.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert model.layers[1].data.edge_num_walks.tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="no observations at all"):
        model.log_likelihood(max_order=1)


def test_temporal_layers_keep_unrealizable_parts(simple_temporal_graph):
    """Walks that no observation can contain stay in the topology but carry weight zero."""
    # With delta=1 only (a->b)@1 has no delta-respecting neighbour at all.
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 9), ("c", "d", 10)])
    model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=2)

    assert model.layers[2].nodes == [("a", "b"), ("b", "c"), ("c", "d")]
    # The order-1 edges are the order-2 nodes, so their weights count the order-2 nodes' occurrences.
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
    assert _nonzero_visits(temporal.layers[0]) == _nonzero_visits(enumerated.layers[0])
    for order in range(1, walk_length + 1):
        temporal_layer, enumerated_layer = temporal.layers[order], enumerated.layers[order]
        assert _nonzero_edge_weights(temporal_layer) == _nonzero_edge_weights(enumerated_layer)


def _nonzero_visits(layer_zero) -> dict:
    emitted = layer_zero.first_order_mapping.to_ids(layer_zero.data.edge_first_order_node).tolist()
    weights = layer_zero.data.edge_weight
    return {node: pytest.approx(float(weight)) for node, weight in zip(emitted, weights) if weight != 0}


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


# ---------------------------------------------------------------------------
# Order 0
# ---------------------------------------------------------------------------


def weighted_paths() -> PathData:
    """Return the walks a -> c -> d (weight 3) and b -> c -> d (weight 1); e is never visited."""
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walk(("a", "c", "d"), weight=3)
    paths.append_walk(("b", "c", "d"), weight=1)
    return paths


def test_from_path_data_builds_order_zero_layer():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=2)
    assert sorted(model.layers) == [0, 1, 2]

    layer_zero = model.layers[0]
    assert layer_zero.order == 0
    assert layer_zero.n_first_order == 5
    # visits weighted by path frequency: a 3, b 1, c 4, d 4, e 0
    weights = dict(zip(layer_zero.data.edge_first_order_node.tolist(), layer_zero.data.edge_weight.tolist()))
    assert weights == {0: 3.0, 1: 1.0, 2: 4.0, 3: 4.0, 4: 0.0}


def test_from_path_data_order_zero_only():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=0)
    assert sorted(model.layers) == [0]
    expected = 3 * np.log(3 / 12) + 1 * np.log(1 / 12) + 8 * np.log(4 / 12)
    assert model.log_likelihood(max_order=0) == pytest.approx(expected)
    assert model.degrees_of_freedom(max_order=0) == 4
    assert np.isfinite(model.log_evidence(max_order=0))


@pytest.mark.parametrize("cached, expected_layers", [(False, [0, 1, 3]), (True, [0, 1, 2, 3])])
def test_from_path_data_uncached_keeps_orders_zero_and_one(cached, expected_layers):
    paths = PathData(IndexMap(list("abcd")))
    paths.append_walk(("a", "b", "c", "d"))
    assert sorted(MultiOrderModel.from_path_data(paths, max_order=3, cached=cached).layers) == expected_layers


@pytest.mark.parametrize("cached, expected_layers", [(False, [0, 1, 3]), (True, [0, 1, 2, 3])])
def test_from_temporal_graph_uncached_keeps_orders_zero_and_one(simple_temporal_graph, cached, expected_layers):
    model = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4, cached=cached)
    assert sorted(model.layers) == expected_layers


def test_from_path_data_with_unvisited_first_order_node():
    """A first-order node with a lower index than the visited ones is never visited (b)."""
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walk(("a", "e"))
    paths.append_walk(("c", "a"))
    layer_one = MultiOrderModel.from_path_data(paths, max_order=1).layers[1]
    assert layer_one.n == 5
    assert layer_one.edges == [("a", "e"), ("c", "a")]
    assert layer_one.data.edge_weight.tolist() == [1.0, 1.0]


def test_order_zero_start_weights_count_path_starts():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=1)
    # path starts: a (weight 3) and b (weight 1); no path starts at c, d or e
    start_weight = model._start_weight_on_edges(0)
    starts = dict(zip(model.layers[0].data.edge_first_order_node.tolist(), start_weight.tolist()))
    assert starts == {0: 3.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_log_likelihood_order_zero_with_unvisited_node():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=1)
    # every visit is a draw from the order-0 layer; the unvisited node e contributes nothing
    expected = 3 * np.log(3 / 12) + 1 * np.log(1 / 12) + 8 * np.log(4 / 12)
    assert model.log_likelihood(max_order=0) == pytest.approx(expected)


def test_log_likelihood_closed_form_with_shared_middle_node():
    paths = PathData(IndexMap(list("abcde")))
    for walk in [("a", "c", "d"), ("b", "c", "e"), ("a", "c", "e"), ("b", "c", "d")]:
        paths.append_walk(walk)
    model = MultiOrderModel.from_path_data(paths, max_order=2)
    assert model.log_likelihood(max_order=0) == pytest.approx(np.log(2 / 12) * 8 + np.log(4 / 12) * 4)
    assert model.log_likelihood(max_order=1) == pytest.approx(np.log(2 / 12) * 4 + 4 * np.log(1 / 2))
    assert model.log_likelihood(max_order=2) == pytest.approx(np.log(1 / 6) * 4 + 4 * np.log(1 / 2))


def test_degrees_of_freedom_order_zero_counts_unvisited_nodes():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=1)
    assert model.degrees_of_freedom(max_order=0) == 4


def test_from_temporal_graph_order_zero_only(simple_temporal_graph):
    model = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=0, delta=4)
    assert sorted(model.layers) == [0]
    assert model.walk_length == 1
    # with walks of one event, every event is an observation: n events give 2n visits
    assert float(model.layers[0].data.edge_weight.sum()) == 2 * model.num_observations


def test_temporal_model_has_order_zero_layer(simple_temporal_graph):
    model = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=2, delta=4)
    assert model.layers[0].order == 0
    assert model.layers[0].n_first_order == simple_temporal_graph.n
    assert model.degrees_of_freedom(max_order=0) == simple_temporal_graph.n - 1
    assert np.isfinite(model.log_likelihood(max_order=0))


def memoryless_paths() -> PathData:
    """Return all four transitions between a and b with equal weight, so that no order beats order 0."""
    paths = PathData(IndexMap(list("ab")))
    for walk in [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]:
        paths.append_walk(walk, weight=10)
    return paths


def test_estimate_order_returns_zero_for_memoryless_paths():
    model = MultiOrderModel.from_path_data(memoryless_paths(), max_order=2)
    assert model.estimate_order(max_order=1) == 0
    assert model.estimate_order() == 0


@pytest.mark.parametrize("max_order", [0, -1, 3])
def test_estimate_order_rejects_invalid_max_order(max_order):
    model = MultiOrderModel.from_path_data(memoryless_paths(), max_order=2)
    with pytest.raises(ValueError):
        model.estimate_order(max_order=max_order)


def test_estimate_order_rejects_missing_layers(simple_temporal_graph):
    paths = PathData(IndexMap(list("abcd")))
    paths.append_walk(("a", "b", "c", "d"))
    # uncached models lack the intermediate layers
    with pytest.raises(ValueError, match="cached=True"):
        MultiOrderModel.from_path_data(paths, max_order=3, cached=False).estimate_order()
    with pytest.raises(ValueError, match="cached=True"):
        MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4, cached=False).estimate_order()


def test_to_dbgnn_data_rejects_order_zero():
    model = MultiOrderModel.from_path_data(weighted_paths(), max_order=1)
    with pytest.raises(ValueError, match="at least 1"):
        model.to_dbgnn_data(max_order=0)
