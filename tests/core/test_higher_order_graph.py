from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from pathpyG.core.graph import Graph
from pathpyG.core.higher_order_graph import HigherOrderGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

# Paths used in the tests:
#
#   a -> c -> d   (weight 3)
#   b -> c -> d   (weight 1)
#
# Node e is part of the mapping but never visited. Weighted visit counts are
# a: 3, b: 1, c: 4, d: 4, e: 0 (12 visits in total).


@pytest.fixture
def paths() -> PathData:
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walk(("a", "c", "d"), weight=3)
    paths.append_walk(("b", "c", "d"), weight=1)
    return paths


@pytest.fixture
def order_zero() -> HigherOrderGraph:
    return HigherOrderGraph.from_node_weights(
        torch.tensor([3.0, 1.0, 4.0, 4.0, 0.0]), first_order_mapping=IndexMap(list("abcde"))
    )


def test_is_graph(paths):
    h = HigherOrderGraph.from_path_data(paths, order=2)
    assert isinstance(h, Graph)


def test_node_sequence_is_internal(paths):
    h = HigherOrderGraph.from_path_data(paths, order=2)
    assert "node_sequence" in h.data
    assert "node_sequence" not in h.node_attrs()


def test_init_creates_identity_node_sequence_for_order_one():
    data = Data(edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
    h = HigherOrderGraph(data)
    assert h.order == 1
    assert h.data.node_sequence.tolist() == [[0], [1], [2]]
    assert h.n_first_order == 3


def test_to_device(paths):
    h = MultiOrderModel.from_path_data(paths, max_order=2).layers[2]
    moved = h.to(torch.device("cpu"))
    assert moved is h
    assert h.data.node_sequence.device.type == "cpu"
    assert h.data.inverse_idx.device.type == "cpu"


# ---------------------------------------------------------------------------
# Order 0
# ---------------------------------------------------------------------------


def test_from_node_weights(order_zero):
    assert order_zero.order == 0
    assert order_zero.n == 1
    assert order_zero.m == 5
    assert order_zero.nodes == [()]
    assert order_zero.n_first_order == 5
    assert order_zero.mapping.has_tuple_ids
    assert order_zero.data.node_sequence.shape == (1, 0)
    # one self-loop per first-order node, including the unvisited node e
    assert order_zero.data.edge_index.as_tensor().tolist() == [[0] * 5, [0] * 5]
    assert sorted(zip(order_zero.data.edge_first_order_node.tolist(), order_zero.data.edge_weight.tolist())) == [
        (0, 3.0),
        (1, 1.0),
        (2, 4.0),
        (3, 4.0),
        (4, 0.0),
    ]


def test_order_zero_transition_probabilities_are_visit_probabilities(order_zero):
    probabilities = order_zero.transition_probabilities(edge_attr="edge_weight")
    by_node = dict(zip(order_zero.data.edge_first_order_node.tolist(), probabilities.tolist()))
    expected = {0: 3 / 12, 1: 1 / 12, 2: 4 / 12, 3: 4 / 12, 4: 0.0}
    assert by_node == pytest.approx(expected)


def order_zero_data(**kwargs) -> Data:
    data = {
        "edge_index": torch.zeros((2, 2), dtype=torch.long),
        "num_nodes": 1,
        "node_sequence": torch.empty((1, 0), dtype=torch.long),
        "edge_first_order_node": torch.tensor([0, 1]),
        "edge_weight": torch.tensor([1.0, 1.0]),
    }
    data.update(kwargs)
    return Data(**{k: v for k, v in data.items() if v is not None})


def test_order_zero_requires_n_first_order():
    with pytest.raises(ValueError):
        HigherOrderGraph(order_zero_data())
    assert HigherOrderGraph(order_zero_data(), n_first_order=2).order == 0


def test_order_zero_has_at_most_one_node():
    data = order_zero_data(num_nodes=2, node_sequence=torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(ValueError):
        HigherOrderGraph(data, n_first_order=2)


def test_order_zero_cannot_be_projected(order_zero):
    with pytest.raises(ValueError):
        order_zero.to_first_order()
    with pytest.raises(ValueError):
        order_zero.bipartite_edge_index()


def test_from_path_data_order_zero(paths):
    h = HigherOrderGraph.from_path_data(paths, order=0)
    assert h.order == 0
    assert sorted(zip(h.data.edge_first_order_node.tolist(), h.data.edge_weight.tolist())) == [
        (0, 3.0),
        (1, 1.0),
        (2, 4.0),
        (3, 4.0),
        (4, 0.0),
    ]


def test_from_temporal_graph_rejects_order_zero():
    t = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2)])
    with pytest.raises(ValueError):
        HigherOrderGraph.from_temporal_graph(t, order=0)


def test_str_order_zero(order_zero):
    assert str(order_zero).startswith("Higher-order graph of order 0 with 1 nodes and 5 edges")


# ---------------------------------------------------------------------------
# Order 1
# ---------------------------------------------------------------------------


def test_aggregate_keeps_unvisited_first_order_nodes():
    # First-order nodes 0..4, node 1 is never visited; the steps are 0->4 and 2->0.
    h = HigherOrderGraph.aggregate(
        torch.tensor([[0, 2], [1, 3]]),
        torch.tensor([[0], [4], [2], [0]]),
        first_order_mapping=IndexMap(list("abcde")),
    )
    assert h.order == 1
    assert h.n == 5
    assert h.edges == [("a", "e"), ("c", "a")]
    assert h.data.edge_weight.tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Combining higher-order graphs
# ---------------------------------------------------------------------------


def test_add_order_zero():
    g1 = HigherOrderGraph.from_node_weights(torch.tensor([1.0, 2.0, 3.0]), first_order_mapping=IndexMap(list("abc")))
    g2 = HigherOrderGraph.from_node_weights(torch.tensor([4.0, 5.0]), first_order_mapping=IndexMap(list("cd")))
    g = g1 + g2

    assert isinstance(g, HigherOrderGraph)
    assert g.order == 0
    assert g.nodes == [()]
    assert g.n_first_order == 4
    assert g.first_order_mapping.node_ids.tolist() == list("abcd")
    # emitted nodes are re-indexed to the joint first-order mapping; loops are kept as multi-edges
    assert sorted(zip(g.data.edge_first_order_node.tolist(), g.data.edge_weight.tolist())) == [
        (0, 1.0),
        (1, 2.0),
        (2, 3.0),
        (2, 4.0),
        (3, 5.0),
    ]


def test_add_order_one(paths):
    h = HigherOrderGraph.from_path_data(paths, order=1)
    g = h + h
    assert isinstance(g, HigherOrderGraph)
    assert g.order == 1
    assert g.data.node_sequence.tolist() == [[i] for i in range(g.n)]


def test_add_rebuilds_node_sequence(paths):
    h = HigherOrderGraph.from_path_data(paths, order=2)
    other = PathData(IndexMap(list("abcde")))
    other.append_walk(("c", "e", "a"))
    g = h + HigherOrderGraph.from_path_data(other, order=2)

    assert g.order == 2
    for node, seq in zip(g.nodes, g.data.node_sequence.tolist()):
        assert tuple(g.first_order_mapping.to_ids(seq).tolist()) == node


def test_add_rejects_different_orders(paths):
    with pytest.raises(ValueError):
        HigherOrderGraph.from_path_data(paths, order=1) + HigherOrderGraph.from_path_data(paths, order=2)


def test_add_rejects_plain_graph(paths):
    with pytest.raises(TypeError):
        HigherOrderGraph.from_path_data(paths, order=1) + Graph.from_edge_list([("a", "b")])
