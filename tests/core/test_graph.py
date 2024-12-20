# pylint: disable=missing-function-docstring,missing-module-docstring
# pylint: disable=invalid-name

from __future__ import annotations

import pytest
import scipy.sparse as s
import torch
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.testing import get_random_edge_index

from pathpyG import Graph, IndexMap


def test_init():
    edge_index = get_random_edge_index(100, 100, 1000)
    data = Data(edge_index=edge_index, num_nodes=100)
    g = Graph(data)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert isinstance(g.edge_to_index, dict)
    assert g.data.node_sequence.size() == (g.n, 1)
    assert g.order == 1


def test_init_with_edge_index():
    edge_index = EdgeIndex(get_random_edge_index(100, 100, 1000))
    data = Data(edge_index=edge_index, num_nodes=100)
    g = Graph(data)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert isinstance(g.edge_to_index, dict)


def test_init_with_mapping():
    edge_index = get_random_edge_index(100, 100, 1000)
    data = Data(edge_index=edge_index, num_nodes=100)
    idxs = [str(i) for i in range(100)]
    mapping = IndexMap(idxs)
    g = Graph(data, mapping)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert isinstance(g.edge_to_index, dict)


def test_from_edge_index():
    edge_index = get_random_edge_index(100, 100, 1000)
    g = Graph.from_edge_index(edge_index)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert isinstance(g.edge_to_index, dict)


def test_from_edge_list():
    edge_list = [
        ("a", "b"),
        ("c", "a"),
        ("b", "c"),
    ]
    g = Graph.from_edge_list(edge_list)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert g.mapping.to_idx("a") == 0
    assert g.mapping.to_idx("b") == 1
    assert g.mapping.to_idx("c") == 2
    assert isinstance(g.edge_to_index, dict)
    assert torch.equal(g.data.edge_index, EdgeIndex([[0, 1, 2], [1, 2, 0]]))

    edge_list = [
        ("a", "c"),
        ("b", "c"),
    ]
    g = Graph.from_edge_list(edge_list)
    assert g.mapping.to_idx("a") == 0
    assert g.mapping.to_idx("b") == 1
    assert g.mapping.to_idx("c") == 2

    edge_list = [
        (1, 12),
        (2, 1),
    ]
    g = Graph.from_edge_list(edge_list)
    assert g.mapping.to_idx(1) == 0
    assert g.mapping.to_idx(2) == 1
    assert g.mapping.to_idx(12) == 2

    edge_list = [("1", "12"), ("2", "1"), ("21", "3")]
    g = Graph.from_edge_list(edge_list)
    assert g.mapping.to_idx("1") == 0
    assert g.mapping.to_idx("2") == 1
    assert g.mapping.to_idx("3") == 2
    assert g.mapping.to_idx("12") == 3
    assert g.mapping.to_idx("21") == 4


def test_from_edge_list_undirected():
    edge_list = [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
        ("c", "a"),
        ("a", "c"),
    ]
    g = Graph.from_edge_list(edge_list, is_undirected=True)
    assert isinstance(g, Graph)
    assert isinstance(g.data, Data)
    assert isinstance(g.mapping, IndexMap)
    assert isinstance(g.edge_to_index, dict)


def test_to_undirected(simple_graph):
    g_u = simple_graph.to_undirected()
    assert g_u.data.edge_index.is_undirected


def test_weighted_graph(simple_graph_multi_edges):
    assert simple_graph_multi_edges.m == 4
    weighted_graph = simple_graph_multi_edges.to_weighted_graph()
    assert weighted_graph.data.num_edges == 3
    assert weighted_graph.data.num_nodes == 3
    assert weighted_graph["edge_weight", "a", "b"] == 2


def test_node_attrs(simple_graph):
    node_attrs = simple_graph.node_attrs()
    assert isinstance(node_attrs, list)
    assert len(node_attrs) == 0

    simple_graph.data["node_class"] = torch.tensor([[1], [2], [3]])
    node_attrs = simple_graph.node_attrs()
    assert len(node_attrs) == 1


def test_edge_attrs(simple_graph):
    edge_attrs = simple_graph.edge_attrs()
    assert isinstance(edge_attrs, list)
    assert len(edge_attrs) == 0

    simple_graph.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    edge_attrs = simple_graph.edge_attrs()
    assert len(edge_attrs) == 1


def test_nodes(simple_graph):
    i = 0
    for node in simple_graph.nodes:
        assert node in ["a", "b", "c"]
        i += 1
    assert i == 3


def test_edges(simple_graph):
    i = 0
    for edge in simple_graph.edges:
        assert tuple(edge) in [("a", "b"), ("b", "c"), ("a", "c")]
        i += 1
    assert i == 3


def test_successors(simple_graph):
    s = simple_graph.successors("a")
    assert s == ["b", "c"]

    s = simple_graph.successors("c")
    assert len(s) == 0


def test_predecessors(simple_graph):
    s = simple_graph.predecessors("b")
    assert s == ["a"]

    s = simple_graph.predecessors("a")
    assert len(s) == 0


def test_is_edge(simple_graph):
    assert simple_graph.is_edge("a", "b")
    assert not simple_graph.is_edge("b", "a")
    assert simple_graph.is_edge("a", "c")
    assert not simple_graph.is_edge("c", "a")
    assert simple_graph.is_edge("b", "c")
    assert not simple_graph.is_edge("c", "b")


def test_sparse_adj_matrix(simple_graph):
    adj = simple_graph.sparse_adj_matrix()
    assert adj.shape == (3, 3)
    assert adj.nnz == 3

    edge_weight = torch.tensor([[1], [1], [2]])
    simple_graph.data["edge_weight"] = edge_weight
    weighted_adj = simple_graph.sparse_adj_matrix("edge_weight")
    assert weighted_adj.shape == (3, 3)
    assert weighted_adj.nnz == 3
    assert isinstance(weighted_adj, s.coo_matrix)
    assert weighted_adj.data[0] == 1
    assert weighted_adj.data[1] == 1
    assert weighted_adj.data[2] == 2

    g = Graph.from_edge_index(torch.tensor([[0], [1]]), num_nodes=5)
    adj = g.sparse_adj_matrix()
    assert adj.shape == (5, 5)
    assert adj.nnz == 1

    g.data.edge_attr = torch.tensor([[1]])
    adj = g.sparse_adj_matrix("edge_attr")
    assert adj.shape == (5, 5)
    assert adj.nnz == 1


def test_degrees(simple_graph):
    in_degrees = simple_graph.degrees("in")
    assert in_degrees["a"] == 0
    assert in_degrees["b"] == 1
    assert in_degrees["c"] == 2

    out_degrees = simple_graph.degrees("out")
    assert out_degrees["a"] == 2
    assert out_degrees["b"] == 1
    assert out_degrees["c"] == 0


def test_in_degrees(simple_graph):
    in_degrees = simple_graph.in_degrees
    assert in_degrees["a"] == 0
    assert in_degrees["b"] == 1
    assert in_degrees["c"] == 2


def test_out_degrees(simple_graph):
    out_degrees = simple_graph.out_degrees
    assert out_degrees["a"] == 2
    assert out_degrees["b"] == 1
    assert out_degrees["c"] == 0


def test_laplacian(simple_graph):
    laplacian = simple_graph.laplacian()
    assert laplacian.shape == (3, 3)
    assert laplacian.nnz == 6
    assert isinstance(laplacian, s.coo_matrix)
    assert laplacian.data[0] == -1
    assert laplacian.data[1] == -1
    assert laplacian.data[2] == -1
    assert laplacian.data[3] == 2
    assert laplacian.data[4] == 1
    assert laplacian.data[5] == 0


def test_add_operator_complete_overlap():
    # complete overlap
    g1 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "c", "d"]))
    g2 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "c", "d"]))
    g = g1 + g2
    assert g.n == g1.n
    assert g.m == g1.m + g2.m
    assert torch.equal(g.data.edge_index, torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 2, 3, 2, 3]]))


def test_add_operator_no_overlap():
    # no overlap
    g1 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "c", "d"]))
    g2 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["e", "f", "g", "h"]))
    g = g1 + g2
    assert g.n == g1.n + g2.n
    assert g.m == g1.m + g2.m
    assert torch.equal(g.data.edge_index, torch.tensor([[0, 1, 1, 4, 5, 5], [1, 2, 3, 5, 6, 7]]))


def test_add_operator_partial_overlap():
    # partial overlap
    g1 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "c", "d"]))
    g2 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "g", "h"]))
    g = g1 + g2
    assert g.n == 6
    assert g.m == g1.m + g2.m
    assert torch.equal(g.data.edge_index, torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 2, 3, 4, 5]]))


def test_get_node_attr(simple_graph):
    simple_graph.data["node_class"] = torch.tensor([[1], [2], [3]])
    assert simple_graph["node_class"].shape == (3, 1)
    assert torch.equal(simple_graph["node_class"], torch.tensor([[1], [2], [3]]))
    assert simple_graph["node_class", "a"].item() == 1
    assert simple_graph["node_class", "b"].item() == 2
    assert simple_graph["node_class", "c"].item() == 3

    with pytest.raises(KeyError):
        simple_graph["node_class", "d"].item()

    with pytest.raises(KeyError):
        simple_graph["node_class_1", "a"].item()


def test_get_edge_attr(simple_graph):
    # Edge indices are sorted during initialization
    # Potentially leads to wrong weight assignment
    simple_graph.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    assert simple_graph["edge_weight"].shape == (3, 1)
    assert torch.equal(simple_graph["edge_weight"], torch.tensor([[1], [1], [2]]))
    assert simple_graph["edge_weight", "a", "b"].item() == 1
    assert simple_graph["edge_weight", "b", "c"].item() == 2
    assert simple_graph["edge_weight", "a", "c"].item() == 1

    with pytest.raises(KeyError):
        simple_graph["edge_weight", "a", "d"].item()

    with pytest.raises(KeyError):
        simple_graph["edge_weight_1", "a", "b"].item()


def test_get_graph_attr(simple_graph):
    simple_graph.data["graph_feature"] = torch.tensor([42])
    assert simple_graph["graph_feature"].item() == 42


def test_set_node_attr(simple_graph):
    simple_graph["node_class"] = torch.tensor([[1], [2], [3]])
    assert simple_graph["node_class"].shape == (3, 1)
    assert torch.equal(simple_graph["node_class"], torch.tensor([[1], [2], [3]]))
    assert simple_graph["node_class", "a"].item() == 1
    assert simple_graph["node_class", "b"].item() == 2
    assert simple_graph["node_class", "c"].item() == 3

    simple_graph["node_class", "a"] = 42
    assert simple_graph["node_class", "a"].item() == 42

    with pytest.raises(KeyError):
        simple_graph["node_class", "d"] = 42

    with pytest.raises(KeyError):
        simple_graph["node_class_1", "a"] = 42


def test_set_edge_attr(simple_graph):
    simple_graph["edge_weight"] = torch.tensor([[1], [1], [2]])
    assert simple_graph["edge_weight"].shape == (3, 1)
    assert torch.equal(simple_graph["edge_weight"], torch.tensor([[1], [1], [2]]))
    assert simple_graph["edge_weight", "a", "b"].item() == 1
    assert simple_graph["edge_weight", "a", "c"].item() == 1
    assert simple_graph["edge_weight", "b", "c"].item() == 2

    simple_graph["edge_weight", "a", "b"] = 42
    assert simple_graph["edge_weight", "a", "b"].item() == 42

    with pytest.raises(KeyError):
        simple_graph["edge_weight", "a", "d"] = 42

    with pytest.raises(KeyError):
        simple_graph["edge_weight_1", "a", "b"] = 42


def test_set_graph_attr(simple_graph):
    simple_graph["graph_feature"] = torch.tensor([42])
    assert simple_graph["graph_feature"].item() == 42

    with pytest.raises(KeyError):
        simple_graph["graph_feature", "a"] = 42


def test_N(simple_graph):
    assert simple_graph.n == 3


def test_M(simple_graph):
    assert simple_graph.m == 3


def test_is_directed(simple_graph):
    assert simple_graph.is_directed()


def test_is_undirected(simple_graph):
    assert not simple_graph.is_undirected()


def test_has_self_loops(simple_graph):
    assert not simple_graph.has_self_loops()


def test_str(simple_graph):
    assert isinstance(str(simple_graph), str)
