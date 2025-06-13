"""This module tests high-level functions of the pandas module."""

# pylint: disable=missing-function-docstring

import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from pathpyG import Graph
from pathpyG.io.pandas import (
    _iterable_re,
    _number_re,
    _integer_re,
    _parse_df_column,
    df_to_graph,
    add_edge_attributes,
    add_node_attributes,
    df_to_temporal_graph,
)


def test_iterable_regex():
    assert _iterable_re.match("[1, 2, 3]")
    assert _iterable_re.match("(1, 2, 3)")
    assert not _iterable_re.match("{1, 2, 3}")
    assert not _iterable_re.match("1, 2, 3")
    assert not _iterable_re.match("1, 2, 3]")
    assert not _iterable_re.match("(1, 2, 3")
    assert _iterable_re.match("[[1, 2], [3, 4]]")


def test_number_regex():
    assert _number_re.match("1")
    assert _number_re.match("1.0")
    assert _number_re.match("1.0e10")
    assert not _number_re.match("1,000")
    assert not _number_re.match("one")
    assert not _number_re.match("1.0.0")


def test_integer_regex():
    assert _integer_re.match("1")
    assert _integer_re.match("1000")
    assert not _integer_re.match("1.0")
    assert not _integer_re.match("1.0e10")
    assert not _integer_re.match("1,000")
    assert not _integer_re.match("one")
    assert not _integer_re.match("1.0.0")


def test_parse_df_column_numeric(backward_idx):
    df = pd.DataFrame({"attr": [1, 2, 3]})
    data = Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    _parse_df_column(df, data, "attr")
    assert torch.equal(data["attr"], torch.tensor([1, 2, 3]))

    df = pd.DataFrame({"attr": ["1", "2", "3"]})
    _parse_df_column(df, data, "attr", prefix="edge_")
    assert torch.equal(data["edge_attr"], torch.tensor([1, 2, 3], device=data.edge_index.device))

    _parse_df_column(df, data, "attr", prefix="node_", idx=backward_idx)
    expected_idx = torch.tensor([3, 2, 1])
    assert torch.equal(data["node_attr"], expected_idx)


def test_parse_df_column_float(backward_idx):
    df = pd.DataFrame({"attr": [1.1, 2.2, 3.3]})
    data = Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    _parse_df_column(df, data, "attr")
    assert torch.allclose(data["attr"], torch.tensor([1.1, 2.2, 3.3], dtype=torch.double))

    df = pd.DataFrame({"attr": ["1.1", "2.2", "3.3"]})
    _parse_df_column(df, data, "attr", prefix="node_")
    assert torch.allclose(
        data["node_attr"], torch.tensor([1.1, 2.2, 3.3], dtype=torch.double, device=data.edge_index.device)
    )

    _parse_df_column(df, data, "attr", prefix="edge_", idx=backward_idx)
    expected_idx = torch.tensor([3.3, 2.2, 1.1], dtype=torch.double)
    assert torch.allclose(data["edge_attr"], expected_idx)


def test_parse_df_column_iterable(backward_idx):
    df = pd.DataFrame({"attr": [[1, 2], [3, 4], [5, 6]]})
    data = Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    _parse_df_column(df, data, "attr")
    expected = torch.tensor([[1, 2], [3, 4], [5, 6]])
    assert torch.equal(data["attr"], expected)

    df = pd.DataFrame({"attr": ["[1, 2]", "[3, 4]", "[5, 6]"]})
    _parse_df_column(df, data, "attr", prefix="edge_")
    assert torch.equal(data["edge_attr"], expected)

    df = pd.DataFrame({"attr": [(1, 2), (3, 4), (5, 6)]})
    _parse_df_column(df, data, "attr", prefix="node_", idx=backward_idx)
    expected_idx = torch.tensor([[5, 6], [3, 4], [1, 2]])
    assert torch.equal(data["node_attr"], expected_idx)


def test_parse_df_column_string(backward_idx):
    df = pd.DataFrame({"attr": ["foo", "bar", "baz"]})
    data = Data(edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    _parse_df_column(df, data, "attr")
    assert np.array_equal(data["attr"], np.array(["foo", "bar", "baz"]))

    _parse_df_column(df, data, "attr", prefix="edge_", idx=backward_idx)
    expected_idx = np.array(["baz", "bar", "foo"])
    assert np.array_equal(data["edge_attr"], expected_idx)


def test_df_to_graph():
    df_graph = pd.DataFrame({"v": ["a", "b", "c"], "w": ["b", "c", "a"]})
    g = df_to_graph(df_graph)
    assert g.n == 3
    assert g.m == 3

    df_graph_attribute = pd.DataFrame({"v": ["a", "b", "c"], "w": ["b", "c", "a"], "edge_weight": [2.0, 1.0, 42.0]})
    g = df_to_graph(df_graph_attribute)
    assert g.n == 3
    assert g.m == 3
    assert "edge_weight" in g.edge_attrs()
    assert torch.equal(g.data.edge_weight, torch.tensor([2.0, 1.0, 42.0]))

    df_graph_attribute_no_header = pd.DataFrame([["a", "b", 2.0], ["b", "c", 1.0], ["c", "a", 42.0]])
    g = df_to_graph(df_graph_attribute_no_header)
    assert g.n == 3
    assert g.m == 3
    assert "edge_attr_0" in g.edge_attrs()
    assert torch.equal(g.data.edge_attr_0, torch.tensor([2.0, 1.0, 42.0]))

    df_graph_with_multi_edges = pd.DataFrame(
        {"v": ["a", "b", "c", "a"], "w": ["b", "c", "a", "b"], "edge_weight": [2.0, 1.0, 42.0, 3.0]}
    )
    g = df_to_graph(df_graph_with_multi_edges, multiedges=False)
    assert g.n == 3
    assert g.m == 3

    g = df_to_graph(df_graph_with_multi_edges, multiedges=True)
    assert g.n == 3
    assert g.m == 4

    df_graph_with_string_attr = pd.DataFrame(
        {"v": ["a", "b", "c"], "w": ["b", "c", "a"], "edge_weight": ["a", "b", "c"]}
    )
    g = df_to_graph(df_graph_with_string_attr, is_undirected=True)
    assert g.n == 3
    assert g.m == 6


def test_add_node_attributes_by_name(simple_graph):
    df = pd.DataFrame({
        "v": ["b", "a", "c"],
        "x": [2, 1, 3],
        "node_y": [0.2, 0.1, 0.3]
    })
    add_node_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["node_x"], torch.tensor([1, 2, 3]))
    assert torch.allclose(simple_graph.data["node_y"], torch.tensor([0.1, 0.2, 0.3], dtype=torch.double))


def test_add_node_attributes_by_index(simple_graph):
    df = pd.DataFrame({
        "index": [1, 0, 2],
        "x": [20, 10, 30]
    })
    add_node_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["node_x"], torch.tensor([10, 20, 30]))


def test_duplicate_node_attribute_raises(simple_graph):
    df = pd.DataFrame({
        "v": ["a", "a", "b", "c"],
        "x": [1, 2, 3, 4],
    })
    with pytest.raises(ValueError, match="multiple attribute values for single node"):
        add_node_attributes(df, simple_graph)


def test_mismatch_nodes_raises(simple_graph):
    df = pd.DataFrame({
        "v": ["a", "b", "d"],
        "x": [1, 2, 3]
    })
    with pytest.raises(ValueError, match="Mismatch between nodes"):
        add_node_attributes(df, simple_graph)


def test_missing_v_and_index_raises(simple_graph):
    df = pd.DataFrame({
        "foo": [1, 2, 3],
        "bar": [4, 5, 6]
    })
    with pytest.raises(ValueError, match="must either have `index` or `v` column"):
        add_node_attributes(df, simple_graph)


def test_add_edge_attributes_basic(simple_graph):
    df = pd.DataFrame({
        "v": ["a", "b", "a"],
        "w": ["b", "c", "c"],
        "weight": [1, 3, 2]
    })
    add_edge_attributes(df, simple_graph)
    assert torch.allclose(simple_graph.data["edge_weight"], torch.tensor([1, 2, 3]))


def test_add_edge_attributes_with_prefix(simple_graph):
    df = pd.DataFrame({
        "v": ["a", "b", "a"],
        "w": ["b", "c", "c"],
        "edge_score": [5, 6, 7]
    })
    add_edge_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["edge_score"], torch.tensor([5, 7, 6]))


def test_add_edge_attributes_missing_edge_raises(simple_graph):
    df = pd.DataFrame({
        "v": ["a", "x", "a"],  # "x" does not exist in graph
        "w": ["b", "c", "c"],
        "weight": [1.0, 2.0, 3.0]
    })
    with pytest.raises(ValueError, match="Please ensure all nodes in the DataFrame are present in the graph."):
        add_edge_attributes(df, simple_graph)

    df = pd.DataFrame({
        "v": ["a", "b", "a"],
        "w": ["a", "c", "c"],  # edge "a -> a" does not exist in graph
        "weight": [1.0, 2.0, 3.0]
    })
    with pytest.raises(ValueError, match="does not exist in the graph"):
        add_edge_attributes(df, simple_graph)


def test_add_edge_attributes_temporal(simple_temporal_graph):
    df = pd.DataFrame({
        "v": ["a", "b", "c", "c"],
        "w": ["b", "c", "e", "d"],
        "t": [1, 5, 9, 9],
        "weight": [1, 2, 4, 3]
    })
    add_edge_attributes(df, simple_temporal_graph, time_attr="t")
    assert torch.allclose(simple_temporal_graph.data["edge_weight"], torch.tensor([1, 2, 3, 4]))


def test_add_edge_attributes_temporal_to_few_edges(simple_temporal_graph):
    df = pd.DataFrame({
        "v": ["a"],
        "w": ["b"],
        "t": [99],  # No such temporal edge
        "weight": [1.0]
    })
    with pytest.raises(ValueError, match="Please ensure the DataFrame matches the number of edges in the graph"):
        add_edge_attributes(df, simple_temporal_graph, time_attr="t")


def test_add_edge_attributes_temporal_missing_raises(simple_temporal_graph):
    df = pd.DataFrame({
        "v": ["a", "b", "c", "c"],
        "w": ["b", "c", "d", "e"],
        "t": [1, 5, 9, 10],  # Time "10" does not exist in the graph
        "weight": [1.0, 2.0, 3.0, 4.0]
    })
    with pytest.raises(ValueError, match="does not exist at time"):
        add_edge_attributes(df, simple_temporal_graph, time_attr="t")


def test_df_to_temporal_graph():
    df_temporal_graph = pd.DataFrame({"v": ["a", "b", "c"], "w": ["b", "c", "a"], "t": [1, 2, 3]})
    g = df_to_temporal_graph(df_temporal_graph)
    assert g.n == 3
    assert g.m == 3
    assert torch.equal(g.data.time, torch.tensor([1.0, 2.0, 3.0]))

    df_temporal_graph_no_header = pd.DataFrame([["a", "b", 1], ["b", "c", 2], ["c", "a", 3]])
    g = df_to_temporal_graph(df_temporal_graph_no_header)
    assert g.n == 3
    assert g.m == 3
    assert torch.equal(g.data.time, torch.tensor([1.0, 2.0, 3.0]))
