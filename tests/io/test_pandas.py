"""This module tests high-level functions of the pandas module."""

# pylint: disable=missing-function-docstring

import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from pathpyG import Graph, TemporalGraph
from pathpyG.io.pandas import (
    _iterable_re,
    _number_re,
    _integer_re,
    _parse_timestamp,
    _parse_df_column,
    df_to_graph,
    add_edge_attributes,
    add_node_attributes,
    df_to_temporal_graph,
    graph_to_df,
    temporal_graph_to_df,
    read_csv_graph,
    read_csv_temporal_graph,
    write_csv,
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


def test_parse_timestamp_object_string():
    df = pd.DataFrame({"t": ["2023-01-01 12:00:00", "2023-01-01 13:00:00"]})
    _parse_timestamp(df)
    # Should be int64 after conversion
    assert np.issubdtype(df["t"].dtype, np.integer)
    assert df["t"].iloc[1] > df["t"].iloc[0]


def test_parse_timestamp_object_string_with_format():
    df = pd.DataFrame({"t": ["01/01/2023 12:00", "01/01/2023 13:00"]})
    _parse_timestamp(df, timestamp_format="%d/%m/%Y %H:%M")
    assert np.issubdtype(df["t"].dtype, np.integer)
    assert df["t"].iloc[1] > df["t"].iloc[0]


def test_parse_timestamp_int64():
    df = pd.DataFrame({"t": [1000, 2000, 3000]})
    _parse_timestamp(df)
    assert np.all(df["t"] == np.array([1000, 2000, 3000]))


def test_parse_timestamp_float64():
    df = pd.DataFrame({"t": [1000.0, 2000.0, 3000.0]})
    _parse_timestamp(df)
    assert np.all(df["t"] == np.array([1000.0, 2000.0, 3000.0]))


def test_parse_timestamp_datetime64():
    df = pd.DataFrame({"t": pd.to_datetime(["2023-01-01", "2023-01-02"])})
    _parse_timestamp(df)
    assert np.issubdtype(df["t"].dtype, np.integer)
    assert df["t"].iloc[1] > df["t"].iloc[0]


def test_parse_timestamp_rescale():
    df = pd.DataFrame({"t": ["2023-01-01 12:00:00", "2023-01-01 13:00:00"]})
    _parse_timestamp(df, time_rescale=10**9)  # convert to seconds
    # Should be seconds since epoch
    assert np.all(df["t"].diff().dropna() == 3600)


def test_parse_timestamp_invalid_type():
    df = pd.DataFrame({"t": [None, None]})
    with pytest.raises(ValueError, match="Column `t` must be of type"):
        _parse_timestamp(df)


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
    df = pd.DataFrame({"v": ["b", "a", "c"], "x": [2, 1, 3], "node_y": [0.2, 0.1, 0.3]})
    add_node_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["node_x"], torch.tensor([1, 2, 3]))
    assert torch.allclose(simple_graph.data["node_y"], torch.tensor([0.1, 0.2, 0.3], dtype=torch.double))


def test_add_node_attributes_by_index(simple_graph):
    df = pd.DataFrame({"index": [1, 0, 2], "x": [20, 10, 30]})
    add_node_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["node_x"], torch.tensor([10, 20, 30]))


def test_duplicate_node_attribute_raises(simple_graph):
    df = pd.DataFrame(
        {
            "v": ["a", "a", "b", "c"],
            "x": [1, 2, 3, 4],
        }
    )
    with pytest.raises(ValueError, match="multiple attribute values for single node"):
        add_node_attributes(df, simple_graph)


def test_mismatch_nodes_raises(simple_graph):
    df = pd.DataFrame({"v": ["a", "b", "d"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="Mismatch between nodes"):
        add_node_attributes(df, simple_graph)


def test_missing_v_and_index_raises(simple_graph):
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    with pytest.raises(ValueError, match="must either have `index` or `v` column"):
        add_node_attributes(df, simple_graph)


def test_add_edge_attributes_basic(simple_graph):
    df = pd.DataFrame({"v": ["a", "b", "a"], "w": ["b", "c", "c"], "weight": [1, 3, 2]})
    add_edge_attributes(df, simple_graph)
    assert torch.allclose(simple_graph.data["edge_weight"], torch.tensor([1, 2, 3]))


def test_add_edge_attributes_with_prefix(simple_graph):
    df = pd.DataFrame({"v": ["a", "b", "a"], "w": ["b", "c", "c"], "edge_score": [5, 6, 7]})
    add_edge_attributes(df, simple_graph)
    assert torch.equal(simple_graph.data["edge_score"], torch.tensor([5, 7, 6]))


def test_add_edge_attributes_missing_edge_raises(simple_graph):
    df = pd.DataFrame(
        {"v": ["a", "x", "a"], "w": ["b", "c", "c"], "weight": [1.0, 2.0, 3.0]}  # "x" does not exist in graph
    )
    with pytest.raises(ValueError, match="Please ensure all nodes in the DataFrame are present in the graph."):
        add_edge_attributes(df, simple_graph)

    df = pd.DataFrame(
        {"v": ["a", "b", "a"], "w": ["a", "c", "c"], "weight": [1.0, 2.0, 3.0]}  # edge "a -> a" does not exist in graph
    )
    with pytest.raises(ValueError, match="does not exist in the graph"):
        add_edge_attributes(df, simple_graph)


def test_add_edge_attributes_temporal(simple_temporal_graph):
    df = pd.DataFrame({"v": ["a", "b", "c", "c"], "w": ["b", "c", "e", "d"], "t": [1, 5, 9, 9], "weight": [1, 2, 4, 3]})
    add_edge_attributes(df, simple_temporal_graph, time_attr="t")
    assert torch.allclose(simple_temporal_graph.data["edge_weight"], torch.tensor([1, 2, 3, 4]))


def test_add_edge_attributes_temporal_to_few_edges(simple_temporal_graph):
    df = pd.DataFrame({"v": ["a"], "w": ["b"], "t": [99], "weight": [1.0]})  # No such temporal edge
    with pytest.raises(ValueError, match="Please ensure the DataFrame matches the number of edges in the graph"):
        add_edge_attributes(df, simple_temporal_graph, time_attr="t")


def test_add_edge_attributes_temporal_missing_raises(simple_temporal_graph):
    df = pd.DataFrame(
        {
            "v": ["a", "b", "c", "c"],
            "w": ["b", "c", "d", "e"],
            "t": [1, 5, 9, 10],  # Time "10" does not exist in the graph
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="does not exist at time"):
        add_edge_attributes(df, simple_temporal_graph, time_attr="t")


def test_df_to_temporal_graph_basic():
    df = pd.DataFrame({"v": ["a", "b", "c"], "w": ["b", "c", "a"], "t": [1, 2, 3]})
    g = df_to_temporal_graph(df)
    assert g.n == 3
    assert g.m == 3
    assert torch.equal(g.data.time, torch.tensor([1, 2, 3]))


def test_df_to_temporal_graph_with_edge_attr():
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "t": [20, 10], "weight": [2.0, 1.0]})
    g = df_to_temporal_graph(df)
    assert hasattr(g.data, "edge_weight")
    # edge weights should be in the same order as edges (sorted by time)
    assert torch.allclose(g.data.edge_weight, torch.tensor([1.0, 2.0], dtype=torch.double))


def test_df_to_temporal_graph_multiedges_false_removes_duplicates():
    df = pd.DataFrame({"v": ["a", "a", "b"], "w": ["b", "b", "c"], "t": [1, 1, 2]})
    g = df_to_temporal_graph(df, multiedges=False)
    assert g.m == 2  # duplicate (a, b, 1) should be removed


def test_df_to_temporal_graph_multiedges_true_keeps_duplicates():
    df = pd.DataFrame({"v": ["a", "a", "b"], "w": ["b", "b", "c"], "t": [1, 1, 2]})
    g = df_to_temporal_graph(df, multiedges=True)
    assert g.m == 3  # duplicate (a, b, 1) is kept


def test_df_to_temporal_graph_no_header():
    df = pd.DataFrame([["a", "b", 1], ["b", "c", 2], ["c", "a", 3], ["a", "b", 4]])
    g = df_to_temporal_graph(df)
    assert g.n == 3
    assert g.m == 4


def test_df_to_temporal_graph_time_rescale():
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "t": [1000, 2000]})
    g = df_to_temporal_graph(df, time_rescale=1000)
    print(g.data.time)
    assert torch.equal(g.data.time, torch.tensor([1, 2]))


def test_df_to_temporal_graph_with_extra_edge_attrs():
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "t": [1, 2], "foo": [10, 20], "edge_bar": [0.1, 0.2]})
    g = df_to_temporal_graph(df)
    assert hasattr(g.data, "edge_foo")
    assert hasattr(g.data, "edge_bar")
    assert torch.equal(g.data.edge_foo, torch.tensor([10, 20]))
    assert torch.allclose(g.data.edge_bar, torch.tensor([0.1, 0.2], dtype=torch.double))


def test_graph_to_df_basic(simple_graph):
    df = graph_to_df(simple_graph)
    assert set(df.columns) == {"v", "w"}
    assert len(df) == 3
    assert set(df["v"]) == {"a", "b"}
    assert set(df["w"]) == {"b", "c"}


def test_graph_to_df_with_edge_attr(simple_graph):
    simple_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
    df = graph_to_df(simple_graph)
    assert "edge_weight" in df.columns
    assert list(df["edge_weight"]) == [1.0, 2.0, 3.0]


def test_graph_to_df_node_indices(simple_graph):
    df = graph_to_df(simple_graph, node_indices=True)
    assert set(df.columns) == {"v", "w"}
    assert set(df["v"]) == {0, 1}
    assert set(df["w"]) == {1, 2}


def test_graph_to_df_with_multiple_edge_attrs(simple_graph):
    simple_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
    simple_graph.data.edge_label = torch.tensor([0, 1, 2])
    df = graph_to_df(simple_graph)
    assert "edge_weight" in df.columns
    assert "edge_label" in df.columns
    assert list(df["edge_label"]) == [0, 1, 2]


def test_temporal_graph_to_df_basic(simple_temporal_graph):
    df = temporal_graph_to_df(simple_temporal_graph)
    assert set(df.columns) == {"v", "w", "t"}
    assert len(df) == 4
    assert set(df["v"]) == {"a", "b", "c"}
    assert set(df["w"]) == {"b", "c", "d", "e"}
    assert set(df["t"]) == {1, 5, 9}


def test_temporal_graph_to_df_with_edge_attr(simple_temporal_graph):
    simple_temporal_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    df = temporal_graph_to_df(simple_temporal_graph)
    assert "edge_weight" in df.columns
    assert list(df["edge_weight"]) == [1.0, 2.0, 3.0, 4.0]


def test_temporal_graph_to_df_node_indices(simple_temporal_graph):
    df = temporal_graph_to_df(simple_temporal_graph, node_indices=True)
    assert set(df.columns) == {"v", "w", "t"}
    assert set(df["v"]) == {0, 1, 2}
    assert set(df["w"]) == {1, 2, 3, 4}
    assert set(df["t"]) == {1, 5, 9}


def test_temporal_graph_to_df_with_multiple_edge_attrs(simple_temporal_graph):
    simple_temporal_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    simple_temporal_graph.data.edge_label = torch.tensor([0, 1, 0, 1])
    df = temporal_graph_to_df(simple_temporal_graph)
    assert "edge_weight" in df.columns
    assert "edge_label" in df.columns
    assert list(df["edge_label"]) == [0, 1, 0, 1]


def test_read_csv_graph_basic(tmp_path):
    # Create a simple CSV file
    csv_path = tmp_path / "graph.csv"
    df = pd.DataFrame({"v": ["a", "b", "a"], "w": ["b", "c", "c"]})
    df.to_csv(csv_path, index=False)
    g = read_csv_graph(str(csv_path))
    assert isinstance(g, Graph)
    assert g.n == 3
    assert g.m == 3
    assert set(g.nodes) == {"a", "b", "c"}


def test_read_csv_graph_with_edge_attr(tmp_path):
    csv_path = tmp_path / "graph_attr.csv"
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "edge_weight": [1.0, 2.0]})
    df.to_csv(csv_path, index=False)
    g = read_csv_graph(str(csv_path))
    assert hasattr(g.data, "edge_weight")
    assert torch.allclose(g.data.edge_weight, torch.tensor([1.0, 2.0], dtype=torch.double))


def test_read_csv_graph_no_header(tmp_path):
    csv_path = tmp_path / "graph_noheader.csv"
    df = pd.DataFrame([["a", "b"], ["b", "c"], ["a", "c"]])
    df.to_csv(csv_path, index=False, header=False)
    g = read_csv_graph(str(csv_path), header=False)
    assert g.n == 3
    assert g.m == 3
    assert set(g.nodes) == {"a", "b", "c"}


def test_read_csv_graph_multiedges(tmp_path):
    csv_path = tmp_path / "graph_multi.csv"
    df = pd.DataFrame({"v": ["a", "a", "b"], "w": ["b", "b", "c"]})
    df.to_csv(csv_path, index=False)
    g = read_csv_graph(str(csv_path), multiedges=False)
    assert g.m == 2  # duplicate (a, b) should be removed
    g2 = read_csv_graph(str(csv_path), multiedges=True)
    assert g2.m == 3  # all edges kept


def test_read_csv_temporal_graph_basic(tmp_path):
    csv_path = tmp_path / "temporal_graph.csv"
    df = pd.DataFrame({"v": ["a", "b", "c", "c"], "w": ["b", "c", "d", "e"], "t": [1, 5, 9, 9]})
    df.to_csv(csv_path, index=False)
    g = read_csv_temporal_graph(str(csv_path))
    assert isinstance(g, TemporalGraph)
    assert g.n == 5
    assert g.m == 4
    assert set(g.nodes) == {"a", "b", "c", "d", "e"}
    assert set(g.data.time.tolist()) == {0, 4, 8, 8} or set(g.data.time.tolist()) == {1, 5, 9}


def test_read_csv_temporal_graph_with_edge_attr(tmp_path):
    csv_path = tmp_path / "temporal_graph_attr.csv"
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "t": [1, 2], "edge_weight": [1.0, 2.0]})
    df.to_csv(csv_path, index=False)
    g = read_csv_temporal_graph(str(csv_path))
    assert hasattr(g.data, "edge_weight")
    assert torch.allclose(g.data.edge_weight, torch.tensor([1.0, 2.0], dtype=torch.double))


def test_read_csv_temporal_graph_no_header(tmp_path):
    csv_path = tmp_path / "temporal_graph_noheader.csv"
    df = pd.DataFrame([["a", "b", 1], ["b", "c", 2], ["c", "d", 3]])
    df.to_csv(csv_path, index=False, header=False)
    g = read_csv_temporal_graph(str(csv_path), header=False)
    assert g.n == 4
    assert g.m == 3
    assert set(g.nodes) == {"a", "b", "c", "d"}


def test_read_csv_temporal_graph_time_rescale(tmp_path):
    csv_path = tmp_path / "temporal_graph_rescale.csv"
    df = pd.DataFrame({"v": ["a", "b"], "w": ["b", "c"], "t": [1000, 2000]})
    df.to_csv(csv_path, index=False)
    g = read_csv_temporal_graph(str(csv_path), time_rescale=1000)
    assert torch.equal(g.data.time, torch.tensor([1, 2]))


def test_write_csv_graph_and_read(tmp_path, simple_graph):
    # Create a simple graph
    simple_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
    csv_path = tmp_path / "graph.csv"
    write_csv(simple_graph, path_or_buf=csv_path)
    # Read back and check content
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"v", "w", "edge_weight"}
    assert len(df) == 3
    assert set(df["v"]) == {"a", "b"}
    assert set(df["w"]) == {"b", "c"}
    assert list(df["edge_weight"]) == [1.0, 2.0, 3.0]


def test_write_csv_temporal_graph_and_read(tmp_path, simple_temporal_graph):
    simple_temporal_graph.data.edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    csv_path = tmp_path / "temporal_graph.csv"
    write_csv(simple_temporal_graph, path_or_buf=csv_path)
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"v", "w", "t", "edge_weight"}
    assert len(df) == 4
    assert set(df["v"]) == {"a", "b", "c"}
    assert set(df["w"]) == {"b", "c", "d", "e"}
    assert set(df["t"]) == {1, 5, 9}
    assert list(df["edge_weight"]) == [1.0, 2.0, 3.0, 4.0]


def test_write_csv_with_node_indices(tmp_path, simple_graph):
    csv_path = tmp_path / "graph_indices.csv"
    write_csv(simple_graph, node_indices=True, path_or_buf=csv_path)
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"v", "w"}
    assert set(df["v"]) == {0, 1}
    assert set(df["w"]) == {1, 2}
