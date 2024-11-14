"""This module tests high-level functions of the pandas module."""

import pytest

from torch import tensor, equal
import numpy as np

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.io.pandas import df_to_graph, df_to_temporal_graph


def test_df_to_graph(df_graph, df_graph_attribute, df_graph_attribute_no_header):
    g: Graph = df_to_graph(df_graph)
    assert g.n == 3
    assert g.m == 3

    g: Graph = df_to_graph(df_graph_attribute)
    assert g.n == 3
    assert g.m == 3
    assert "edge_weight" in g.edge_attrs()
    assert equal(g.data.edge_weight, tensor([2.0, 1.0, 42.0]))

    g: Graph = df_to_graph(df_graph_attribute_no_header)
    assert g.n == 3
    assert g.m == 3
    assert "edge_attr_0" in g.edge_attrs()
    assert equal(g.data.edge_attr_0, tensor([2.0, 1.0, 42.0]))


def test_df_to_temporal_graph(df_temporal_graph, df_temporal_graph_no_header):
    g: TemporalGraph = df_to_temporal_graph(df_temporal_graph)
    assert g.n == 3
    assert g.m == 3
    assert equal(g.data.time, tensor([1.0, 2.0, 3.0]))

    g: TemporalGraph = df_to_temporal_graph(df_temporal_graph_no_header)
    assert g.n == 3
    assert g.m == 3
    assert equal(g.data.time, tensor([1.0, 2.0, 3.0]))
