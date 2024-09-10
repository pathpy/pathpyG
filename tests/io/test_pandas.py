"""This module tests high-level functions of the pandas module."""

import pytest

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.io.pandas import df_to_graph, df_to_temporal_graph

def test_df_to_graph(df_graph):
    g: Graph = df_to_graph(df_graph)
    assert g.N == 3
    assert g.M == 3

def test_df_to_graph(df_graph_attribute):
    g: Graph = df_to_graph(df_graph_attribute)
    assert g.N == 3
    assert g.M == 3

def test_df_to_temporal_graph(df_temporal_graph):
    g: TemporalGraph = df_to_graph(df_temporal_graph)
    assert g.N == 3
    assert g.M == 3
