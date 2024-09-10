from __future__ import annotations

import pytest

from pathpyG.core.Graph import Graph
import pandas as pd

@pytest.fixture
def df_graph() -> pd.DataFrame:
    """DataFrame for simple graph with header and no edge attributes."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a']})
    return df

@pytest.fixture
def df_graph_attribute() -> pd.DataFrame:
    """DataFrame for simple graph with edge attributes and header."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a'],
        'edge_weight': [2.0, 1.0, 42.0]})
    return df

@pytest.fixture
def df_graph_attribute_no_header() -> pd.DataFrame:
    """DataFrame for simple graph with edge attributes and no header."""
    df = pd.DataFrame([['a', 'b', 2.0], ['b', 'c', 1.0], ['c', 'a', 42.0]])
    return df

@pytest.fixture
def df_temporal_graph() -> pd.DataFrame:
    """DataFrame for simple temporal graph with header."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a'],
        't': [1, 2, 3]})
    return df


@pytest.fixture
def df_temporal_graph_no_header() -> pd.DataFrame:
    """DataFrame for simple temporal graph without header."""
    df = pd.DataFrame([['a', 'b', 1], ['b', 'c', 2], ['c', 'a', 3]])
    return df
