from __future__ import annotations

import pytest

from pathpyG.core.Graph import Graph
import pandas as pd

@pytest.fixture
def df_graph() -> pd.DataFrame:
    """Return a DataFrame for a simple graph."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a']})
    return df

@pytest.fixture
def df_graph_attribute() -> pd.DataFrame:
    """Return a DataFrame for a simple attributed graph."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a'],
        'weight': [2.0, 1.0, 42.0]})
    return df


@pytest.fixture
def df_temporal_graph() -> pd.DataFrame:
    """Return a DataFrame for a simple temporal graph."""
    df = pd.DataFrame({
        'v': ['a', 'b', 'c'],
        'w': ['b', 'c', 'a'],
        't': [1, 2, 3]})
    return df


