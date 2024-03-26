from __future__ import annotations

import pytest
import torch

from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.DAGData import DAGData


@pytest.fixture
def simple_dags() -> DAGData:
    """Return a simple example for path data."""
    dags = DAGData(mapping=IndexMap(['A', 'B', 'C', 'D', 'E']))
    dags.append_walk(('A', 'C', 'D'), weight=2.0)
    dags.append_walk(('B', 'C', 'E'), weight=2.0)
    return dags
