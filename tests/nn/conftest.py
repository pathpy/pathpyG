from __future__ import annotations

import pytest
import torch

from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.PathData import PathData


@pytest.fixture
def simple_paths() -> PathData:
    """Return a simple example for path data."""
    paths = PathData()
    paths.add_walk(torch.tensor([[0, 2], [2, 3]]))  # A -> C -> D
    paths.add_walk(torch.tensor([[0, 2], [2, 3]]))  # A -> C -> D
    paths.add_walk(torch.tensor([[1, 2], [2, 4]]))  # B -> C -> E
    paths.add_walk(torch.tensor([[1, 2], [2, 4]]))  # B -> C -> E
    paths.mapping = IndexMap(['A', 'B', 'C', 'D', 'E'])
    return paths