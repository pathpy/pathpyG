from __future__ import annotations

import pytest
import torch

from pathpyG.core.IndexMap import IndexMap


@pytest.fixture
def simple_paths() -> WalkData:
    """Return a simple example for path data."""
    paths = WalkData(mapping=IndexMap(['A', 'B', 'C', 'D', 'E']))
    paths.add_walk_seq(('A', 'C', 'D'), freq=2)
    paths.add_walk_seq(('B', 'C', 'E'), freq=2)
    return paths
