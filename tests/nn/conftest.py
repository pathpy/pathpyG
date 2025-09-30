from __future__ import annotations

import pytest

from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData


@pytest.fixture
def simple_walks() -> PathData:
    """Return a simple example for path data."""
    paths = PathData(mapping=IndexMap(["A", "B", "C", "D", "E"]))
    paths.append_walk(("A", "C", "D"), weight=2.0)
    paths.append_walk(("B", "C", "E"), weight=2.0)
    return paths
