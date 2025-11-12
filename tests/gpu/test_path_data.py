from __future__ import annotations

import pytest

from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData


@pytest.mark.gpu
def test_to_device(gpu, cpu):
    paths = PathData(IndexMap(["a", "c", "b", "d", "e"]), device=gpu)
    paths.append_walks([("a", "c", "d"), ("a", "c", "e"), ("b", "c", "d"), ("b", "c", "e")], weights=[1.0] * 4)
    assert paths.data.dag_weight.device == gpu

    paths.to(cpu)
    assert paths.data.dag_weight.device == cpu
