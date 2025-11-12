import numpy as np
import pytest

from pathpyG.core.index_map import IndexMap


@pytest.mark.gpu
def test_index_mapping_bulk(gpu):
    mapping = IndexMap()
    mapping.add_ids(["a", "b", "c", "d", "e"])
    assert mapping.to_idxs(np.array(["a", "b", "c", "d", "e"]), device=gpu).device == gpu
