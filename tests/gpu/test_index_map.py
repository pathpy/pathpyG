
import pytest
import numpy as np

from pathpyG.core.IndexMap import IndexMap


@pytest.mark.gpu
def test_index_mapping_bulk(gpu):
    mapping = IndexMap()
    mapping.add_ids(['a', 'b', 'c', 'd', 'e'])
    assert mapping.to_idxs(np.array(['a', 'b', 'c', 'd', 'e']), device=gpu).device == gpu