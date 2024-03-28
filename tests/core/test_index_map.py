from __future__ import annotations

import torch
import numpy as np

from pathpyG.core.IndexMap import IndexMap

def test_index_mapping():
    mapping = IndexMap()

    assert mapping.to_idx(0) == 0
    assert mapping.to_idx(42) == 42

    assert mapping.to_id(0) == 0
    assert mapping.to_id(42) == 42

    mapping.add_id('a')

    assert mapping.to_idx('a') == 0
    assert mapping.to_id(0) == 'a'
    assert mapping.num_ids() == 1
    assert mapping.node_ids == ['a']

    mapping.add_id('a')

    assert mapping.num_ids() == 1
    assert mapping.node_ids == ['a']

    mapping.add_id('c')

    assert mapping.to_idx('c') == 1
    assert mapping.to_id(1) == 'c'
    assert mapping.num_ids() == 2
    assert (mapping.node_ids == ['a', 'c']).all()


def test_index_mapping_bulk():
    mapping = IndexMap()

    mapping.add_ids(['a', 'b', 'c', 'd', 'e'])
    assert mapping.num_ids() == 5
    assert (mapping.node_ids == ['a', 'b', 'c', 'd', 'e']).all()
    assert mapping.to_idxs(['a', 'b', 'c', 'd', 'e']).tolist() == [0, 1, 2, 3, 4]
    assert mapping.to_ids([0, 1, 2, 3, 4]) == ['a', 'b', 'c', 'd', 'e']

    mapping.add_ids(('a', 'a', 'f', 'f'))
    assert mapping.num_ids() == 6
    assert (mapping.node_ids == ['a', 'b', 'c', 'd', 'e', 'f']).all()
    assert mapping.to_idxs(['a', 'b', 'c', 'd', 'e', 'f']).tolist() == [0, 1, 2, 3, 4, 5]
    assert mapping.to_ids([0, 1, 2, 3, 4, 5]) == ['a', 'b', 'c', 'd', 'e', 'f']

    mapping.add_id('a')
    assert mapping.num_ids() == 6
    assert (mapping.node_ids == ['a', 'b', 'c', 'd', 'e', 'f']).all()
    assert mapping.to_idxs(('a', 'b', 'c', 'd', 'e', 'f')).tolist() == [0, 1, 2, 3, 4, 5]
    assert mapping.to_ids(torch.tensor([0, 1, 2, 3, 4, 5])) == ['a', 'b', 'c', 'd', 'e', 'f']
    assert mapping.to_idxs(np.array(['a', 'b', 'c', 'd', 'e', 'f'])).tolist() == [0, 1, 2, 3, 4, 5]
