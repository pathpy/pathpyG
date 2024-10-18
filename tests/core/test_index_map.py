from __future__ import annotations

import torch
import numpy as np

from pathpyG.core.index_map import IndexMap


def test_index_mapping():
    mapping = IndexMap()

    assert mapping.to_idx(0) == 0
    assert mapping.to_idx(42) == 42

    assert mapping.to_id(0) == 0
    assert mapping.to_id(42) == 42

    assert mapping.to_ids([0, 1, 2]) == [0, 1, 2]
    assert (mapping.to_idxs([0, 1, 2]) == torch.tensor([0, 1, 2])).all()

    mapping.add_id("a")

    assert mapping.to_idx("a") == 0
    assert mapping.to_id(0) == "a"
    assert mapping.num_ids() == 1
    assert mapping.node_ids == ["a"]

    mapping.add_id("a")

    assert mapping.num_ids() == 1
    assert mapping.node_ids == ["a"]

    mapping.add_id("c")

    assert mapping.to_idx("c") == 1
    assert mapping.to_id(1) == "c"
    assert mapping.num_ids() == 2
    assert (mapping.node_ids == ["a", "c"]).all()


def test_index_mapping_bulk():
    mapping = IndexMap()

    mapping.add_ids(["a", "b", "c", "d", "e"])
    assert mapping.num_ids() == 5
    assert (mapping.node_ids == ["a", "b", "c", "d", "e"]).all()
    assert mapping.to_idxs(["a", "b", "c", "d", "e"]).tolist() == [0, 1, 2, 3, 4]
    assert mapping.to_ids([0, 1, 2, 3, 4]) == ["a", "b", "c", "d", "e"]

    mapping.add_ids(("a", "a", "f", "f"))
    assert mapping.num_ids() == 6
    assert (mapping.node_ids == ["a", "b", "c", "d", "e", "f"]).all()
    assert mapping.to_idxs(["a", "b", "c", "d", "e", "f"]).tolist() == [0, 1, 2, 3, 4, 5]
    assert mapping.to_ids([0, 1, 2, 3, 4, 5]) == ["a", "b", "c", "d", "e", "f"]

    mapping.add_id("a")
    assert mapping.num_ids() == 6
    assert (mapping.node_ids == ["a", "b", "c", "d", "e", "f"]).all()
    assert mapping.to_idxs(("a", "b", "c", "d", "e", "f")).tolist() == [0, 1, 2, 3, 4, 5]
    assert mapping.to_ids(torch.tensor([0, 1, 2, 3, 4, 5])) == ["a", "b", "c", "d", "e", "f"]
    assert mapping.to_idxs(np.array(["a", "b", "c", "d", "e", "f"])).tolist() == [0, 1, 2, 3, 4, 5]


def test_integer_ids():
    mapping = IndexMap([0, 2, 3, 1, 4])

    assert mapping.to_idx(0) == 0
    assert mapping.to_idx(1) == 3

    assert mapping.to_id(0) == 0
    assert mapping.to_id(3) == 1

    assert mapping.to_ids([0, 1, 2]) == [0, 2, 3]
    assert (mapping.to_idxs([0, 1, 2]) == torch.tensor([0, 3, 1])).all()


def test_tuple_ids():
    mapping = IndexMap([(1, 2), (3, 4), (5, 6)])

    assert mapping.to_idx((1, 2)) == 0
    assert mapping.to_idx((3, 4)) == 1

    assert mapping.to_id(0) == (1, 2)
    assert mapping.to_id(1) == (3, 4)


def test_float_ids():
    mapping = IndexMap([0.0, 2.0, 3.0, 1.0, 4.0])
    mapping.add_id(1.5)
    mapping.add_ids(np.array([8.0, 9.0]))

    assert mapping.to_idx(0.0) == 0
    assert mapping.to_idx(1.0) == 3

    assert mapping.to_id(0) == 0.0
    assert mapping.to_id(3) == 1.0

    assert mapping.to_ids([0, 1, 7]) == [0.0, 2.0, 9.0]
    assert (mapping.to_idxs([0.0, 1.0, 2.0]) == torch.tensor([0, 3, 1])).all()
