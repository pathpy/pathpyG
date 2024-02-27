# pylint: disable=missing-function-docstring,missing-module-docstring

from __future__ import annotations

from pathpyG.core.IndexMap import IndexMap


def test_index_mapping():
    mapping = IndexMap()

    assert mapping.to_idx(0) == 0
    assert mapping.to_idx(42) == 42

    assert mapping.to_id(0) == 0
    assert mapping.to_id(42) == 42

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
    assert mapping.node_ids == ["a", "c"]
