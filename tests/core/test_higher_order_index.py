# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest
from typing import List, Union, Tuple

from torch import IntTensor

from pathpyG.core.HigherOrderIndexMap import HigherOrderIndexMap


def test_constructor():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    fo_node_ids = ["A", "B", "C"]
    map = HigherOrderIndexMap(ho_nodes, fo_node_ids)
    assert map.has_ids == True
    assert map.fo_node_ids == ["A", "B", "C"]
    assert map.idx_to_id == {0: ("A", "B"), 1: ("A", "C"), 2: ("B", "C")}
    assert map.id_to_idx == {("A", "B"): 0, ("A", "C"): 1, ("B", "C"): 2}


def test_constructor_no_ids():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    map = HigherOrderIndexMap(ho_nodes)
    assert map.has_ids == False
    assert map.fo_node_ids == []
    assert map.idx_to_id == {}
    assert map.id_to_idx == {}


def test_to_id():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    fo_node_ids = ["A", "B", "C"]
    map = HigherOrderIndexMap(ho_nodes, fo_node_ids)
    assert map.to_id(0) == ("A", "B")
    assert map.to_id(1) == ("A", "C")
    assert map.to_id(2) == ("B", "C")


def test_to_id_no_ids():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    map = HigherOrderIndexMap(ho_nodes)
    assert map.to_id(1) == 1


def test_to_idx():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    fo_node_ids = ["A", "B", "C"]
    map = HigherOrderIndexMap(ho_nodes, fo_node_ids)
    assert map.to_idx(("A", "B")) == 0
    assert map.to_idx(("A", "C")) == 1
    assert map.to_idx(("B", "C")) == 2


def test_to_idx_no_ids():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    map = HigherOrderIndexMap(ho_nodes)
    assert map.to_idx(1) == 1


def test_str():
    ho_nodes = IntTensor([[0, 1], [0, 2], [1, 2]])
    fo_node_ids = ["A", "B", "C"]
    map = HigherOrderIndexMap(ho_nodes, fo_node_ids)
    assert str(map) == "('A', 'B') -> 0\n('A', 'C') -> 1\n('B', 'C') -> 2\n"
