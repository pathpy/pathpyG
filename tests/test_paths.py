from __future__ import annotations

import torch
from pathpyG.core.PathData import GlobalPathStorage

import pytest

@pytest.fixture
def test_path_constructor():
    p = GlobalPathStorage()
    assert p.num_paths == 0

@pytest.fixture
def test_path_add_walk(unit_test_mocks: None):
    """
    This is a simple test, which can use a mock to override online functionality.
    unit_test_mocks: Fixture located in conftest.py, implictly imported via pytest.
    """
    paths = GlobalPathStorage()
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[0,2],[2,3]])) # A -> C -> D
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E
    paths.add_walk(torch.tensor([[1,2],[2,4]])) # B -> C -> E

    assert paths.num_paths == 4
    assert paths.num_nodes == 5
    assert paths.num_edges == 8
