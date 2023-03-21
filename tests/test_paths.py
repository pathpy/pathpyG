#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

from pathpyG.Paths import Paths


def test_constructor():
    """
    This defines the expected usage, which can then be used in various test cases.
    """
    p = Paths()
    assert p.get_edge_count() == 0


def test_add_edge(unit_test_mocks: None):
    """
    This is a simple test, which can use a mock to override online functionality.
    unit_test_mocks: Fixture located in conftest.py, implictly imported via pytest.
    """
    p = Paths()
    p.add_edge(0, 0 , 1)
    assert p.get_edge_count() == 1


def test_add_walk(unit_test_mocks: None):
    """
    This is a simple test, which can use a mock to override online functionality.
    unit_test_mocks: Fixture located in conftest.py, implictly imported via pytest.
    """
    p = Paths()
    p.add_walk(0, [0,1,2])
    assert p.get_edge_count() == 2