# pylint: disable=missing-function-docstring,missing-module-docstring
# pylint: disable=invalid-name

from __future__ import annotations

import pytest
import scipy.sparse as s
import torch
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.testing import get_random_edge_index

from pathpyG import Graph, IndexMap


@pytest.mark.gpu
def test_init():
    assert False