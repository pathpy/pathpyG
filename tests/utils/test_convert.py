import torch
import numpy as np
from torch_geometric import EdgeIndex

from pathpyG.utils import to_numpy


def test_to_numpy():
    tensor = torch.tensor([1, 2, 3])
    assert isinstance(to_numpy(tensor), np.ndarray)
    assert np.array_equal(to_numpy(tensor), np.array([1, 2, 3]))

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert isinstance(to_numpy(edge_index), np.ndarray)
    assert np.array_equal(to_numpy(edge_index), np.array([[0, 1, 1, 2], [1, 0, 2, 1]]))

    edge_index = EdgeIndex(edge_index)
    assert isinstance(to_numpy(edge_index), np.ndarray)
    assert np.array_equal(to_numpy(edge_index), np.array([[0, 1, 1, 2], [1, 0, 2, 1]]))

    index_col, index_row = edge_index
    assert isinstance(to_numpy(index_col), np.ndarray)
    assert np.array_equal(to_numpy(index_col), np.array([0, 1, 1, 2]))
    assert isinstance(to_numpy(index_row), np.ndarray)
    assert np.array_equal(to_numpy(index_row), np.array([1, 0, 2, 1]))
