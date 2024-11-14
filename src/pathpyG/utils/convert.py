"""
Utility functions for converting between different data types.
"""

import torch
import numpy as np
from torch_geometric import EdgeIndex

# Ensure backward compatibility with torch_geometric==2.5
try:
    from torch_geometric import Index
except ImportError:

    class Index:
        def __init__(self) -> None:
            raise NotImplementedError("torch_geometric.Index is not available. Please upgrade to torch_geometric>=2.6.")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor or tensor subclasses like `torch_geometric.Edge_Index` to numpy.

    Args:
        tensor: Tensor or tensor subclass.

    Returns:
        Numpy array.
    """
    if isinstance(tensor, (EdgeIndex, Index)):
        return tensor.as_tensor().numpy()
    return tensor.numpy()
