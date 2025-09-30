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


def to_numpy(input_iterable: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """
    Convert an iterable (including a tensor or tensor subclasses like `torch_geometric.Edge_Index`) to numpy.

    Args:
        input_iterable: Tensor, tensor subclass, numpy array or list.

    Returns:
        Numpy array.
    """
    if isinstance(input_iterable, (EdgeIndex, Index)):
        return input_iterable.as_tensor().numpy()
    elif isinstance(input_iterable, torch.Tensor):
        return input_iterable.numpy()
    elif isinstance(input_iterable, (list, tuple)):
        return np.array(input_iterable)
    elif isinstance(input_iterable, np.ndarray):
        return input_iterable
