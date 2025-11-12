"""Utility functions for converting between different data types."""

import numpy as np
import torch
from torch_geometric import EdgeIndex

# Ensure backward compatibility with torch_geometric==2.5
try:
    from torch_geometric import Index
except ImportError:

    class Index:  # type: ignore[no-redef]
        """Placeholder for torch_geometric.Index when not available."""
        def __init__(self) -> None:  # noqa: D107
            raise NotImplementedError("torch_geometric.Index is not available. Please upgrade to torch_geometric>=2.6.")


def to_numpy(input_iterable: torch.Tensor | np.ndarray | list | tuple) -> np.ndarray:
    """Convert an iterable (including a tensor or tensor subclasses like [`torch_geometric.EdgeIndex`][torch_geometric.EdgeIndex]) to numpy.

    Args:
        input_iterable: [Tensor][torch.Tensor], tensor subclass, [numpy array][numpy.ndarray] or list.

    Returns:
        Numpy array.
    """
    if isinstance(input_iterable, (EdgeIndex, Index)):
        return input_iterable.as_tensor().cpu().numpy()
    elif isinstance(input_iterable, torch.Tensor):
        return input_iterable.cpu().numpy()
    elif isinstance(input_iterable, (list, tuple)):
        return np.array(input_iterable)
    elif isinstance(input_iterable, np.ndarray):
        return input_iterable
