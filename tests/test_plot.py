import pytest

# import torch
from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.plot import _get_plot_backend


def test_PathPyPlot() -> None:
    """Test PathPyPlot class."""
    plot = PathPyPlot()

    assert isinstance(plot.data, dict)
    assert isinstance(plot.config, dict)


def test_get_plot_backend() -> None:
    """Test to get a valid plot backend."""

    # backend which does not exist
    with pytest.raises(ImportError):
        _get_plot_backend(default="does not exist")
