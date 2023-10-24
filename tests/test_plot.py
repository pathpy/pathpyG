# from __future__ import annotations

from pathpyG.visualisations.plot import PathPyPlot

# import torch


def test_PathPyPlot() -> None:
    """Test PathPyPlot class."""
    plot = PathPyPlot()

    assert isinstance(plot.data, dict)
    assert isinstance(plot.config, dict)
