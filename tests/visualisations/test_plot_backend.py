"""Unit tests for PlotBackend base class."""

from unittest.mock import Mock

import pytest

from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend


def test_plot_backend_initialization() -> None:
    """Test that PlotBackend initializes correctly with plot and show_labels."""
    # Create a mock PathPyPlot instance
    mock_plot = Mock(spec=PathPyPlot)
    mock_plot.data = {"nodes": [], "edges": []}
    mock_plot.config = {"node": {"color": "blue"}, "edge": {"color": "black"}}

    # Initialize PlotBackend
    backend = PlotBackend(plot=mock_plot, show_labels=True)

    # Check that attributes are set correctly
    assert backend.data == mock_plot.data
    assert backend.config == mock_plot.config
    assert backend.show_labels is True


def test_plot_backend_methods_not_implemented() -> None:
    """Test that PlotBackend methods raise NotImplementedError."""
    # Create a mock PathPyPlot instance
    mock_plot = Mock(spec=PathPyPlot)
    mock_plot.data = {"nodes": [], "edges": []}
    mock_plot.config = {"node": {"color": "blue"}, "edge": {"color": "black"}}

    # Initialize PlotBackend
    backend = PlotBackend(plot=mock_plot, show_labels=False)

    # Test that save method raises NotImplementedError
    with pytest.raises(NotImplementedError):
        backend.save("output.png")

    # Test that show method raises NotImplementedError
    with pytest.raises(NotImplementedError):
        backend.show()
