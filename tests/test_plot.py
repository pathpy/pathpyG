import torch
import pytest

from types import ModuleType
from pathpyG.core.Graph import Graph
from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.plot import _get_plot_backend
from pathpyG.visualisations.network_plots import network_plot


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

    # load matplotlib backend
    plt = _get_plot_backend(backend="matplotlib")
    assert isinstance(plt, ModuleType)

    # test .png file
    png = _get_plot_backend(filename="test.png")
    assert isinstance(png, ModuleType)

    assert png == plt


def test_network_plot() -> None:
    """Test to plot a static network."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    net.data["edge_size"] = torch.tensor([[3], [4], [5]])

    net.data["node_size"] = torch.tensor([[9], [8], [7]])

    plot = network_plot(net)
    plot.save("test.png")
