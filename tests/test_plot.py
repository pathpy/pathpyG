import torch
import pytest

from types import ModuleType
from pathpyG.core.Graph import Graph
from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.plot import _get_plot_backend
from pathpyG.visualisations.network_plots import network_plot
from pathpyG.visualisations import plot


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

    # load d3js backend
    d3js = _get_plot_backend(backend="d3js")
    assert isinstance(d3js, ModuleType)

    # test .html file
    html = _get_plot_backend(filename="test.html")
    assert isinstance(html, ModuleType)

    assert d3js == html

    # load tikz backend
    tikz = _get_plot_backend(backend="tikz")
    assert isinstance(tikz, ModuleType)

    # test .tex file
    tex = _get_plot_backend(filename="test.tex")
    assert isinstance(tex, ModuleType)

    assert tikz == tex


def test_network_plot_png() -> None:
    """Test to plot a static network as png file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    net.data["edge_size"] = torch.tensor([[3], [4], [5]])
    net.data["node_size"] = torch.tensor([[90], [8], [7]])

    plot = network_plot(net, edge_color="green", layout="fr")
    plot.save("test.png")


def test_network_plot_html() -> None:
    """Test to plot a static network as html file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])

    plot = network_plot(net)
    plot.save("test.html")


def test_plot_function() -> None:
    """Test generic plot function."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    fig = plot(net)
    fig.save("generic.html")


def test_network_plot_tex() -> None:
    """Test to plot a static network as tex file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])

    plot = network_plot(net, layout="fr")
    plot.save("test.pdf")
    plot.save("test.tex")
