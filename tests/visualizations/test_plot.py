import torch
import pytest

from types import ModuleType
from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.plot import _get_plot_backend
from pathpyG.visualisations.network_plots import (
    network_plot,
    temporal_plot,
    static_plot,
)
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


# Uses a default pytest fixture: see https://docs.pytest.org/en/6.2.x/tmpdir.html
def test_network_plot_png(tmp_path) -> None:
    """Test to plot a static network as png file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    net.data["edge_size"] = torch.tensor([[3], [4], [5]])
    net.data["node_size"] = torch.tensor([[90], [8], [7]])

    plot = network_plot(net, edge_color="green", layout="fr")
    plot.save(tmp_path / "test.png")
    assert (tmp_path / "test.png").exists()


def test_network_plot_html(tmp_path) -> None:
    """Test to plot a static network as html file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["node_size"] = torch.tensor([[90], [8], [7]])
    plot = network_plot(net)
    plot.save(tmp_path / "test.html")
    assert (tmp_path / "test.html").exists()


def test_plot_function(tmp_path) -> None:
    """Test generic plot function."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    fig = plot(net)
    fig.save(tmp_path / "generic.html")
    assert (tmp_path / "generic.html").exists()


def test_network_plot_tex(tmp_path) -> None:
    """Test to plot a static network as tex file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])

    plot = network_plot(net, layout="fr")
    # PDF probably not supported at github
    # plot.save("test.pdf")
    plot.save(tmp_path / "test.tex")
    assert (tmp_path / "test.tex").exists()


def test_temporal_plot(tmp_path) -> None:
    """Test to plot a temporal network."""
    net = TemporalGraph.from_edge_list(
        [
            ("a", "b", 1),
            ("b", "c", 5),
            ("c", "d", 9),
            ("d", "a", 9),
            ("a", "b", 10),
            ("b", "c", 10),
        ]
    )
    net.data["edge_size"] = torch.tensor([[3], [4], [5], [1], [2], [3]])

    color = {"a": "blue", "b": "red", "c": "green", "d": "yellow"}
    plot = temporal_plot(
        net,
        node_color=color,
        start=3,
        end=25,
        delta=1000,
        layout="fr",
        d3js_local=False,
    )
    plot.save(tmp_path / "temp.html")
    assert (tmp_path / "temp.html").exists()
