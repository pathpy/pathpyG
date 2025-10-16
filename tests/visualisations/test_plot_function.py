import torch
import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._matplotlib.backend import MatplotlibBackend
from pathpyG.visualisations._tikz.backend import TikzBackend
from pathpyG.visualisations._manim.backend import ManimBackend
from pathpyG.visualisations._d3js.backend import D3jsBackend
from pathpyG.visualisations.plot_function import _get_plot_backend, plot


def test_get_plot_backend() -> None:
    """Test to get a valid plot backend."""

    # backend which does not exist
    with pytest.raises(ImportError):
        _get_plot_backend(default="does not exist", backend=None, filename=None)

    # load matplotlib backend
    plt = _get_plot_backend(backend="matplotlib", default=None, filename=None)
    assert plt == MatplotlibBackend

    # test .png file
    png = _get_plot_backend(filename="test.png", default=None, backend=None)
    assert png == MatplotlibBackend

    # load d3js backend
    d3js = _get_plot_backend(backend="d3js", default=None, filename=None)
    assert d3js == D3jsBackend

    # test .html file
    html = _get_plot_backend(filename="test.html", default=None, backend=None)
    assert html == D3jsBackend

    # load tikz backend
    tikz = _get_plot_backend(backend="tikz", default=None, filename=None)
    assert tikz == TikzBackend

    # test .tex file
    tex = _get_plot_backend(filename="test.tex", default=None, backend=None)
    assert tex == TikzBackend

    # test .pdf file
    pdf = _get_plot_backend(filename="test.pdf", default=None, backend=None)
    assert pdf == TikzBackend

    # test .svg file
    svg = _get_plot_backend(filename="test.svg", default=None, backend=None)
    assert svg == TikzBackend

    # load manim backend
    manim = _get_plot_backend(backend="manim", default=None, filename=None)
    assert manim == ManimBackend

    # test .mp4 file
    mp4 = _get_plot_backend(filename="test.mp4", default=None, backend=None)
    assert mp4 == ManimBackend

    # test .gif file
    gif = _get_plot_backend(filename="test.gif", default=None, backend=None)
    assert gif == ManimBackend


# Uses a default pytest fixture: see https://docs.pytest.org/en/6.2.x/tmpdir.html
def test_network_plot_png(tmp_path) -> None:
    """Test to plot a static network as png file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["edge_weight"] = torch.tensor([[1], [1], [2]])
    net.data["edge_size"] = torch.tensor([[3], [4], [5]])
    net.data["node_size"] = torch.tensor([[90], [8], [7]])

    out = plot(net, edge_color="green", layout="fr")
    out.save(tmp_path / "test.png")
    assert (tmp_path / "test.png").exists()


def test_network_plot_html(tmp_path) -> None:
    """Test to plot a static network as html file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
    net.data["node_size"] = torch.tensor([[90], [8], [7]])
    out = plot(net)
    out.save(tmp_path / "test.html")
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

    out = plot(net, layout="fr")
    out.save(tmp_path / "test.tex")
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
    out = plot(
        net,
        node_color=color,
        delta=1000,
        layout="fr",
        d3js_local=False,
    )
    out.save(tmp_path / "temp.html")
    assert (tmp_path / "temp.html").exists()
