import logging

import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._d3js.backend import D3jsBackend
from pathpyG.visualisations._manim.backend import ManimBackend
from pathpyG.visualisations._matplotlib.backend import MatplotlibBackend
from pathpyG.visualisations._tikz.backend import TikzBackend
from pathpyG.visualisations.plot_function import Backends, _get_plot_backend, plot


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
# Runs the test for all different file endings
@pytest.mark.parametrize("file_ending", [".jpg", ".png", ".html", ".tex", ".pdf", ".svg"])
def test_network_plot_save(caplog, tmp_path, file_ending) -> None:
    """Test to plot a static network as a file."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])

    with caplog.at_level(logging.DEBUG):
        plot(net, filename=(tmp_path / ("test" + file_ending)).as_posix())
    assert (tmp_path / ("test" + file_ending)).exists()
    assert "Using backend" in caplog.text


def test_network_plot_save_fails(caplog, tmp_path) -> None:
    """Test to plot a static network as a file with unsupported file type."""
    net = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])

    with caplog.at_level(logging.DEBUG):
        plot(net, filename=(tmp_path / "test.exe").as_posix())
    assert (tmp_path / "test.exe").exists()
    assert "Using default backend" in caplog.text


@pytest.mark.parametrize("file_ending", [".html", ".mp4", ".gif"])
def test_temporal_plot_save(caplog, tmp_path, file_ending) -> None:
    """Test to plot a temporal network."""
    net = TemporalGraph.from_edge_list(
        [
            ("a", "b", 1),
            ("b", "c", 2),
            ("c", "d", 4),
        ]
    )

    with caplog.at_level(logging.DEBUG):
        plot(net, filename=(tmp_path / ("temp" + file_ending)).as_posix())
    assert (tmp_path / ("temp" + file_ending)).exists()
    assert "Using backend" in caplog.text


def test_temporal_plot_save_fails(caplog, tmp_path) -> None:
    """Test to plot a temporal network with unsupported file type."""
    net = TemporalGraph.from_edge_list(
        [
            ("a", "b", 1),
            ("b", "c", 2),
            ("c", "d", 4),
        ]
    )

    with caplog.at_level(logging.DEBUG):
        plot(net, filename=(tmp_path / "temp.exe").as_posix())
    assert (tmp_path / "temp.exe").exists()
    assert "Using default backend" in caplog.text


def test_backend_enum() -> None:
    """Test the Backends enum."""
    assert Backends.matplotlib == "matplotlib"
    assert Backends.tikz == "tikz"
    assert Backends.d3js == "d3js"
    assert Backends.manim == "manim"
    assert Backends.is_backend("matplotlib")
    assert not Backends.is_backend("not_a_backend")
