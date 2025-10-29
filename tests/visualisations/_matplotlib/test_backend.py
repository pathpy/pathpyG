"""Unit tests for Matplotlib backend in pathpyG.visualisations."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._matplotlib.backend import MatplotlibBackend
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot
from pathpyG.visualisations.unfolded_network_plot import TimeUnfoldedNetworkPlot


def test_supports_unfolded_network_plot():
    """Test that Matplotlib backend supports TimeUnfoldedNetworkPlot."""
    tg = TemporalGraph.from_edge_list(
        [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
    )
    unfolded_plot = TimeUnfoldedNetworkPlot(tg)

    backend = MatplotlibBackend(unfolded_plot, show_labels=True)
    assert backend.data is unfolded_plot.data
    assert backend.config is unfolded_plot.config
    assert backend.show_labels is True
    assert backend._kind == "unfolded"


class TestMatplotlibBackendInitialization:
    """Test MatplotlibBackend initialization and configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple graph
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        self.g = Graph.from_edge_list(edges)
        self.plot = NetworkPlot(self.g)

    def test_backend_initialization_with_network_plot(self):
        """Test that MatplotlibBackend initializes with NetworkPlot."""
        backend = MatplotlibBackend(self.plot, show_labels=True)
        assert backend is not None
        assert backend._kind == "static"
        assert backend.show_labels is True

    def test_backend_initialization_with_temporal_plot_raises(self):
        """Test that MatplotlibBackend raises error with temporal plot."""
        tedges = [("a", "b", 1), ("b", "c", 2)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg)
        
        with pytest.raises(ValueError, match="not supported"):
            MatplotlibBackend(temp_plot, show_labels=False)

    def test_backend_inherits_plot_data_and_config(self):
        """Test that backend has access to plot data and config."""
        backend = MatplotlibBackend(self.plot, show_labels=True)
        
        assert hasattr(backend, "data")
        assert hasattr(backend, "config")
        assert isinstance(backend.data, dict)
        assert isinstance(backend.config, dict)
        assert "nodes" in backend.data
        assert "edges" in backend.data

    def test_backend_initialization_with_directed_graph(self):
        """Test backend initialization with directed graph."""
        # Default is directed
        backend = MatplotlibBackend(self.plot, show_labels=False)
        assert backend.config["directed"] is True

    def test_backend_initialization_with_undirected_graph(self):
        """Test backend initialization with undirected graph."""
        g_undirected = self.g.to_undirected()
        plot_undirected = NetworkPlot(g_undirected)
        backend = MatplotlibBackend(plot_undirected, show_labels=False)
        assert backend.config["directed"] is False


class TestMatplotlibBackendFigureCreation:
    """Test MatplotlibBackend figure and axes creation."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        self.g = Graph.from_edge_list(edges)
        self.plot = NetworkPlot(self.g, layout="spring")

    def test_to_fig_returns_figure_and_axes(self):
        """Test that to_fig() returns matplotlib Figure and Axes."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        fig, ax = backend.to_fig()
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_to_fig_turns_off_axis(self):
        """Test that axis frame is turned off."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        fig, ax = backend.to_fig()
        
        # Axis should be off (frame invisible)
        assert not ax.axison

    def test_to_fig_adds_collections(self):
        """Test that figure contains collections for nodes and edges."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        fig, ax = backend.to_fig()
        
        # Should have collections (edges and nodes)
        collections = ax.collections
        assert len(collections) > 0

    def test_to_fig_with_labels_adds_annotations(self):
        """Test that labels are added when show_labels=True."""
        backend = MatplotlibBackend(self.plot, show_labels=True)
        fig, ax = backend.to_fig()
        
        # Should have text annotations for node labels
        texts = ax.texts
        assert len(texts) > 0
        # Should have one label per node
        assert len(texts) == len(self.plot.data["nodes"])

    def test_to_fig_without_labels_has_no_annotations(self):
        """Test that no labels are added when show_labels=False."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        fig, ax = backend.to_fig()
        
        # Should have no text annotations
        texts = ax.texts
        assert len(texts) == 0


class TestMatplotlibBackendEdgeRendering:
    """Test edge rendering for undirected and directed graphs."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)

    def test_undirected_edges_use_line_collection(self):
        """Test that undirected graphs use LineCollection."""
        g_undirected = self.g.to_undirected()
        plot = NetworkPlot(g_undirected, layout="spring")
        backend = MatplotlibBackend(plot, show_labels=False)
        
        fig, ax = backend.to_fig()
        
        # Should contain LineCollection for undirected edges
        from matplotlib.collections import LineCollection
        line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) > 0

    def test_directed_edges_use_path_collection(self):
        """Test that directed graphs use PathCollection."""
        plot = NetworkPlot(self.g, layout="spring")
        backend = MatplotlibBackend(plot, show_labels=False)
        
        fig, ax = backend.to_fig()
        
        # Should contain PathCollection for directed edges
        from matplotlib.collections import PathCollection
        path_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(path_collections) > 0

    def test_directed_edges_have_arrowheads(self):
        """Test that directed edges have arrowhead collections."""
        plot = NetworkPlot(self.g, layout="spring")
        backend = MatplotlibBackend(plot, show_labels=False)
        
        fig, ax = backend.to_fig()
        
        # Directed graphs should have multiple PathCollections (edges + arrowheads)
        from matplotlib.collections import PathCollection
        path_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
        # Should have exactly 2 (edge paths and arrowheads)
        assert len(path_collections) == 2


class TestMatplotlibBackendBezierCurves:
    """Test Bezier curve generation for directed edges."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.plot = NetworkPlot(self.g, layout="spring")
        self.backend = MatplotlibBackend(self.plot, show_labels=False)

    def test_get_bezier_curve_returns_vertices_and_codes(self):
        """Test that get_bezier_curve returns proper format."""
        source_coords = np.array([[0, 0], [1, 1]], dtype=float)
        target_coords = np.array([[1, 0], [2, 1]], dtype=float)
        source_size = np.array([[0.1], [0.1]])
        target_size = np.array([[0.1], [0.1]])
        
        vertices, codes = self.backend.get_bezier_curve(
            source_coords, target_coords, source_size, target_size, head_length=0.02
        )
        
        assert len(vertices) == 3
        assert len(codes) == 3

    def test_get_bezier_curve_handles_zero_distance(self):
        """Test that Bezier curve handles zero-length edges and fall back to straight lines."""
        source_coords = np.array([[0, 0], [0, 1]], dtype=float)
        target_coords = np.array([[0, 0], [0, 0.99]], dtype=float)  # Same point
        source_size = np.array([[0.05], [0.05]])
        target_size = np.array([[0.05], [0.05]])
        
        # Should not crash
        vertices, codes = self.backend.get_bezier_curve(
            source_coords, target_coords, source_size, target_size, head_length=0.02
        )
        
        assert len(vertices) == 2
        assert len(codes) == 2


class TestMatplotlibBackendArrowheads:
    """Test arrowhead generation for directed edges."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.plot = NetworkPlot(self.g, layout="spring")
        self.backend = MatplotlibBackend(self.plot, show_labels=False)

    def test_get_arrowhead_returns_vertices_and_codes(self):
        """Test that get_arrowhead returns proper format."""
        # Create simple Bezier curve vertices
        P0 = np.array([[0, 0], [0, 0]], dtype=float)
        P1 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        P2 = np.array([[1, 0], [1, 0]], dtype=float)
        vertices = [P0, P1, P2]
        
        arrow_vertices, arrow_codes = self.backend.get_arrowhead(vertices, head_length=0.02)
        
        assert len(arrow_vertices) == 4
        assert len(arrow_codes) == 4

    def test_get_arrowhead_scales_with_edge_size(self):
        """Test that arrowhead size scales with edge width."""
        P0 = np.array([[0, 0], [0, 0]])
        P1 = np.array([[0.5, 0.5], [0.5, 0.5]])
        P2 = np.array([[1, 0], [1, 0]])
        vertices = [P0, P1, P2]
        
        # Set different edge sizes
        self.backend.data["edges"]["size"] = np.array([1.0, 5.0])
        
        arrow_vertices, arrow_codes = self.backend.get_arrowhead(vertices, head_length=0.02)
        
        # First arrowhead vertices
        arrow1_vertices = [v[0] for v in arrow_vertices]
        arrow2_vertices = [v[1] for v in arrow_vertices]
        
        # Arrowheads should have different sizes
        # Calculate width of each arrowhead
        width1 = np.linalg.norm(arrow1_vertices[0] - arrow1_vertices[2])
        width2 = np.linalg.norm(arrow2_vertices[0] - arrow2_vertices[2])
        
        # Second arrowhead should be larger
        assert width2 > width1


class TestMatplotlibBackendFileOperations:
    """Test file saving and display operations."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.plot = NetworkPlot(self.g, layout="spring")

    def test_save_creates_file(self):
        """Test that save() creates a file."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_plot.png"
            backend.save(str(output_file))
            
            # File should exist
            assert output_file.exists()
            # File should have content
            assert output_file.stat().st_size > 0

    def test_save_supports_multiple_formats(self):
        """Test that save() works with different file formats."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        
        formats = ["png", "jpg"]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fmt in formats:
                output_file = Path(tmp_dir) / f"test_plot.{fmt}"
                backend.save(str(output_file))
                
                # File should exist
                assert output_file.exists()
                assert output_file.stat().st_size > 0

    @patch("matplotlib.pyplot.show")
    def test_show_calls_plt_show(self, mock_show):
        """Test that show() calls matplotlib's show function."""
        backend = MatplotlibBackend(self.plot, show_labels=False)
        backend.show()
        
        # plt.show() should have been called
        mock_show.assert_called_once()
