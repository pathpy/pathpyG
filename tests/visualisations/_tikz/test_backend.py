"""Unit tests for TikZ backend in pathpyG.visualisations."""

import os
import shutil
import tempfile

import pytest
import torch

from pathpyG.core.graph import Graph
from pathpyG.visualisations._tikz.backend import TikzBackend
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TestTikzBackendInitialization:
    """Test TikZ backend initialization and validation."""

    def setup_method(self):
        """Create test graph and plot."""
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        self.plot = NetworkPlot(self.g, layout="spring")

    def test_backend_accepts_network_plot(self):
        """Test that TikZ backend initializes with NetworkPlot."""
        backend = TikzBackend(self.plot, show_labels=True)
        assert backend.data is self.plot.data
        assert backend.config is self.plot.config
        assert backend.show_labels is True
        assert backend._kind == "static"

    def test_backend_with_labels_disabled(self):
        """Test initialization with labels disabled."""
        backend = TikzBackend(self.plot, show_labels=False)
        assert backend.show_labels is False

    def test_backend_rejects_temporal_network_plot(self):
        """Test that TikZ backend rejects TemporalNetworkPlot."""
        from pathpyG.core.temporal_graph import TemporalGraph

        tg = TemporalGraph.from_edge_list(
            [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
        )
        temporal_plot = TemporalNetworkPlot(tg)

        with pytest.raises(ValueError, match="not supported"):
            TikzBackend(temporal_plot, show_labels=True)


class TestTikzBackendTexGeneration:
    """Test TeX and TikZ code generation."""

    def setup_method(self):
        """Create test graph and backend."""
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c")])
        self.plot = NetworkPlot(self.g, layout="spring")
        self.backend = TikzBackend(self.plot, show_labels=True)

    def test_to_tikz_generates_node_commands(self):
        """Test that to_tikz generates Vertex commands for nodes."""
        tikz_code = self.backend.to_tikz()
        assert "\\Vertex[" in tikz_code
        # Should have commands for all three nodes
        assert tikz_code.count("\\Vertex[") == 3

    def test_to_tikz_generates_edge_commands(self):
        """Test that to_tikz generates Edge commands."""
        tikz_code = self.backend.to_tikz()
        assert "\\Edge[" in tikz_code
        assert tikz_code.count("\\Edge[") == 2

    def test_to_tikz_includes_node_labels_when_enabled(self):
        """Test that node labels appear when show_labels=True."""
        backend = TikzBackend(self.plot, show_labels=True)
        tikz_code = backend.to_tikz()
        assert "label=" in tikz_code
        # Node names should be in labels
        assert "a" in tikz_code
        assert "b" in tikz_code
        assert "c" in tikz_code

    def test_to_tikz_excludes_labels_when_disabled(self):
        """Test that labels are excluded when show_labels=False."""
        backend = TikzBackend(self.plot, show_labels=False)
        tikz_code = backend.to_tikz()
        assert "label=" not in tikz_code

    def test_to_tikz_includes_color_styling(self):
        """Test that color information is included."""
        # Node color
        plot = NetworkPlot(self.g, node_color="#ff0000")
        backend = TikzBackend(plot, show_labels=False)
        tikz_code = backend.to_tikz()
        assert "color=" in tikz_code or "RGB" in tikz_code

        # Edge color
        plot = NetworkPlot(self.g, edge_color="#0000ff")
        backend = TikzBackend(plot, show_labels=False)
        tikz_code = backend.to_tikz()
        assert "color=" in tikz_code or "RGB" in tikz_code

    def test_to_tikz_includes_size_information(self):
        """Test that size information is included."""
        tikz_code = self.backend.to_tikz()
        assert "size=" in tikz_code

    def test_to_tikz_includes_opacity(self):
        """Test that opacity information is included."""
        plot = NetworkPlot(self.g, node_opacity=0.03, edge_opacity=0.05)
        backend = TikzBackend(plot, show_labels=False)
        tikz_code = backend.to_tikz()
        assert "opacity=" in tikz_code
        assert "0.03" in tikz_code
        assert "0.05" in tikz_code

    def test_to_tikz_includes_position_coordinates(self):
        """Test that x,y coordinates are included."""
        tikz_code = self.backend.to_tikz()
        assert "x=" in tikz_code
        assert "y=" in tikz_code

    def test_to_tikz_directed_graph_includes_bend(self):
        """Test that directed graphs have bend/Direct options."""
        g_directed = Graph.from_edge_list([("a", "b"), ("b", "c")])
        plot_directed = NetworkPlot(g_directed, layout="spring")
        backend = TikzBackend(plot_directed, show_labels=False)
        tikz_code = backend.to_tikz()
        assert "bend=" in tikz_code or "Direct" in tikz_code

    def test_to_tex_generates_complete_document(self):
        """Test that to_tex generates a complete LaTeX document."""
        tex_code = self.backend.to_tex()
        assert "\\documentclass" in tex_code
        assert "\\begin{document}" in tex_code
        assert "\\end{document}" in tex_code
        assert "\\Vertex[" in tex_code
        assert "\\Edge[" in tex_code

    def test_to_tex_includes_tikz_network_package(self):
        """Test that the template includes tikz-network package."""
        tex_code = self.backend.to_tex()
        assert "tikz" in tex_code.lower()

    def test_to_tex_includes_dimensions(self):
        """Test that document dimensions are included."""
        tex_code = self.backend.to_tex()
        # Should have width/height specifications
        assert "width" in tex_code.lower()
        assert "height" in tex_code.lower()


class TestTikzBackendSaveOperation:
    """Test save functionality for different formats."""

    def setup_method(self):
        """Create test graph, plot, and temporary directory."""
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c")])
        self.plot = NetworkPlot(self.g, layout="spring")
        self.backend = TikzBackend(self.plot, show_labels=True)
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_tex_file(self):
        """Test saving to .tex format."""
        filepath = os.path.join(self.temp_dir, "test.tex")
        self.backend.save(filepath)
        assert os.path.exists(filepath)

        # Verify file content
        with open(filepath, "r") as f:
            content = f.read()
        assert "\\documentclass" in content
        assert "\\Vertex[" in content

    def test_save_unsupported_format_raises(self):
        """Test that unsupported formats raise NotImplementedError."""
        filepath = os.path.join(self.temp_dir, "test.png")
        with pytest.raises(NotImplementedError):
            self.backend.save(filepath)


class TestTikzBackendLatexSymbolReplacement:
    """Test LaTeX symbol replacement in node labels."""

    def setup_method(self):
        """Create backend for testing."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c")])
        plot = NetworkPlot(g, layout="spring")
        self.backend = TikzBackend(plot, show_labels=True)

    def test_arrow_symbol_replacement(self):
        """Test that arrow symbols are replaced with LaTeX commands."""
        self.backend.config["separator"] = "->"
        result = self.backend._replace_with_LaTeX_math_symbol("a->b")
        assert "\\to" in result

    def test_double_arrow_symbol_replacement(self):
        """Test double arrow replacement."""
        self.backend.config["separator"] = "=>"
        result = self.backend._replace_with_LaTeX_math_symbol("a=>b")
        assert "\\Rightarrow" in result

    def test_bidirectional_arrow_replacement(self):
        """Test bidirectional arrow replacement."""
        self.backend.config["separator"] = "<->"
        result = self.backend._replace_with_LaTeX_math_symbol("a<->b")
        assert "\\leftrightarrow" in result

    def test_inequality_symbol_replacement(self):
        """Test inequality symbol replacement."""
        self.backend.config["separator"] = "!="
        result = self.backend._replace_with_LaTeX_math_symbol("a!=b")
        assert "\\neq" in result

    def test_no_replacement_for_regular_labels(self):
        """Test that regular labels are unchanged."""
        result = self.backend._replace_with_LaTeX_math_symbol("node_123")
        assert result == "node_123"


class TestTikzBackendEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_graph(self):
        """Test backend with an empty graph."""
        g = Graph.from_edge_list([])
        plot = NetworkPlot(g, layout=None)
        backend = TikzBackend(plot, show_labels=False)
        tikz_code = backend.to_tikz()
        # Should generate valid TikZ even if empty
        assert isinstance(tikz_code, str)

    def test_higher_order_network_with_separators(self):
        """Test higher-order networks with custom separators."""
        from pathpyG.core.index_map import IndexMap
        from pathpyG.core.multi_order_model import MultiOrderModel
        from pathpyG.core.path_data import PathData

        paths = PathData(IndexMap(["a", "b", "c", "d"]))
        paths.append_walks(
            [["a", "b", "c"], ["b", "c", "d"], ["a", "b", "d"]], weights=[1, 1, 1]
        )
        ho_g = MultiOrderModel.from_path_data(paths, max_order=2).layers[2]

        plot = NetworkPlot(ho_g, layout="spring")
        backend = TikzBackend(plot, show_labels=True)
        tikz_code = backend.to_tikz()

        # Should generate valid TikZ for higher-order nodes
        assert "\\Vertex[" in tikz_code
        assert isinstance(tikz_code, str)
        assert "\\to" in tikz_code
