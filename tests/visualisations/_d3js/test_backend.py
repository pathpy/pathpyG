"""Unit tests for D3.js backend in pathpyG.visualisations."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._d3js.backend import D3jsBackend
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TestD3jsBackendInitialization:
    """Test D3jsBackend initialization and configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple static network
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")

        # Create temporal network
        tedges = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
        self.tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(self.tg)

    def test_backend_initialization_with_static_plot(self):
        """Test that D3jsBackend initializes with NetworkPlot."""
        backend = D3jsBackend(self.static_plot, show_labels=True)

        assert backend is not None
        assert backend._kind == "static"
        assert backend.show_labels is True

    def test_backend_initialization_with_temporal_plot(self):
        """Test that D3jsBackend initializes with TemporalNetworkPlot."""
        backend = D3jsBackend(self.temp_plot, show_labels=False)

        assert backend is not None
        assert backend._kind == "temporal"
        assert backend.show_labels is False

    def test_backend_initialization_with_unsupported_plot_raises(self):
        """Test that unsupported plot types raise ValueError."""
        # Create a custom unsupported plot type
        from pathpyG.visualisations.pathpy_plot import PathPyPlot

        unsupported_plot = PathPyPlot()

        with pytest.raises(ValueError, match="not supported"):
            D3jsBackend(unsupported_plot, show_labels=False)

    def test_backend_inherits_plot_data_and_config(self):
        """Test that backend has access to plot data and config."""
        backend = D3jsBackend(self.static_plot, show_labels=True)

        assert hasattr(backend, "data")
        assert hasattr(backend, "config")
        assert isinstance(backend.data, dict)
        assert isinstance(backend.config, dict)

    def test_backend_stores_show_labels(self):
        """Test that backend correctly stores show_labels parameter."""
        backend_with_labels = D3jsBackend(self.static_plot, show_labels=True)
        backend_without_labels = D3jsBackend(self.static_plot, show_labels=False)

        assert backend_with_labels.show_labels is True
        assert backend_without_labels.show_labels is False


class TestD3jsBackendDataPreparation:
    """Test data preparation methods for D3.js format."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_prepare_data_structure(self):
        """Test that _prepare_data returns correct structure."""
        data_dict = self.backend._prepare_data()

        assert isinstance(data_dict, dict)
        assert "nodes" in data_dict
        assert "edges" in data_dict
        assert isinstance(data_dict["nodes"], list)
        assert isinstance(data_dict["edges"], list)

    def test_prepare_data_node_structure(self):
        """Test that nodes have correct structure."""
        data_dict = self.backend._prepare_data()
        nodes = data_dict["nodes"]

        # Should have 3 nodes
        assert len(nodes) == 3

        # Each node should have uid and position
        for node in nodes:
            assert "uid" in node
            assert "xpos" in node  # x renamed to xpos
            assert "ypos" in node  # y renamed to ypos

    def test_prepare_data_edge_structure(self):
        """Test that edges have correct structure."""
        data_dict = self.backend._prepare_data()
        edges = data_dict["edges"]

        # Should have edges
        assert len(edges) > 0

        # Each edge should have uid, source, target
        for edge in edges:
            assert "uid" in edge
            assert "source" in edge
            assert "target" in edge

    def test_prepare_data_preserves_attributes(self):
        """Test that node and edge attributes are preserved."""
        data_dict = self.backend._prepare_data()
        nodes = data_dict["nodes"]
        edges = data_dict["edges"]

        # Nodes should have color, size, opacity
        for node in nodes:
            assert "color" in node
            assert "size" in node
            assert "opacity" in node

        # Edges should have color, size, opacity
        for edge in edges:
            assert "color" in edge
            assert "size" in edge
            assert "opacity" in edge


class TestD3jsBackendConfigPreparation:
    """Test configuration preparation for D3.js."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_prepare_config_structure(self):
        """Test that _prepare_config returns correct structure."""
        config_dict = self.backend._prepare_config()

        assert isinstance(config_dict, dict)
        assert "node" in config_dict
        assert "edge" in config_dict
        assert "width" in config_dict
        assert "height" in config_dict
        assert "show_labels" in config_dict

    def test_prepare_config_converts_colors_to_hex(self):
        """Test that colors are converted to hex format."""
        config_dict = self.backend._prepare_config()

        # Node and edge colors should be hex strings
        node_color = config_dict["node"]["color"]
        edge_color = config_dict["edge"]["color"]

        assert isinstance(node_color, str)
        assert node_color.startswith("#")
        assert isinstance(edge_color, str)
        assert edge_color.startswith("#")

    def test_prepare_config_converts_dimensions_to_pixels(self):
        """Test that width and height are converted to numeric pixels."""
        config_dict = self.backend._prepare_config()

        # Width and height should be numeric
        assert isinstance(config_dict["width"], (int, float))
        assert isinstance(config_dict["height"], (int, float))
        assert config_dict["width"] > 0
        assert config_dict["height"] > 0


class TestD3jsBackendJSONSerialization:
    """Test JSON serialization methods."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_to_json_returns_tuple(self):
        """Test that to_json returns tuple of two strings."""
        result = self.backend.to_json()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_to_json_produces_valid_json(self):
        """Test that to_json produces valid JSON strings."""
        data_json, config_json = self.backend.to_json()

        # Should be parseable as JSON
        data = json.loads(data_json)
        config = json.loads(config_json)

        assert isinstance(data, dict)
        assert isinstance(config, dict)

    def test_to_json_data_structure(self):
        """Test that JSON data has correct structure."""
        data_json, _ = self.backend.to_json()
        data = json.loads(data_json)

        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_to_json_config_structure(self):
        """Test that JSON config has correct structure."""
        _, config_json = self.backend.to_json()
        config = json.loads(config_json)

        assert "node" in config
        assert "edge" in config
        assert "show_labels" in config


class TestD3jsBackendTemplateSystem:
    """Test template loading and assembly."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_get_template_returns_string(self):
        """Test that get_template returns JavaScript code."""
        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src/pathpyG/visualisations/_d3js/templates",
        )

        if os.path.exists(template_dir):
            js_template = self.backend.get_template(template_dir)
            assert isinstance(js_template, str)
            assert len(js_template) > 0


class TestD3jsBackendHTMLGeneration:
    """Test HTML generation and assembly."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_to_html_returns_string(self):
        """Test that to_html returns HTML string."""
        html = self.backend.to_html()

        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_html_contains_essential_elements(self):
        """Test that HTML contains essential elements."""
        html = self.backend.to_html()

        # Should contain CSS
        assert "<style>" in html
        assert "</style>" in html

        # Should contain div container
        assert "<div" in html
        assert "</div>" in html

        # Should contain script tags
        assert "<script" in html
        assert "</script>" in html

    def test_to_html_includes_d3js_library(self):
        """Test that HTML includes D3.js library reference."""
        html = self.backend.to_html()

        # Should reference D3.js (either local or CDN)
        assert "d3" in html.lower()

    def test_to_html_includes_data_and_config(self):
        """Test that HTML includes embedded data and config."""
        html = self.backend.to_html()

        # Should contain data and config declarations
        assert "const data" in html
        assert "const config" in html

    def test_to_html_has_unique_dom_id(self):
        """Test that generated HTML has unique DOM ID."""
        html1 = self.backend.to_html()
        html2 = self.backend.to_html()

        # Extract div IDs (they should be different)
        # Multiple calls should generate different IDs
        assert 'id = "x' in html1
        assert 'id = "x' in html2
        assert html1 != html2


class TestD3jsBackendFileOperations:
    """Test file save and display operations."""

    def setup_method(self):
        """Set up test fixtures."""
        edges = [("a", "b"), ("b", "c")]
        self.g = Graph.from_edge_list(edges)
        self.static_plot = NetworkPlot(self.g, layout="spring")
        self.backend = D3jsBackend(self.static_plot, show_labels=True)

    def test_save_creates_html_file(self):
        """Test that save creates HTML file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_output.html"

            self.backend.save(str(output_file))

            # Verify file was created
            assert output_file.exists()

            # Verify file contains HTML
            content = output_file.read_text()
            assert len(content) > 0
            assert "<script" in content

    @patch("pathpyG.visualisations._d3js.backend.config")
    @patch("pathpyG.visualisations._d3js.backend.webbrowser")
    def test_show_in_browser_opens_file(self, mock_browser, mock_config):
        """Test that show opens browser in non-interactive mode."""
        mock_config.__getitem__.return_value = {"interactive": False}

        self.backend.show()

        # Verify browser was opened
        mock_browser.open.assert_called_once()
