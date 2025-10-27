"""Unit tests for TemporalNetworkPlot class in pathpyG.visualisations."""

import pandas as pd
import pytest
import torch

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TestTemporalNetworkPlot:
    """Test TemporalNetworkPlot initialization and basic functionality."""

    def setup_method(self):
        """Create a simple temporal graph for testing."""
        # Create temporal graph with edges at different times
        self.tg = TemporalGraph.from_edge_list(
            [("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "b", 3)]
        )

    def test_initialization(self):
        """Test that TemporalNetworkPlot initializes correctly."""
        plot = TemporalNetworkPlot(self.tg)
        assert plot.network is self.tg
        assert plot._kind == "temporal"
        assert isinstance(plot.data, dict)

    def test_node_data_structure(self):
        """Test that node data has correct temporal structure."""
        plot = TemporalNetworkPlot(self.tg)
        nodes = plot.data["nodes"]
        assert isinstance(nodes, pd.DataFrame)
        # Should have MultiIndex with uid
        assert "uid" in nodes.index.names
        # Should have temporal columns
        assert "start" in nodes.columns
        assert "end" in nodes.columns

    def test_edge_data_structure(self):
        """Test that edge data has correct temporal structure."""
        plot = TemporalNetworkPlot(self.tg)
        edges = plot.data["edges"]
        assert isinstance(edges, pd.DataFrame)
        # Should have MultiIndex with source and target
        assert "source" in edges.index.names
        assert "target" in edges.index.names
        # Should have temporal columns
        assert "start" in edges.columns
        assert "end" in edges.columns

    def test_temporal_config(self):
        """Test that temporal-specific config is set correctly."""
        plot = TemporalNetworkPlot(self.tg)
        assert plot.config["directed"] is True
        assert plot.config["curved"] is False

    def test_node_lifetime_tracking(self):
        """Test that node start and end times are computed correctly."""
        plot = TemporalNetworkPlot(self.tg)
        nodes = plot.data["nodes"]
        # All nodes should have valid start/end times
        assert (nodes["start"] >= 0).all()
        assert (nodes["end"] > nodes["start"]).all()

    def test_edge_lifetime_tracking(self):
        """Test that edge start and end times are computed correctly."""
        plot = TemporalNetworkPlot(self.tg)
        edges = plot.data["edges"]
        # Edges should have valid start/end times
        assert (edges["start"] >= 0).all()
        assert (edges["end"] > edges["start"]).all()
        # By default, edges last for one time step
        assert (edges["end"] - edges["start"] == 1).all()


class TestTemporalNetworkPlotAttributes:
    """Test temporal node and edge attribute assignment."""

    def setup_method(self):
        """Create a temporal graph for attribute testing."""
        self.tg = TemporalGraph.from_edge_list(
            [("a", "b", 0), ("b", "c", 1), ("c", "a", 2)]
        )

    def test_node_constant_attributes(self):
        """Test assigning constant attributes to all nodes."""
        plot = TemporalNetworkPlot(self.tg, node_color="#ff0000", node_size=10)
        nodes = plot.data["nodes"]
        assert (nodes["color"] == "#ff0000").all()
        assert (nodes["size"] == 10).all()

    def test_node_temporal_dict_attributes(self):
        """Test assigning node attributes by (node, time) tuples."""
        node_colors = {
            ("a", 0): "#ff0000",
            ("b", 1): "#00ff00",
            ("c", 2): "#0000ff",
        }
        plot = TemporalNetworkPlot(self.tg, node_color=node_colors)
        nodes = plot.data["nodes"]
        # Should have assigned colors at specific times
        assert "#ff0000" in nodes["color"].values or "#00ff00" in nodes["color"].values

    def test_node_static_dict_attributes(self):
        """Test assigning node attributes by node ID (applies to all times)."""
        node_sizes = {"a": 10, "b": 20, "c": 30}
        plot = TemporalNetworkPlot(self.tg, node_size=node_sizes)
        nodes = plot.data["nodes"]
        # Check that at least some nodes have the assigned sizes
        assert 10 in nodes["size"].values or 20 in nodes["size"].values

    def test_edge_constant_attributes(self):
        """Test assigning constant attributes to all edges."""
        plot = TemporalNetworkPlot(self.tg, edge_color="#0000ff", edge_size=5)
        edges = plot.data["edges"]
        assert (edges["color"] == "#0000ff").all()
        assert (edges["size"] == 5).all()

    def test_node_attribute_forward_fill(self):
        """Test that node attributes are forward-filled over time."""
        # Assign color only at time 0
        node_colors = {("a", 0): "#ff0000"}
        plot = TemporalNetworkPlot(self.tg, node_color=node_colors)
        nodes = plot.data["nodes"]
        # Node 'a' should have the color forward-filled
        a_nodes = nodes[nodes.index.get_level_values("uid") == "a"]
        if len(a_nodes) > 1:
            # Color should be consistent across time steps
            assert len(a_nodes["color"].unique()) == 1


class TestTemporalNetworkPlotLayout:
    """Test temporal layout computation with windowing."""

    def setup_method(self):
        """Create a temporal graph for layout testing."""
        self.tg = TemporalGraph.from_edge_list(
            [("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "d", 3)]
        )

    def test_layout_with_default_window(self):
        """Test layout computation with default window size."""
        plot = TemporalNetworkPlot(self.tg, layout="spring")
        nodes = plot.data["nodes"]
        # Should have x, y coordinates
        assert "x" in nodes.columns
        assert "y" in nodes.columns
        # Coordinates should be normalized to [0, 1]
        assert nodes["x"].notna().any()
        assert nodes["y"].notna().any()

    def test_layout_with_symmetric_window(self):
        """Test layout with symmetric window size."""
        plot = TemporalNetworkPlot(self.tg, layout="spring", layout_window_size=2)
        nodes = plot.data["nodes"]
        assert "x" in nodes.columns and "y" in nodes.columns

    def test_layout_with_asymmetric_window(self):
        """Test layout with asymmetric window [past, future]."""
        plot = TemporalNetworkPlot(self.tg, layout="spring", layout_window_size=[1, 2])
        nodes = plot.data["nodes"]
        assert "x" in nodes.columns and "y" in nodes.columns

    def test_layout_with_all_past_window(self):
        """Test layout using all past time steps (negative value)."""
        plot = TemporalNetworkPlot(self.tg, layout="spring", layout_window_size=[-1, 1])
        nodes = plot.data["nodes"]
        assert "x" in nodes.columns and "y" in nodes.columns

    def test_layout_with_all_future_window(self):
        """Test layout using all future time steps (negative value)."""
        plot = TemporalNetworkPlot(self.tg, layout="spring", layout_window_size=[1, -1])
        nodes = plot.data["nodes"]
        assert "x" in nodes.columns and "y" in nodes.columns

    def test_layout_none_skips_computation(self):
        """Test that layout=None skips layout computation."""
        plot = TemporalNetworkPlot(self.tg, layout=None)
        nodes = plot.data["nodes"]
        # Should not have x, y coordinates (or they should be NaN)
        if "x" in nodes.columns:
            assert nodes["x"].isna().all()

    def test_invalid_layout_window_raises(self):
        """Test that invalid window size raises an error."""
        with pytest.raises(AttributeError):
            TemporalNetworkPlot(self.tg, layout="spring", layout_window_size="invalid")

    def test_layout_coordinates_normalized(self):
        """Test that layout coordinates are normalized to [0, 1]."""
        plot = TemporalNetworkPlot(self.tg, layout="spring")
        nodes = plot.data["nodes"]
        x_coords = nodes["x"].dropna()
        y_coords = nodes["y"].dropna()
        if len(x_coords) > 0:
            assert (x_coords >= 0).all() and (x_coords <= 1).all()
            assert (y_coords >= 0).all() and (y_coords <= 1).all()


class TestTemporalNetworkPlotEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_temporal_graph(self):
        """Test handling of empty temporal graph."""
        tg = TemporalGraph.from_edge_list([])
        plot = TemporalNetworkPlot(tg)
        assert len(plot.data["nodes"]) == 0
        assert len(plot.data["edges"]) == 0

    def test_single_time_step(self):
        """Test graph with edges at single time step."""
        tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 0)])
        plot = TemporalNetworkPlot(tg)
        nodes = plot.data["nodes"]
        edges = plot.data["edges"]
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_large_time_gap(self):
        """Test graph with large gaps in time steps."""
        tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 100)])
        plot = TemporalNetworkPlot(tg)
        nodes = plot.data["nodes"]
        # Nodes should span the time range
        max_end = nodes["end"].max()
        assert max_end > 100

    def test_node_attribute_from_network_data(self):
        """Test that node attributes from temporal graph data are used."""
        tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1)])
        tg.data.node_color = ["#ff0000", "#00ff00", "#0000ff"]
        plot = TemporalNetworkPlot(tg)
        nodes = plot.data["nodes"]
        # Should use network attributes
        assert set(nodes["color"].unique()).intersection({"#ff0000", "#00ff00", "#0000ff"})

    def test_edge_attribute_from_network_data(self):
        """Test that edge attributes from temporal graph data are used."""
        tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1)])
        tg.data.edge_size = torch.tensor([10, 20])
        plot = TemporalNetworkPlot(tg)
        edges = plot.data["edges"]
        # Should use network attributes
        assert set(edges["size"].unique()).issubset({10, 20})

    def test_simulation_mode_without_layout(self):
        """Test that simulation mode is enabled when no layout is specified."""
        plot = TemporalNetworkPlot(
            TemporalGraph.from_edge_list([("a", "b", 0)]), layout=None
        )
        assert plot.config["simulation"] is True

    def test_simulation_mode_with_layout(self):
        """Test that simulation mode is disabled when layout is specified."""
        plot = TemporalNetworkPlot(
            TemporalGraph.from_edge_list([("a", "b", 0)]), layout="spring"
        )
        assert plot.config["simulation"] is False

    def test_rgb_color_assignment(self):
        """Test that RGB tuple colors are converted to hex strings."""
        tg = TemporalGraph.from_edge_list(
            [("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "d", 3)]
        )
        node_colors = {
            "a": (255, 0, 0),
            ("b", 1): (0, 255, 0),
            ("c", 2): (0, 0, 255),
        }
        edge_colors = {
            ("a", "b", 0): (255, 255, 0),
            ("b", "c", 1): (0, 255, 255),
        }
        plot = TemporalNetworkPlot(tg, node_color=node_colors, edge_color=edge_colors)
        nodes = plot.data["nodes"]
        edges = plot.data["edges"]
        # Check that colors are hex strings
        for color in nodes["color"].dropna().unique():
            assert isinstance(color, str)
            assert color.startswith("#") and len(color) == 7
        for color in edges["color"].dropna().unique():
            assert isinstance(color, str)
            assert color.startswith("#") and len(color) == 7