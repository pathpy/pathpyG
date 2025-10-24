"""Unit tests for TimeUnfoldedNetworkPlot class in pathpyG.visualisations."""

import pandas as pd
import pytest

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations.unfolded_network_plot import TimeUnfoldedNetworkPlot


class TestTimeUnfoldedNetworkPlotInitialization:
    """Test TimeUnfoldedNetworkPlot initialization and basic functionality."""

    def setup_method(self):
        """Create a simple temporal graph for testing."""
        # Create temporal graph with edges at different times
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "b", 3)])

    def test_initialization(self):
        """Test that TimeUnfoldedNetworkPlot initializes correctly."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        assert plot.network is self.tg
        assert plot._kind == "unfolded"
        assert isinstance(plot.data, dict)

    def test_initialization_with_config_options(self):
        """Test initialization with various configuration options."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="right", node_color="#ff0000", edge_color="#0000ff")
        assert plot.config.get("orientation") == "right"
        assert plot.config["directed"] is True
        assert plot.config["curved"] is False


class TestTimeUnfoldedNetworkPlotNodeData:
    """Test node data structure and indexing."""

    def setup_method(self):
        """Create a temporal graph for testing."""
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "d", 3)])

    def test_node_data_has_correct_index(self):
        """Test that node data index includes uid and time information."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        nodes = plot.data["nodes"]

        # Index should be a Index with uid and time as tuples
        assert isinstance(nodes.index, pd.Index)
        # Should have (node_id, time) tuples
        assert len(nodes) > 0
        # Each row should have uid and time information in the tuple index
        assert all(isinstance(idx, tuple) and len(idx) == 2 for idx in nodes.index)

    def test_node_data_has_position_columns(self):
        """Test that node data includes x and y position columns."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        nodes = plot.data["nodes"]

        assert "x" in nodes.columns
        assert "y" in nodes.columns

    def test_node_positions_are_normalized(self):
        """Test that node positions are normalized to [0, 1]."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        nodes = plot.data["nodes"]

        x_coords = nodes["x"]
        y_coords = nodes["y"]

        # Coordinates should be between 0 and 1
        assert (x_coords >= 0).all() and (x_coords <= 1).all()
        assert (y_coords >= 0).all() and (y_coords <= 1).all()

    def test_node_data_includes_start_time(self):
        """Test that node data includes start time information."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        nodes = plot.data["nodes"]

        assert "start" in nodes.columns
        assert (nodes["start"] >= 0).all()

    def test_node_data_multiple_instances_per_node(self):
        """Test that each node appears at different time steps."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        nodes = plot.data["nodes"]

        # Get unique node IDs from index
        node_ids = set(idx[0] for idx in nodes.index)

        # Should have at least nodes a, b, c
        assert len(node_ids) >= 3


class TestTimeUnfoldedNetworkPlotEdgeData:
    """Test edge data structure and indexing."""

    def setup_method(self):
        """Create a temporal graph for testing."""
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "d", 3)])

    def test_edge_data_has_correct_index(self):
        """Test that edge data index includes source-time and target-time tuples."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        edges = plot.data["edges"]

        # Index should be a MultiIndex with source and target
        assert isinstance(edges.index, pd.MultiIndex)
        assert edges.index.names == ["source", "target"]

    def test_edge_index_includes_temporal_information(self):
        """Test that edge index preserves temporal information in tuples."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        edges = plot.data["edges"]

        # Index values should be tuples of (node, time)
        for source, target in edges.index:
            assert isinstance(source, tuple) and len(source) == 2
            assert isinstance(target, tuple) and len(target) == 2

    def test_edge_data_has_temporal_columns(self):
        """Test that edge data includes temporal columns."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        edges = plot.data["edges"]

        assert "start" in edges.columns
        assert "end" in edges.columns

    def test_edge_count_matches_temporal_graph(self):
        """Test that number of edges matches temporal graph edges."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        edges = plot.data["edges"]

        # Should have same number of edges as temporal graph
        assert len(edges) == len(self.tg.temporal_edges)


class TestTimeUnfoldedNetworkPlotLayout:
    """Test layout computation for time-unfolded networks."""

    def setup_method(self):
        """Create a temporal graph for layout testing."""
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2), ("a", "d", 3)])

    def test_layout_orientation_right(self):
        """Test layout with right orientation."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="right")
        nodes = plot.data["nodes"]
        x_values = nodes["x"].unique()
        y_values = nodes["y"].unique()

        # Should have multiple time steps (x values) and nodes (y values)
        for i in range(len(x_values) - 1):
            assert x_values[i] < x_values[i + 1]  # Increasing x for time steps
        assert len(y_values) > 1  # Multiple nodes

    def test_layout_orientation_left(self):
        """Test layout with left orientation."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="left")
        nodes = plot.data["nodes"]
        x_values = nodes["x"].unique()
        y_values = nodes["y"].unique()

        # X should decrease for time steps when orientation is left
        for i in range(len(x_values) - 1):
            assert x_values[i] > x_values[i + 1]  # Decreasing x for time steps
        assert len(y_values) > 1  # Multiple nodes

    def test_layout_orientation_up(self):
        """Test layout with up orientation."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="up")
        nodes = plot.data["nodes"]
        x_values = nodes["x"].unique()
        y_values = nodes["y"].unique()

        # Y should increase for time steps when orientation is up
        for i in range(len(y_values) - 1):
            assert y_values[i] < y_values[i + 1]  # Increasing y for time steps
        assert len(x_values) > 1  # Multiple nodes

    def test_layout_orientation_down(self):
        """Test layout with down orientation."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="down")
        nodes = plot.data["nodes"]
        x_values = nodes["x"].unique()
        y_values = nodes["y"].unique()

        # Y should decrease for time steps when orientation is down
        for i in range(len(y_values) - 1):
            assert y_values[i] > y_values[i + 1]  # Decreasing y for time steps
        assert len(x_values) > 1  # Multiple nodes

    def test_layout_invalid_orientation_raises(self):
        """Test that invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid orientation"):
            TimeUnfoldedNetworkPlot(self.tg, orientation="invalid")

    def test_layout_nodes_at_each_time_step(self):
        """Test that nodes appear at each time step in the unfolding."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="right")
        nodes = plot.data["nodes"]
        assert len(nodes) == self.tg.n * (self.tg.data.time.max() + 2)


class TestTimeUnfoldedNetworkPlotConfig:
    """Test configuration settings for time-unfolded plots."""

    def setup_method(self):
        """Create a temporal graph for testing."""
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2)])

    def test_orientation_default_is_set(self):
        """Test that orientation has a default value."""
        plot = TimeUnfoldedNetworkPlot(self.tg)
        assert plot.config.get("orientation") is not None

    def test_orientation_can_be_customized(self):
        """Test that orientation can be set in config."""
        plot = TimeUnfoldedNetworkPlot(self.tg, orientation="up")
        assert plot.config.get("orientation") == "up"


class TestTimeUnfoldedNetworkPlotAttributes:
    """Test node and edge attribute assignment."""

    def setup_method(self):
        """Create a temporal graph for attribute testing."""
        self.tg = TemporalGraph.from_edge_list([("a", "b", 0), ("b", "c", 1), ("c", "a", 2)])

    def test_node_constant_attributes(self):
        """Test assigning constant attributes to all nodes."""
        plot = TimeUnfoldedNetworkPlot(self.tg, node_color="#ff0000", node_size=10)
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
        plot = TimeUnfoldedNetworkPlot(self.tg, node_color=node_colors)
        nodes = plot.data["nodes"]

        # Should have assigned colors at specific times
        assert "#ff0000" in nodes["color"].values or "#00ff00" in nodes["color"].values

    def test_edge_constant_attributes(self):
        """Test assigning constant attributes to all edges."""
        plot = TimeUnfoldedNetworkPlot(self.tg, edge_color="#0000ff", edge_size=5)
        edges = plot.data["edges"]

        assert (edges["color"] == "#0000ff").all()
        assert (edges["size"] == 5).all()
