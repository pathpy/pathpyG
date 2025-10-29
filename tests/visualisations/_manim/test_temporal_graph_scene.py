"""Unit tests for TemporalGraphScene in pathpyG.visualisations._manim."""

from unittest.mock import MagicMock, patch

import numpy as np

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._manim.temporal_graph_scene import TemporalGraphScene
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TestTemporalGraphSceneInitialization:
    """Test TemporalGraphScene initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple temporal network data
        tedges = [("a", "b", 0), ("b", "c", 1), ("c", "a", 2)]
        tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(tg, node_size=10)

    def test_scene_initialization_with_data(self):
        """Test that scene initializes with temporal network data."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=True
        )
        
        assert scene is not None
        assert hasattr(scene, "data")
        assert hasattr(scene, "config")
        assert scene.show_labels is True

    def test_scene_scales_node_sizes(self):
        """Test that scene scales node sizes appropriately."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )
        
        # Verify node size was scaled
        assert "radius" in scene.data["nodes"].columns
        # Original sizes (10) should be multiplied by 0.025
        for _, node in scene.data["nodes"].iterrows():
            assert node["radius"] == 10 * 0.025

    def test_scene_renames_node_columns(self):
        """Test that scene renames columns for manim compatibility."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )
        
        # Check renamed columns
        assert "radius" in scene.data["nodes"].columns
        assert "fill_color" in scene.data["nodes"].columns
        assert "fill_opacity" in scene.data["nodes"].columns

    def test_scene_renames_edge_columns(self):
        """Test that scene renames edge columns for manim compatibility."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )
        
        # Check renamed edge columns
        assert "stroke_color" in scene.data["edges"].columns
        assert "stroke_opacity" in scene.data["edges"].columns
        assert "stroke_width" in scene.data["edges"].columns

    def test_scene_scales_layout_coordinates(self):
        """Test that scene scales and centers layout coordinates."""
        # Create plot with explicit layout
        tedges = [("a", "b", 0), ("b", "c", 1)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg, layout="spring")
        
        scene = TemporalGraphScene(
            data=temp_plot.data,
            config=temp_plot.config,
            show_labels=False
        )
        
        # Verify layout was moved to center
        if "x" in scene.data["nodes"] and "y" in scene.data["nodes"]:
            x_coords = scene.data["nodes"]["x"]
            assert (x_coords > 0 ).any()
            assert (x_coords < 0 ).any()

            y_coords = scene.data["nodes"]["y"]
            assert (y_coords > 0 ).any()
            assert (y_coords < 0 ).any()

class TestTemporalGraphSceneBoundaryCalculation:
    """Test boundary point calculation for edge attachment."""

    def setup_method(self):
        """Set up test fixtures."""
        tedges = [("a", "b", 0)]
        tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(tg)
        self.scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )

    def test_get_boundary_point_basic(self):
        """Test boundary point calculation with simple inputs."""
        center = np.array([0, 0, 0])
        direction = np.array([1, 0, 0])
        radius = 0.5
        
        result = self.scene.get_boundary_point(center, direction, radius)
        
        # Should return point at radius distance in direction
        expected = np.array([0.5, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_boundary_point_diagonal(self):
        """Test boundary point calculation with diagonal direction."""
        center = np.array([0, 0, 0])
        direction = np.array([1, 1, 0])
        radius = 1.0
        
        result = self.scene.get_boundary_point(center, direction, radius)
        
        # Should be normalized and scaled by radius
        distance = np.linalg.norm(result - center)
        assert abs(distance - radius) < 0.001

    def test_get_boundary_point_zero_direction(self):
        """Test boundary point with zero direction vector."""
        center = np.array([1, 2, 3])
        direction = np.array([0, 0, 0])
        radius = 0.5
        
        result = self.scene.get_boundary_point(center, direction, radius)
        
        # Should return center when direction is zero
        np.testing.assert_array_equal(result, center)

    def test_get_boundary_point_negative_direction(self):
        """Test boundary point with negative direction."""
        center = np.array([0, 0, 0])
        direction = np.array([-2, 0, 0])
        radius = 1.0
        
        result = self.scene.get_boundary_point(center, direction, radius)
        
        # Should point in negative x direction
        expected = np.array([-1, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)


class TestTemporalGraphSceneConstruction:
    """Test scene construction and animation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        tedges = [("a", "b", 0), ("b", "c", 1), ("c", "a", 2)]
        tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(tg)

    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Create")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Transform")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.GrowArrow")
    def test_construct_creates_initial_nodes(self, mock_grow, mock_transform, mock_create):
        """Test that construct creates initial nodes."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )
        
        # Mock the scene methods
        scene.play = MagicMock()
        scene.wait = MagicMock()
        
        scene.construct()
        
        # Verify Create was called for initial nodes
        assert mock_create.called

    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Text")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Transform")
    def test_construct_updates_time_display(self, mock_transform, mock_text):
        """Test that construct updates time display."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )
        
        # Mock scene methods
        scene.play = MagicMock()
        scene.wait = MagicMock()
        
        scene.construct()
        
        # Verify Text was called for time display
        assert mock_text.called
        # Check that time text was created
        call_args = [call[0][0] if call[0] else "" for call in mock_text.call_args_list]
        assert any("Time:" in arg for arg in call_args)

    @patch("pathpyG.visualisations._manim.temporal_graph_scene.LabeledDot")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Dot")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Transform")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Create")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.Arrow")
    @patch("pathpyG.visualisations._manim.temporal_graph_scene.GrowArrow")
    def test_construct_with_labels(self, mock_grow, mock_arrow, mock_create, mock_transform, mock_dot, mock_labeled_dot):
        """Test construct with and without node labels."""
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=True
        )
        
        # Mock scene methods to prevent actual rendering
        scene.play = MagicMock()
        scene.wait = MagicMock()
        scene.get_boundary_point = MagicMock()
        
        # Should not raise any errors
        scene.construct()

        # Verify that LabeledDots were created for each node
        assert mock_labeled_dot.call_count == len(scene.data["nodes"])
        assert mock_dot.call_count == 0

        # Construct without labels
        scene = TemporalGraphScene(
            data=self.temp_plot.data,
            config=self.temp_plot.config,
            show_labels=False
        )

        # Mock scene methods to prevent actual rendering
        scene.play = MagicMock()
        scene.wait = MagicMock()
        scene.get_boundary_point = MagicMock()

        # Should not raise any errors
        scene.construct()

        # Verify that Dots were created for each node
        assert mock_dot.call_count == len(scene.data["nodes"])

    def test_construct_with_empty_network(self):
        """Test construct with network that has no edges at t=0."""
        # Create temporal network with edges starting later
        tedges = [("a", "b", 5), ("b", "c", 10)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg)
        
        scene = TemporalGraphScene(
            data=temp_plot.data,
            config=temp_plot.config,
            show_labels=False
        )
        
        # Mock scene methods
        scene.play = MagicMock()
        scene.wait = MagicMock()
        
        # Should handle empty initial state
        scene.construct()


class TestTemporalGraphSceneEdgeCases:
    """Test edge cases and error handling in TemporalGraphScene."""

    def test_scene_handles_duplicate_edges(self):
        """Test that scene handles duplicate edges gracefully."""
        # Create data with duplicate edges
        tedges = [("a", "b", 1), ("a", "b", 1), ("b", "c", 2)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg)
        
        scene = TemporalGraphScene(
            data=temp_plot.data,
            config=temp_plot.config,
            show_labels=False
        )
        
        # Mock scene methods
        scene.play = MagicMock()
        scene.wait = MagicMock()
        
        # Should handle duplicates without crashing
        scene.construct()

    def test_scene_with_custom_delta(self):
        """Test scene with custom delta parameter."""
        tedges = [("a", "b", 0), ("b", "c", 1)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg, delta=500)  # Custom delta
        
        scene = TemporalGraphScene(
            data=temp_plot.data,
            config=temp_plot.config,
            show_labels=False
        )
        
        # Verify config was set
        assert scene.config["delta"] == 500
