"""Unit tests for Manim backend in pathpyG.visualisations."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._manim.backend import ManimBackend
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TestManimBackendInitialization:
    """Test ManimBackend initialization and configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple temporal graph
        tedges = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
        self.tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(self.tg)

    def test_backend_initialization_with_temporal_plot(self):
        """Test that ManimBackend initializes with TemporalNetworkPlot."""
        backend = ManimBackend(self.temp_plot, show_labels=True)
        assert backend is not None
        assert backend._kind == "temporal"
        assert backend.show_labels is True

    def test_backend_initialization_with_static_plot_raises(self):
        """Test that ManimBackend raises error with non-temporal plot."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c")])
        static_plot = NetworkPlot(g)
        
        with pytest.raises(ValueError, match="not supported"):
            ManimBackend(static_plot, show_labels=False)

    def test_backend_configures_manim_settings(self):
        """Test that ManimBackend configures manim settings correctly."""
        from manim import config as manim_config
        
        _ = ManimBackend(self.temp_plot, show_labels=False)
        
        # Check that manim config is set
        assert manim_config.pixel_height > 0
        assert manim_config.pixel_width > 0
        assert manim_config.quality == "high_quality"

    def test_backend_inherits_plot_data_and_config(self):
        """Test that backend has access to plot data and config."""
        backend = ManimBackend(self.temp_plot, show_labels=True)
        
        assert hasattr(backend, "data")
        assert hasattr(backend, "config")
        assert isinstance(backend.data, dict)
        assert isinstance(backend.config, dict)


class TestManimBackendRendering:
    """Test ManimBackend rendering and file operations."""

    def setup_method(self):
        """Set up test fixtures."""
        tedges = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
        self.tg = TemporalGraph.from_edge_list(tedges)
        self.temp_plot = TemporalNetworkPlot(self.tg)

    @patch("pathpyG.visualisations._manim.backend.TemporalGraphScene")
    def test_render_video_creates_scene(self, mock_scene_class):
        """Test that render_video creates and renders TemporalGraphScene."""
        mock_scene = MagicMock()
        mock_scene_class.return_value = mock_scene
        
        backend = ManimBackend(self.temp_plot, show_labels=True)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock the prepare_tempfile to use our temp dir
            with patch("pathpyG.visualisations._manim.backend.prepare_tempfile") as mock_prepare:
                mock_prepare.return_value = (tmp_dir, Path.cwd())
                
                backend.render_video()
                
                # Verify scene was created and rendered
                mock_scene_class.assert_called_once()
                mock_scene.render.assert_called_once()

    def test_save_mp4_creates_file(self):
        """Test that save() creates an MP4 file."""
        backend = ManimBackend(self.temp_plot, show_labels=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_output.mp4"
            
            # Mock render_video to return a test file
            with patch.object(backend, "render_video") as mock_render:
                temp_video = Path(tmp_dir) / "temp_video.mp4"
                temp_video.write_text("test video content")
                temp_subdir = Path(tmp_dir) / "temp"
                temp_subdir.mkdir()
                mock_render.return_value = (temp_video, temp_subdir)
                
                backend.save(str(output_file))
                
                # Verify file was created
                assert output_file.exists()
                assert output_file.read_text() == "test video content"

    def test_save_gif_calls_conversion(self):
        """Test that save() with .gif extension calls convert_to_gif."""
        backend = ManimBackend(self.temp_plot, show_labels=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_output.gif"
            
            # Mock render_video and convert_to_gif
            with patch.object(backend, "render_video") as mock_render, \
                 patch.object(backend, "convert_to_gif") as mock_convert:
                temp_video = Path(tmp_dir) / "temp_video.mp4"
                temp_video.write_text("test video")
                temp_gif = temp_video.with_suffix(".gif")
                temp_gif.write_text("test gif")
                temp_subdir = Path(tmp_dir) / "temp"
                temp_subdir.mkdir()
                mock_render.return_value = (temp_video, temp_subdir)
                
                backend.save(str(output_file))
                
                # Verify conversion was called
                mock_convert.assert_called_once_with(temp_video)

    @patch("subprocess.run")
    def test_convert_to_gif_calls_ffmpeg(self, mock_subprocess):
        """Test that convert_to_gif calls ffmpeg with correct arguments."""
        backend = ManimBackend(self.temp_plot, show_labels=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.mp4"
            test_file.write_text("video")
            
            backend.convert_to_gif(test_file)
            
            # Verify subprocess.run was called
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            
            # Check ffmpeg command structure
            assert call_args[0] == "ffmpeg"
            assert "-i" in call_args
            assert test_file in call_args
            assert test_file.with_suffix(".gif") in call_args


class TestManimBackendIntegration:
    """Integration tests for ManimBackend with real temporal networks."""

    def test_backend_with_minimal_temporal_network(self):
        """Test backend with minimal two-node temporal network."""
        tedges = [("a", "b", 0)]
        tg = TemporalGraph.from_edge_list(tedges)
        temp_plot = TemporalNetworkPlot(tg)
        
        backend = ManimBackend(temp_plot, show_labels=True)
        
        # Verify backend is properly initialized
        assert backend._kind == "temporal"
        assert "nodes" in backend.data
        assert "edges" in backend.data
