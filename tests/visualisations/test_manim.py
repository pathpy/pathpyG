import unittest
from unittest.mock import patch, MagicMock
import tempfile
import subprocess
from pathlib import Path
from manim import Scene, manim_colors, Graph, tempconfig
import numpy as np
import pathpyG as pp

import pathpyG.visualisations._manim.core as core
from pathpyG.visualisations._manim.network_plots import NetworkPlot
from pathpyG.visualisations._manim.network_plots import TemporalNetworkPlot


class ManimTest(unittest.TestCase):
    """Test class for manim visualizations"""
    def setUp(self):
        """setting up patch for manim.config and example input data"""
        patcher = patch("manim.config")
        self.mock_config = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_config.media_dir = None  # initializing with default values
        self.mock_config.output_file = None
        self.mock_config.pixel_height = 0
        self.mock_config.pixel_width = 0
        self.mock_config.frame_rate = 0
        self.mock_config.quality = ""
        self.mock_config.background_color = None

        self.data = {  # example input data
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 2, "start": 0},
                      {"source": 1, "target": 2, "start": 1},
                      {"source": 2, "target": 0, "start": 2},
                      {"source": 1, "target": 3, "start": 3},
                      {"source": 3, "target": 2, "start": 4},
                      {"source": 2, "target": 1, "start": 5}]
        }

        self.data3 = {
            "nodes": [{"uid": "A", "label": "Alpha"}],
            "edges": [{"source": "A", "target": "A", "start": 0, "end": 1}],
        }

    def test_manim_network_plot(self):
        """Test for initializing the NetworkPlot class"""
        networkplot = NetworkPlot(self.data)

        self.assertIsInstance(networkplot, NetworkPlot)
        self.assertIsInstance(networkplot.data, dict)
        self.assertIsInstance(networkplot.config, dict)
        self.assertIsInstance(networkplot.raw_data, dict)

        self.assertEqual(networkplot.data, {})
        self.assertEqual(networkplot.raw_data, self.data)

    def test_manim_network_plot_empty(self):
        """Test for initializing the NetworkPlot class with empty input data"""
        data = {}
        networkplot = NetworkPlot(data)

        self.assertIsInstance(networkplot, NetworkPlot)
        self.assertIsInstance(networkplot.data, dict)
        self.assertIsInstance(networkplot.config, dict)
        self.assertIsInstance(networkplot.raw_data, dict)

        self.assertEqual(networkplot.data, {})
        self.assertEqual(networkplot.raw_data, {})

    def test_manim_temporal_network_plot(self):
        """Test for initializing the TemporalNetworkPlot class"""
        kwargs = {
            "delta": 2000,
            "start": 100,
            "end": 10000,
            "intervals": 30,
            "dynamic_layout_interval": 10,
            "background_color": "#3A1F8C",
            "font_size": 12,

            "node_opacity": 0.6,
            "node_size": 5.2,
            "node_label": {0: 'x', 1: "A", 2: ">", 3: "x"},
            "node_color": (255, 0, 0),
            "node_color_timed": [(0, (1, 0.5)), (1, (2, 0.2))],

            "edge_opacity": 0.75,
            "edge_size": 4.0,
            "edge_color": ['blue', 'pink']
        }
        temp_network_plot = TemporalNetworkPlot(self.data, **kwargs)

        self.assertIsInstance(temp_network_plot, TemporalNetworkPlot)
        self.assertIsInstance(temp_network_plot, NetworkPlot)
        self.assertIsInstance(temp_network_plot, Scene)

        self.assertIsInstance(temp_network_plot.data, dict)
        self.assertIsInstance(temp_network_plot.config, dict)
        self.assertIsInstance(temp_network_plot.raw_data, dict)

        self.assertEqual(temp_network_plot.delta, 2000)
        self.assertEqual(temp_network_plot.start, 100)
        self.assertEqual(temp_network_plot.end, 10000)
        self.assertEqual(temp_network_plot.intervals, 30)
        self.assertEqual(temp_network_plot.dynamic_layout_interval, 10)
        self.assertEqual(temp_network_plot.config.get("background_color"), "#3A1F8C")
        self.assertEqual(temp_network_plot.config.get("font_size"), 12)

        self.assertEqual(temp_network_plot.config.get("node_opacity"), 0.6)
        self.assertEqual(temp_network_plot.config.get("node_size"), 5.2)
        self.assertEqual(temp_network_plot.config.get("node_label"), {0: 'x', 1: "A", 2: ">", 3: "x"})
        self.assertEqual(temp_network_plot.config.get("node_color"), (255, 0, 0))
        self.assertEqual(temp_network_plot.config.get("node_color_timed"), [(0, (1, 0.5)), (1, (2, 0.2))])

        self.assertEqual(temp_network_plot.config.get("edge_opacity"), 0.75)
        self.assertEqual(temp_network_plot.config.get("edge_size"), 4.0)
        self.assertEqual(temp_network_plot.config.get("edge_color"), ['blue', 'pink'])

    def test_manim_temporal_network_plot_empty(self):
        """Test for initializing the TemporalNetworkPlot class with empty input data"""
        data = {}
        tempnetworkplot = TemporalNetworkPlot(data)

        self.assertIsInstance(tempnetworkplot, TemporalNetworkPlot)
        self.assertIsInstance(tempnetworkplot, NetworkPlot)
        self.assertIsInstance(tempnetworkplot, Scene)

        self.assertIsInstance(tempnetworkplot.data, dict)
        self.assertIsInstance(tempnetworkplot.config, dict)
        self.assertIsInstance(tempnetworkplot.raw_data, dict)

        self.assertEqual(tempnetworkplot.delta, 1000)
        self.assertEqual(tempnetworkplot.start, 0)
        self.assertEqual(tempnetworkplot.end, None)
        self.assertEqual(tempnetworkplot.intervals, None)
        self.assertEqual(tempnetworkplot.dynamic_layout_interval, None)
        self.assertEqual(tempnetworkplot.font_size, 8)

        self.assertEqual(tempnetworkplot.node_opacity, 1)
        self.assertEqual(tempnetworkplot.edge_opacity, 1)
        self.assertEqual(tempnetworkplot.node_size, 0.4)
        self.assertEqual(tempnetworkplot.edge_size, 0.4)
        self.assertEqual(tempnetworkplot.node_label, {})#

    def test_manim_temp_np_mock_config(self):
        """Test for the TemporalNetworkPlot class"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = "test_output.mp4"

            _ = TemporalNetworkPlot(self.data, output_dir=output_dir, output_file=output_file)

            self.assertEqual(Path(self.mock_config.media_dir).resolve(), output_dir.resolve())
            self.assertEqual(self.mock_config.output_file, output_file)

            self.assertEqual(self.mock_config.pixel_height, 1080)
            self.assertEqual(self.mock_config.pixel_width, 1920)
            self.assertEqual(self.mock_config.frame_rate, 15)
            self.assertEqual(self.mock_config.quality, "high_quality")
            self.assertEqual(self.mock_config.background_color, manim_colors.WHITE)

    def test_manim_temp_np_path(self):
        """Test for the TemporalNetworkPlot class"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = "test_output.mp4"

            _ = TemporalNetworkPlot(self.data, output_dir=output_dir, output_file=output_file)

            from manim import config as manim_config

            self.assertEqual(Path(manim_config.media_dir).resolve(), output_dir.resolve())
            self.assertEqual(manim_config.output_file, output_file)

    def test_manim_temp_np_edge_index(self):
        """Test for the method compute_edge_index in the TemporalNetworkPlot class"""
        edgelist = [(0, 2, 0), (1, 2, 1), (2, 0, 2), (1, 3, 3), (3, 2, 4), (2, 1, 5)]
        temp_network_plot = TemporalNetworkPlot(self.data)

        self.assertEqual(temp_network_plot.compute_edge_index()[0], edgelist)
        self.assertEqual(temp_network_plot.compute_edge_index()[1], 5)

    def test_manim_temp_np_layout(self):
        """"Test for the method get_layout in the TemporalNetworkPlot class"""
        edgelist = [(0, 2, 0), (1, 2, 1), (2, 0, 2), (1, 3, 3), (3, 2, 4), (2, 1, 5)]
        nodes = {0, 1, 2, 3}
        graph = pp.TemporalGraph.from_edge_list(edgelist)
        temp_network_plot = TemporalNetworkPlot(self.data)
        old_layout = {node: [0.0, 0.0] for node in graph.nodes}

        layout = temp_network_plot.get_layout(graph, old_layout=old_layout)

        self.assertIsInstance(layout, dict)
        self.assertEqual(layout.keys(), nodes)

        for coordinate in layout.values():
            self.assertIsInstance(coordinate, np.ndarray)
            self.assertEqual(coordinate.shape, (3,))

    def test_manim_temp_np_get_color_at_time(self):
        """Test for the method get_color_at_time int the TemporalNetworkPlot class"""
        temp_network_plot = TemporalNetworkPlot(self.data)
        node_data = {"color": manim_colors.RED}
        self.assertEqual(temp_network_plot.get_color_at_time(node_data, 0), manim_colors.RED)
        self.assertEqual(temp_network_plot.get_color_at_time({}, 0), manim_colors.BLUE)

        node_data_2 = {
            "color": manim_colors.PURPLE,
            "color_change": [
                {"time": 5, "color": manim_colors.TEAL},
                {"time": 10, "color": manim_colors.GREEN},
            ]
        }
        self.assertEqual(temp_network_plot.get_color_at_time(node_data_2, 1), manim_colors.PURPLE)
        self.assertEqual(temp_network_plot.get_color_at_time(node_data_2, 5), manim_colors.TEAL)
        self.assertEqual(temp_network_plot.get_color_at_time(node_data_2, 7), manim_colors.TEAL)
        self.assertEqual(temp_network_plot.get_color_at_time(node_data_2, 10), manim_colors.GREEN)
        self.assertEqual(temp_network_plot.get_color_at_time(node_data_2, 11), manim_colors.GREEN)

    @patch.object(TemporalNetworkPlot, "compute_edge_index")
    @patch.object(TemporalNetworkPlot, "get_layout")
    @patch.object(TemporalNetworkPlot, "get_color_at_time")
    def test_manim_temp_np_construct(self, mock_color, mock_layout, mock_edge_index):
        """Test for the construct method from the TemporalNetworkPlot class"""
        with tempfile.TemporaryDirectory() as tmp_path:
            with tempconfig({
                "disable_caching": True,
                "dry_run": True,
                "disable_output": True,
                "media_dir": str(tmp_path),
            }):
                mock_edge_index.return_value = ([("A", "A", 0)], 1)
                mock_layout.return_value = {"A": np.array([0, 0, 0])}
                mock_color.return_value = (0, 0, 1)

                temp_network_plot = TemporalNetworkPlot(self.data3)

                temp_network_plot.add = MagicMock()
                temp_network_plot.play = MagicMock()
                temp_network_plot.wait = MagicMock()
                temp_network_plot.remove = MagicMock()

                temp_network_plot.construct()

                self.assertTrue(temp_network_plot.add.called)
                self.assertTrue(temp_network_plot.wait.called)
                self.assertTrue(temp_network_plot.remove.called)
                self.assertTrue(any(isinstance(call[0][0], Graph) for call in temp_network_plot.add.call_args_list))

    def test_manim_plot_save_mp4(self):
        """Test for saving a mp4 file with the method save from the ManimPlot class"""
        with tempfile.TemporaryDirectory() as tmp_scene_dir, tempfile.TemporaryDirectory() as tmp_save_dir:

            scene_output = Path(tmp_scene_dir)
            save_output = Path(tmp_save_dir)
            output_file = "TemporalNetworkPlot"

            manim_plot = TemporalNetworkPlot(self.data, output_dir=scene_output, output_file=output_file)

            with patch.object(Scene, "render", new=render_side_effect):
                manim_plot.save("testvideo.mp4", save_dir=save_output)

            target_path = save_output / "testvideo.mp4"
            self.assertTrue(target_path.exists())
            self.assertEqual(target_path.read_text(), "test video")

    def test_manim_plot_save_gif(self):
        """Test for saving a gif with the method save from the ManimPlot class"""
        with tempfile.TemporaryDirectory() as tmp_scene_dir, tempfile.TemporaryDirectory() as tmp_save_dir:

            scene_output = Path(tmp_scene_dir)
            save_output = Path(tmp_save_dir)
            output_file = "TemporalNetworkPlot"

            manim_plot = TemporalNetworkPlot(self.data, output_dir=scene_output, output_file=output_file)

            with patch.object(Scene, "render", new=render_side_effect_gif):
                manim_plot.save("testvideo.gif", save_dir=save_output, save_as=format)

            target_path = save_output / "testvideo.gif"
            self.assertTrue(target_path.exists())
            #self.assertEqual(target_path.read_text(), "test video")        

    @patch("pathpyG.visualisations._manim.core.display")
    @patch.object(core, "in_jupyter_notebook", return_value=True)
    def test_manim_plot_show(self, mock_jupyter, mock_display):
        """Test for the method show from the ManimPlot class"""
        manim_plot = TemporalNetworkPlot(self.data)

        with patch.object(Scene, "render", new=render_side_effect):
            manim_plot.show()

        mock_display.assert_called_once()


def render_side_effect(*args):
    """Mocking Scene.render"""
    from manim import config as manim_config

    output_dir = Path(manim_config.media_dir) if manim_config.media_dir else Path.cwd()

    video_dir = output_dir / "videos" / "1080p60"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_file = video_dir / f"{TemporalNetworkPlot.__name__}.mp4"
    video_file.write_text("test video")


def render_side_effect_gif(*args):
    """Mocking Scene.render"""
    from manim import config as manim_config

    output_dir = Path(manim_config.media_dir) if manim_config.media_dir else Path.cwd()

    video_dir = output_dir / "videos" / "1080p60"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_file = video_dir / f"{TemporalNetworkPlot.__name__}.mp4"

    command = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(video_file)
    ]
    subprocess.run(command, check=True)
