import unittest
import pytest
from manim import *
import tempfile
from pathlib import Path
from unittest.mock import patch
from matplotlib.pyplot import get_cmap

from pathpyG.visualisations._manim.core import ManimPlot
from pathpyG.visualisations._manim.network_plots import NetworkPlot
from pathpyG.visualisations._manim.network_plots import TemporalNetworkPlot
from pathpyG.visualisations._manim.network_plots import StaticNetworkPlot


class ManimTest(unittest.TestCase):
    """Manim Tests"""
    def setUp(self):     
        patcher = patch("manim.config")  # setting up patch for manim.config
        self.mock_config = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_config.media_dir = None  # initializing with default values
        self.mock_config.output_file = None
        self.mock_config.pixel_height = 0
        self.mock_config.pixel_width = 0
        self.mock_config.frame_rate = 0
        self.mock_config.quality = ""
        self.mock_config.background_color = None

    #def test_manim_plot(self):
    #   """Test ManimPlot class"""
        #manimplot = ManimPlot()

    def test_manim_network_plot(self):
        """Test for the NetworkPlot class"""
        data = {
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 2, "start": 0},
                      {"source": 1, "target": 2, "start": 1},
                      {"source": 2, "target": 0, "start": 2},
                      {"source": 1, "target": 3, "start": 3},
                      {"source": 3, "target": 2, "start": 4},
                      {"source": 2, "target": 1, "start": 5}]
        }
        networkplot = NetworkPlot(data)
        
        self.assertIsInstance(networkplot, NetworkPlot)
        self.assertIsInstance(networkplot.data, dict)
        self.assertIsInstance(networkplot.config, dict)
        self.assertIsInstance(networkplot.raw_data, dict)
        
        self.assertEqual(networkplot.data, {})
        self.assertEqual(networkplot.raw_data, data)

    def test_manim_network_plot_empty(self):
        """Test for the NetworkPlot class with empty input data"""
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
        data = {
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 2, "start": 0},
                      {"source": 1, "target": 2, "start": 1},
                      {"source": 2, "target": 0, "start": 2},
                      {"source": 1, "target": 3, "start": 3},
                      {"source": 3, "target": 2, "start": 4},
                      {"source": 2, "target": 1, "start": 5}]
        }
        kwargs = {
            "delta": 2000,
            "start": 100,
            "dynamic_layout_interval": 10,
            "node_color": RED,
            "edge_color": GREEN
        }
        tempnetworkplot = TemporalNetworkPlot(data, **kwargs)
        
        self.assertIsInstance(tempnetworkplot, TemporalNetworkPlot)
        self.assertIsInstance(tempnetworkplot, NetworkPlot)
        self.assertIsInstance(tempnetworkplot, Scene)

        self.assertIsInstance(tempnetworkplot.data, dict)
        self.assertIsInstance(tempnetworkplot.config, dict)
        self.assertIsInstance(tempnetworkplot.raw_data, dict)
        
        self.assertEqual(tempnetworkplot.data, data)

        self.assertEqual(tempnetworkplot.delta, 2000)
        self.assertEqual(tempnetworkplot.start, 100)
        self.assertEqual(tempnetworkplot.dynamic_layout_interval, 10)
        self.assertEqual(tempnetworkplot.node_color, RED)
        self.assertEqual(tempnetworkplot.edge_color, GREEN)

    def test_manim_temporal_network_plot_empty(self):
        """Test for the TemporalNetworkPlot class"""
        data = {}
        tempnetworkplot = TemporalNetworkPlot(data)
        
        self.assertIsInstance(tempnetworkplot, TemporalNetworkPlot)
        self.assertIsInstance(tempnetworkplot, NetworkPlot)
        self.assertIsInstance(tempnetworkplot, Scene)

        self.assertIsInstance(tempnetworkplot.data, dict)
        self.assertIsInstance(tempnetworkplot.config, dict)
        self.assertIsInstance(tempnetworkplot.raw_data, dict)
        
        self.assertEqual(tempnetworkplot.data, {})
        
        self.assertEqual(tempnetworkplot.delta, 1000)
        self.assertEqual(tempnetworkplot.start, 0)
        self.assertEqual(tempnetworkplot.end, None)
        self.assertEqual(tempnetworkplot.intervals, None)
        self.assertEqual(tempnetworkplot.dynamic_layout_interval, 5)
        self.assertEqual(tempnetworkplot.node_color, BLUE)
        self.assertEqual(tempnetworkplot.edge_color, GRAY)
        self.assertEqual(tempnetworkplot.node_cmap, get_cmap())
        self.assertEqual(tempnetworkplot.edge_cmap, get_cmap())
        self.assertEqual(tempnetworkplot.node_opacity, None)
        self.assertEqual(tempnetworkplot.edge_opacity, None)
        self.assertEqual(tempnetworkplot.node_size, None)
        self.assertEqual(tempnetworkplot.edge_size, None)
        self.assertEqual(tempnetworkplot.node_label, None)
        self.assertEqual(tempnetworkplot.edge_label, None)

    def test_manim_temp_np_path_config(self):
        """Test for the TemporalNetworkPlot class"""
        data = {
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 2, "start": 0},
                      {"source": 1, "target": 2, "start": 1},
                      {"source": 2, "target": 0, "start": 2},
                      {"source": 1, "target": 3, "start": 3},
                      {"source": 3, "target": 2, "start": 4},
                      {"source": 2, "target": 1, "start": 5}]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_file = "test_output.mp4"

            #with patch.dict("manim.config.__dict__", clear=False):
            tempnetworkplot = TemporalNetworkPlot(data, output_dir=output_dir, output_file=output_file)

            self.assertEqual(Path(self.mock_config.media_dir).resolve(), output_dir.resolve())
            self.assertEqual(self.mock_config.output_file, output_file)

            self.assertEqual(self.mock_config.pixel_height, 1080)
            self.assertEqual(self.mock_config.pixel_width, 1920)
            self.assertEqual(self.mock_config.frame_rate, 15)
            self.assertEqual(self.mock_config.quality, "medium_quality")
            self.assertEqual(self.mock_config.background_color, DARK_GREY)

    def test_manim_temp_np_edge_index(self):
        """Test for the method compute_edg_index in the TemporalNetworkPlot class"""
        data = {
            "nodes": [0, 1, 2, 3],
            "edges": [{"source": 0, "target": 2, "start": 0},
                      {"source": 1, "target": 2, "start": 1},
                      {"source": 2, "target": 0, "start": 2},
                      {"source": 1, "target": 3, "start": 3},
                      {"source": 3, "target": 2, "start": 4},
                      {"source": 2, "target": 1, "start": 5}]
        }
        edgelist = [(0, 2, 0), (1, 2, 1), (2, 0, 2), (1, 3, 3), (3, 2, 4), (2, 1, 5)]
        tempnetworkplot = TemporalNetworkPlot(data)

        self.assertEqual(tempnetworkplot.compute_edg_index()[0], edgelist)
        self.assertEqual(tempnetworkplot.compute_edg_index()[1], 5)

    #def test_manim_static_network_plot(self):
  #   """Test StaticNetworkPlot class"""


if __name__ == "__main__":
    unittest.main()
