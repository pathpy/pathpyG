import unittest
import torch
import pytest
from manim import *

from pathpyG.visualisations._manim.core import ManimPlot
from pathpyG.visualisations._manim.network_plots import NetworkPlot
from pathpyG.visualisations._manim.network_plots import TemporalNetworkPlot
from pathpyG.visualisations._manim.network_plots import StaticNetworkPlot


class ManimTest(unittest.TestCase):
    """Manim Tests"""
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
        #self.assertEqual(networkplot.config, kwargs)
        #self.assertEqual(networkplot._kind, "network")

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
        #self.assertEqual(networkplot.config, kwargs)

    def test_manim_temporal_network_plot(self):
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
        tempnetworkplot = TemporalNetworkPlot(data)
        
        self.assertIsInstance(tempnetworkplot, TemporalNetworkPlot)
        self.assertIsInstance(tempnetworkplot, NetworkPlot)
        self.assertIsInstance(tempnetworkplot, Scene)

        self.assertIsInstance(tempnetworkplot.data, dict)
        self.assertIsInstance(tempnetworkplot.config, dict)
        self.assertIsInstance(tempnetworkplot.raw_data, dict)
        
        self.assertEqual(tempnetworkplot.data, data)
        #self.assertEqual(tempnetworkplot.raw_data, data)
        #self.assertEqual(networkplot.config, kwargs)
        #self.assertEqual(tempnetworkplot._kind, "temporal")

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
        #self.assertEqual(tempnetworkplot.raw_data, data)
        #self.assertEqual(networkplot.config, kwargs)
        #self.assertEqual(tempnetworkplot._kind, "temporal")

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
        edgelist = [(0,2,0), (1,2,1), (2,0,2), (1,3,3), (3,2,4), (2,1,5)]
        tempnetworkplot = TemporalNetworkPlot(data)

        self.assertEqual(tempnetworkplot.compute_edg_index(), edgelist)

    #def test_manim_static_network_plot(self):
  #   """Test StaticNetworkPlot class"""


if __name__ == "__main__":
    unittest.main()
