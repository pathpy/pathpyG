import unittest
import torch
import pytest

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
        """Test NetworkPlot class"""
        data = {}  # add example data
        networkplot = NetworkPlot(data)
        self.assertIsInstance(networkplot, NetworkPlot)
        self.assertIsInstance(networkplot.data, dict)
        self.assertIsInstance(networkplot.config, dict)
        self.assertIsInstance(networkplot.raw_data, dict)
        self.assertEqual(networkplot.data, {})
        self.assertEqual(networkplot.raw_data, data)

  #def test_manim_temporal_network_plot(self):
  #   """Test TemporalNetworkPlot class"""

  #def test_manim_static_network_plot(self):
  #   """Test StaticNetworkPlot class"""


if __name__ == "__main__":
    unittest.main()
