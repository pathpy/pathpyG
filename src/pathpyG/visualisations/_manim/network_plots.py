"""Network plots with manim."""

# =============================================================================
# File      : network_plots.py -- Network plots with manim
# =============================================================================

from typing import Any

import logging
from pathpyG.visualisations._manim.core import ManimPlot
from manim import *
# create logger
logger = logging.getLogger("root")


class NetworkPlot(ManimPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.data = {}
        self.config = kwargs
        self.raw_data = data
        self.generate()
        

    def generate(self):
        # however we generate something -> Moritz
        self.data["data"] = {"nodes": self.raw_data.get("nodes", []), "edges": self.raw_data.get("edges", [])}


class TemporalNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "temporal"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        raise NotImplementedError



class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "static"
