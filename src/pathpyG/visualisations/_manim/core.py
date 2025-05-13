"""Generic manim plot class."""

from __future__ import annotations

import logging

from typing import Any

from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")



class ManimPlot(PathPyPlot):
    """Base class for Manim Plots."""

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    def save(self, filename: str, **kwargs: Any) -> None:

        raise NotImplementedError

    def show(self, **kwargs: Any) -> None:
        """renders manim scene to current directory"""
        from manim import config

        config.pixel_height = 1080
        config.pixel_width = 1920
        config.frame_rate = 30
        config.quality = "medium_quality"
        config.output_file = kwargs.get("output_file", None)
        
        self.render()
        

