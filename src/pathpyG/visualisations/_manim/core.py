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

        raise NotImplementedError # manim is saved in scene ?

    def show(self, **kwargs: Any) -> None:
        raise NotImplementedError 
        """Render Manim Scene"""
        ''' import from wherever its created and render '''
        '''from pathpyG.visualisations._manim.testscene import NetworkScene
        from manim import config

        scene = NetworkScene(self.data["data"])
        scene.render('''
        

