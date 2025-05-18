"""Generic manim plot class."""

from __future__ import annotations
import logging
from typing import Any
from pathlib import Path
from datetime import datetime
from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


class ManimPlot(PathPyPlot):
    """Base class for Manim Plots."""

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    def save(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def show(self, *, output_dir: str | Path = None, output_file: str = None, **kwargs: Any) -> None:
        """
        This function generates and saves a Manim Plot
        """

        output_dir = Path(output_dir or Path.cwd())
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.__class__.__name__}_{timestamp}"

        scene = self.__class__(data=self.raw_data, output_dir=output_dir, output_file=output_file, **kwargs)
        scene.render()
