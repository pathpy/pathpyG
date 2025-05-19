"""Generic manim plot class."""

from __future__ import annotations
import logging
from typing import Any
from pathlib import Path
from datetime import datetime
from pathpyG.visualisations.plot import PathPyPlot
from IPython.display import Video, display
from IPython import get_ipython

# create logger
logger = logging.getLogger("root")


def in_jupyter_notebook() -> bool:
    try:

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


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

        if in_jupyter_notebook():
            video_path = output_dir / "videos" / "720p30" / f"{output_file}.mp4"
            if video_path.exists():
                display(Video(str(video_path), width=580, height=340))
            else:
                logger.warning(f"Expected video not found: {video_path}")
