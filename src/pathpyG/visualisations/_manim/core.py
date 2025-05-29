"""Generic manim plot class."""

from __future__ import annotations

import base64
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from IPython.core.getipython import get_ipython
from IPython.display import HTML, display

from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


def in_jupyter_notebook() -> bool:
    """
    Detects whether the current Python session is running inside
    a Jupyter Notebook.

    Returns:
        bool: True if running inside a Jupyter notebook, False otherwise
    """
    try:
        return "IPKernelApp" in get_ipython().config
    except NameError:
        return False
    except AttributeError:
        return False


class ManimPlot(PathPyPlot):
    """
    Base class for Manim Plots integrated with Jupyter notebooks

    This class defines the interface for Manim plots that are generated
    from data and can be rendered for either saving or displaying inline.
    """

    def generate(self) -> None:
        """
        Generate the plot.
        """
        raise NotImplementedError

    def save(self, filename: str, **kwargs: Any) -> None:
        """
        Save the rendered Manim plot to disk.
        """
        format = kwargs.get("save_as", "mp4").lower()
        save_dir = kwargs.get("save_dir", None)

        if save_dir is None:
            save_dir = Path.cwd()
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            temp_output_file = f"{self.__class__.__name__}"

            scene = self.__class__(data=self.raw_data, output_dir=tmp_path, output_file=temp_output_file, **kwargs)
            scene.render()

            video_path = tmp_path / "videos" / "720p30" / f"{temp_output_file}.{format}"

            if not video_path.exists():
                logger.warning("Rendered video not found at expected path: %s ", video_path)
                return

            target_path = save_dir / f"{filename}.{format}"

            shutil.copy(video_path, target_path)

    def show(self, **kwargs: Any) -> None:
        """
        Renders and displays a Manim animation within a Jupyter Notebook

        This method creates a temporary scene using the instance's `raw data`,
        renders it with Manim, and embeds the resulting video in the notebook.
        It is specifically for use in Juypter Environment
        and will warn if used elsewhere.

        Args:
            **kwargs (Any): Additional keyword arguments forwarded to the scene constructor.
            These can be used to customize the rendering behaviour or pass scene-specific parameters

        Notes:
            - The scene is renderd into a temporary directory and not saved permanently
            - Manim is expected to output the video under `videos/720p30/TemporalNetworkPlot.mp4` which is the default

        Tip:
            - use `**kwargs` to control aspects of the scene such as animation timing, layout, or styling
        """
        if not in_jupyter_notebook():
            logger.warning("This function is designed for use within a Jupyter notebook.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_file = f"{self.__class__.__name__}"

            scene = self.__class__(data=self.raw_data, output_dir=tmp_path, output_file=output_file, **kwargs)
            scene.render()

            video_dir = tmp_path / "videos" / "720p30"
            video_path = video_dir / f"{output_file}.mp4"

            if video_path.exists():
                video_bytes = video_path.read_bytes()
                video_b64 = base64.b64encode(video_bytes).decode()

                video_html = f"""
                <video width="580" height="340" controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """
                display(HTML(video_html))

                for folder in ["media", "videos", "images"]:
                    folder_path = tmp_path / folder
                    if folder_path.exists():
                        shutil.rmtree(folder_path)
            else:
                logger.warning("Expected video not found: %s", video_path)
