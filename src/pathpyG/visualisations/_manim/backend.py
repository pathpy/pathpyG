"""Generic manim plot class."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

from manim import WHITE
from manim import config as manim_config

from pathpyG import config
from pathpyG.visualisations._manim.temporal_graph_scene import TemporalGraphScene
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot
from pathpyG.visualisations.utils import prepare_tempfile, unit_str_to_float

# create logger
logger = logging.getLogger("root")

SUPPORTED_KINDS: dict[type, str] = {
    TemporalNetworkPlot: "temporal",
}


class ManimBackend(PlotBackend):
    """Base class for Manim Plots integrated with Jupyter notebooks.

    This class defines the interface for Manim plots that are generated
    from data and can be rendered for either saving or displaying inline.
    """

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initializes the Manim backend with a given plot."""
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Matplotlib backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

        # Optional config settings
        manim_config.pixel_height = int(unit_str_to_float(self.config.get("height"), "px"))
        manim_config.pixel_width = int(unit_str_to_float(self.config.get("width"), "px"))
        manim_config.frame_rate = 15
        manim_config.quality = "high_quality"
        manim_config.background_color = self.config.get("background_color", WHITE)

    def render_video(
        self,
    ):
        """Renders the Manim animation.

        This method sets up the scene and prepares it for rendering.
        """
        temp_dir, current_dir = prepare_tempfile()
        manim_config.media_dir = temp_dir
        manim_config.output_file = "default.mp4"
        self.scene = TemporalGraphScene(data=self.data, config=self.config, show_labels=self.show_labels)
        self.scene.render()
        os.chdir(current_dir)
        return Path(temp_dir) / "videos" / "1080p60" / "default.mp4", temp_dir

    def save(self, filename: str) -> None:
        """Renders and saves a Manim animation to the working directory.

        This method creates a temporary scene using the instance's `raw data`,
        renders it with Manim, and saves the resulting video.

        Args:
            filename (str): Name for the File that will be saved. Is necessary for this function to work.

        Tip:
            - use `**kwargs` to control aspects of the scene such as animation timing, layout, or styling
        """
        # render temporary .mp4
        temp_file, temp_dir = self.render_video()
        if filename.endswith(".gif"):
            self.convert_to_gif(temp_file)
        shutil.copy(temp_file, filename)
        shutil.rmtree(temp_dir)

    def convert_to_gif(self, filename: str) -> None:
        """Convert the rendered mp4 video to a gif file."""
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    filename,
                    "-vf",
                    "fps=20,scale=720:-1:flags=lanczos",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    filename.replace(".mp4", ".gif"),
                ],
                check=True,
            )
        except Exception as e:
            logger.error(f"GIF conversion failed: {e}")

    def show(self) -> None:
        """Renders and displays a Manim animation.

        This method creates a temporary scene using the instance's `raw data`,
        renders it with Manim, and embeds the resulting video in the notebook.
        It is specifically for use in Juypter Environment
        and will warn if used elsewhere.

        Notes:
            - The scene is renderd into a temporary directory and not saved permanently
            - Manim is expected to output the video under `videos/1080p60/TemporalNetworkPlot.mp4` which is the default
        """
        temp_file, temp_dir = self.render_video()

        if config["environment"]["interactive"]:
            from IPython.display import HTML, display

            video_bytes = temp_file.read_bytes()
            video_b64 = base64.b64encode(video_bytes).decode()
            video_html = f"""
            <video width="580" height="340" controls>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            display(HTML(video_html))
        else:
            # open the file in the webbrowser
            webbrowser.open(r"file:///" + temp_file)
        shutil.rmtree(temp_dir)
