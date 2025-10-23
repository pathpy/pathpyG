"""Manim backend for high-quality temporal network animations.

Professional animation backend using Manim Community Edition for creating
high-quality temporal network visualizations. Optimized for temporal
graphs with smooth transitions and customizable animation parameters.

Features:
    - High-quality video output (MP4, GIF)
    - Temporal network animation with smooth transitions
    - Jupyter notebook integration with inline video display
    - FFmpeg integration for format conversion

## Workflow Overview

```mermaid
graph LR
    A[Graph Data] --> B[Manim Scene Creation]
    B --> C[Rendering]
    C --> D[MP4 Output]
    D --> E[Conversion]
    E --> F[GIF Output]
```
"""

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
    """Manim backend for temporal network animation.

    Integrates Manim Community Edition for creating smooth temporal network
    animations. Supports both MP4 and GIF output formats with Jupyter notebook
    integration for inline display.

    Features:
        - Temporal network animation with smooth node/edge transitions
        - Multiple output formats (MP4, GIF via FFmpeg)
        - Jupyter integration with base64 video embedding

    Example:
        Create and display a simple temporal network animation:
        ```python
        import pathpyG as pp
        
        tedges = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
        tg = pp.TemporalGraph.from_edge_list(tedges)
        pp.plot(tg, backend="manim", filename="temporal_network.gif")
        ```
        <img src="../../plot/temporal_network.gif" alt="Example Matplotlib Backend Output" width="550"/>

    !!! note "Temporal Networks Only"
        This backend is specifically designed for TemporalNetworkPlot
        objects and does not support static network visualization.

    !!! warning "Performance Requirements"
        High-quality animations require significant computational resources.
        Rendering time scales with network size, animation duration, and quality settings.
    """

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initialize Manim backend with temporal network validation and configuration.

        Sets up Manim configuration parameters including resolution, frame rate,
        quality settings, and background color. Validates that the plot type
        is supported (currently only TemporalNetworkPlot).

        Args:
            plot: PathPyPlot instance (must be TemporalNetworkPlot)
            show_labels: Whether to display node labels in animation

        Raises:
            ValueError: If plot type is not supported by Manim backend

        !!! info "Manim Configuration"
            Automatically configures Manim settings using pathpyG config and fixed defaults:
            
            - **Resolution**: From width/height config parameters
            - **Frame Rate**: Default 15 fps for smooth playback
            - **Quality**: High quality
            - **Background**: White background for clarity
        """
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Matplotlib backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

        # Optional config settings
        manim_config.pixel_height = int(unit_str_to_float(self.config.get("height"), "px"))  # type: ignore[arg-type]
        manim_config.pixel_width = int(unit_str_to_float(self.config.get("width"), "px"))  # type: ignore[arg-type]
        manim_config.quality = "high_quality"
        manim_config.background_color = WHITE

    def render_video(self) -> tuple[Path, str]:
        """Render temporal network animation using Manim engine.

        Creates temporary directory, configures Manim settings, instantiates
        TemporalGraphScene, and renders the complete animation sequence.
        Handles all Manim-specific setup and teardown.

        Returns:
            tuple: (video_file_path, temp_directory_path) for post-processing

        !!! info "Rendering Pipeline"
            1. **Setup**: Create temporary directory for Manim output
            2. **Configuration**: Set output path and filename
            3. **Scene Creation**: Instantiate TemporalGraphScene with data
            4. **Rendering**: Execute Manim rendering process
            5. **Cleanup**: Return paths for further processing and returns to original directory
        """
        temp_dir, current_dir = prepare_tempfile()
        manim_config.media_dir = temp_dir
        manim_config.output_file = "default.mp4"
        self.scene = TemporalGraphScene(data=self.data, config=self.config, show_labels=self.show_labels)
        self.scene.render()
        os.chdir(current_dir)
        return Path(temp_dir) / "videos" / "1080p60" / "default.mp4", temp_dir

    def save(self, filename: str) -> None:
        """Render and save temporal network animation to specified file.

        Creates high-quality animation video and saves to disk. Supports both
        MP4 and GIF formats with automatic format detection from filename
        extension. GIF conversion uses FFmpeg.

        Args:
            filename: Output file path with extension (.mp4 or .gif)

        !!! warning "GIF Conversion"
            GIF creation requires FFmpeg to be installed and available in PATH.
            Conversion may take additional time for long animations.
        """
        # render temporary .mp4
        temp_file, temp_dir = self.render_video()
        if filename.endswith(".gif"):
            self.convert_to_gif(temp_file)
            temp_file = temp_file.with_suffix(".gif")
        shutil.copy(temp_file, filename)
        shutil.rmtree(temp_dir)

    def convert_to_gif(self, filename: Path) -> None:
        """Convert rendered MP4 video to animated GIF using FFmpeg.

        Uses FFmpeg with optimized settings for web-friendly GIF output:
        30 fps for smooth animation, Lanczos scaling for quality preservation,
        and 1080p resolution maintenance.

        Args:
            filename: Path to source MP4 file (output GIF uses same path with .gif extension)

        Raises:
            Exception: If FFmpeg conversion fails (logged as error)
        """
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    filename,
                    "-vf",
                    "fps=30,scale=1080:-1:flags=lanczos",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    filename.with_suffix(".gif"),
                ],
                check=True,
            )
        except Exception as e:
            logger.error(f"GIF conversion failed: {e}")

    def show(self) -> None:
        """Display temporal network animation in interactive environment.

        Renders animation and displays inline in Jupyter notebooks using base64
        video embedding, or opens in system browser for non-interactive environments.
        Automatically cleans up temporary files after display.
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
            webbrowser.open(r"file:///" + temp_file.as_posix())
        shutil.rmtree(temp_dir)
