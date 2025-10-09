from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import webbrowser
from string import Template

from pathpyG import config
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.utils import unit_str_to_float, hex_to_rgb

# create logger
logger = logging.getLogger("root")

SUPPORTED_KINDS = {
    NetworkPlot: "static",
}


class TikzBackend(PlotBackend):
    """Backend for tikz/latex output."""

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initialize the backend with a plot."""
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Tikz backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save the plot to the hard drive."""
        if filename.endswith("tex"):
            with open(filename, "w+") as new:
                new.write(self.to_tex())
        elif filename.endswith("pdf"):
            # compile temporary pdf
            temp_file, temp_dir = self.compile_pdf()
            # Copy a file with new name
            shutil.copy(temp_file, filename)
            # remove the temporal directory
            shutil.rmtree(temp_dir)
        elif filename.endswith("svg"):
            # compile temporary svg
            temp_file, temp_dir = self.compile_svg()
            # Copy a file with new name
            shutil.copy(temp_file, filename)
            # remove the temporal directory
            shutil.rmtree(temp_dir)
        else:
            raise NotImplementedError

    def show(self) -> None:
        """Show the plot on the device."""
        # compile temporary pdf
        temp_file, temp_dir = self.compile_svg()

        if config["environment"]["interactive"]:
            from IPython.display import SVG, display

            # open the file, read the content and display it
            # workaround because it is not possible to embed files in vs code
            # https://github.com/microsoft/vscode-jupyter/discussions/13769
            with open(temp_file, "r") as svg_file:
                svg = SVG(svg_file.read())
            display(svg)
        else:
            # open the file in the webbrowser
            webbrowser.open(r"file:///" + temp_file)

        # Wait for .1 second before temp file is deleted
        time.sleep(0.1)

        # remove the temporal directory
        shutil.rmtree(temp_dir)

    def compile_svg(self) -> tuple:
        """Compile svg from tex."""
        temp_dir, current_dir, basename = self.prepare_compile()

        # latex compiler
        command = [
            "latexmk",
            "--interaction=nonstopmode",
            basename + ".tex",
        ]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error("latexmk compiler failed with output:\n%s", e.output.decode())
            raise AttributeError from e

        # dvisvgm command
        command = [
            "dvisvgm",
            basename + ".dvi",
            "-o",
            basename + ".svg",
        ]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error("dvisvgm command failed with output:\n%s", e.output.decode())
            raise AttributeError from e
        finally:
            # change back to the current directory
            os.chdir(current_dir)

        # return the name of the folder and temp svg file
        return os.path.join(temp_dir, basename + ".svg"), temp_dir

    def compile_pdf(self) -> tuple:
        """Compile pdf from tex."""
        temp_dir, current_dir, basename = self.prepare_compile()

        # latex compiler
        command = [
            "latexmk",
            "--pdf",
            "-shell-escape",
            "--interaction=nonstopmode",
            basename + ".tex",
        ]

        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error("latexmk compiler failed with output:\n%s", e.output.decode())
            raise AttributeError from e
        finally:
            # change back to the current directory
            os.chdir(current_dir)

        # return the name of the folder and temp pdf file
        return os.path.join(temp_dir, basename + ".pdf"), temp_dir

    def prepare_compile(self) -> tuple[str, str, str]:
        """Prepare compilation of tex to pdf or svg by saving the tex file."""
        # basename
        basename = "default"
        # get current directory
        current_dir = os.getcwd()

        # get temporal directory
        temp_dir = tempfile.mkdtemp()

        # change to output dir
        os.chdir(temp_dir)

        # save the tex file
        self.save(basename + ".tex")
        return temp_dir, current_dir, basename

    def to_tex(self) -> str:
        """Convert data to tex."""
        # get path to the pathpy templates
        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            os.path.normpath("_tikz/templates"),
        )

        # get template files
        with open(os.path.join(template_dir, f"{self._kind}.tex")) as template:
            tex_template = template.read()

        # generate data
        data = self.to_tikz()

        # fill template with data
        tex = Template(tex_template).substitute(
            classoptions=self.config.get("latex_class_options"),
            width=unit_str_to_float(self.config.get("width"), "cm"),
            height=unit_str_to_float(self.config.get("height"), "cm"),
            margin=self.config.get("margin"),
            tikz=data,
        )

        return tex

    def to_tikz(self) -> str:
        """Convert to Tex."""
        tikz = ""
        # generate node strings
        node_strings = "\\Vertex["
        # show labels if specified
        if self.show_labels:
            node_strings += "label=" + self.data["nodes"].index.astype(str) + ","
            node_strings += (
                "fontsize=\\fontsize{" + str(int(0.75 * self.data["nodes"]["size"].mean())) + "}{10}\selectfont,"
            )
        # Convert hex colors to rgb if necessary
        if self.data["nodes"]["color"].str.startswith("#").all():
            self.data["nodes"]["color"] = self.data["nodes"]["color"].map(hex_to_rgb)
            node_strings += "RGB,color={" + self.data["nodes"]["color"].astype(str).str.strip("()") + "},"
        else:
            node_strings += "color=" + self.data["nodes"]["color"] + ","
        # add other options
        node_strings += "size=" + (self.data["nodes"]["size"] * 0.05).astype(str) + ","
        node_strings += "opacity=" + self.data["nodes"]["opacity"].astype(str) + ","
        # add position
        node_strings += (
            "x=" + ((self.data["nodes"]["x"] - 0.5) * unit_str_to_float(self.config["width"], "cm")).astype(str) + ","
        )
        node_strings += (
            "y=" + ((self.data["nodes"]["y"] - 0.5) * unit_str_to_float(self.config["height"], "cm")).astype(str) + "]"
        )
        # add node name
        node_strings += "{" + self.data["nodes"].index.astype(str) + "}\n"
        tikz += node_strings.str.cat()

        # generate edge strings
        edge_strings = "\\Edge["
        if self.config["directed"]:
            edge_strings += "bend=15,Direct,"
        if self.data["edges"]["color"].str.startswith("#").all():
            self.data["edges"]["color"] = self.data["edges"]["color"].map(hex_to_rgb)
            edge_strings += "RGB,color={" + self.data["edges"]["color"].astype(str).str.strip("()") + "},"
        else:
            edge_strings += "color=" + self.data["edges"]["color"] + ","
        edge_strings += "lw=" + self.data["edges"]["size"].astype(str) + ","
        edge_strings += "opacity=" + self.data["edges"]["opacity"].astype(str) + "]"
        edge_strings += (
            "(" + self.data["edges"]["source"].astype(str) + ")(" + self.data["edges"]["target"].astype(str) + ")\n"
        )
        tikz += edge_strings.str.cat()

        return tikz
