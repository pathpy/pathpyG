"""TikZ/LaTeX Backend for High-Quality Network Visualizations.

This backend generates publication-ready vector graphics using LaTeX's TikZ package.
It provides precise control over visual elements and produces scalable output suitable
for academic papers, presentations, and professional documentation.

!!! abstract "Backend Capabilities"
    - **Static networks only** - Temporal networks not supported
    - **Vector output** - SVG, PDF, and raw TeX formats
    - **LaTeX compilation** - Automatic document generation and compilation
    - **Custom styling** - Full control over colors, sizes, and layouts

The backend handles the complete workflow from graph data to compiled output,
including template processing, LaTeX compilation, and format conversion.

## Workflow Overview

```mermaid
graph LR
    A[Graph Data] --> B[TikZ Template]
    B --> C[LaTeX Document]
    C --> D[Compilation]
    D --> E[PDF Output]
    D --> F[DVI Output]
    F --> H[Conversion]
    H --> I[SVG Output]
    C --> G[TeX Output]
```

!!! tip "Performance Considerations"
    - Compilation time scales with network complexity
    - Large networks (>500 nodes) may require significant processing time
    - Consider `matplotlib` backend for rapid prototyping of complex networks
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import webbrowser
from string import Template

import pandas as pd

from pathpyG import config
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.unfolded_network_plot import TimeUnfoldedNetworkPlot
from pathpyG.visualisations.utils import hex_to_rgb, prepare_tempfile, unit_str_to_float

# create logger
logger = logging.getLogger("root")

SUPPORTED_KINDS = {
    NetworkPlot: "static",
    TimeUnfoldedNetworkPlot: "unfolded",
}


class TikzBackend(PlotBackend):
    """TikZ/LaTeX Backend for Publication-Quality Network Graphics.
    
    Generates high-quality vector graphics using LaTeX's TikZ package. 
    The backend mainly uses the [`tikz-network`](https://github.com/hackl/tikz-network)
    package to create detailed and customizable visualizations. This backend
    is optimized for static networks and provides publication-ready output with
    precise control over visual elements.
    
    !!! info "Supported Operations"
        - **Formats**: SVG, PDF, TeX
        - **Networks**: Static graphs only
        - **Styling**: Full customization support
        - **Layouts**: All pathpyG layout algorithms
    
    The backend automatically handles LaTeX compilation, temporary file management,
    and format conversion to deliver clean, scalable graphics suitable for
    academic publications and professional presentations.
    
    Attributes:
        plot: The PathPyPlot instance containing graph data and configuration
        show_labels: Whether to display node labels in the output
        _kind: Type of plot being processed (for now only "static" supported)
    
    Example:
        ```python
        # The backend is typically used via pp.plot()
        import pathpyG as pp
        g = pp.Graph.from_edge_list([("A", "B"), ("B", "C")])
        pp.plot(g, backend="tikz")
        ```
        <img src="../../plot/tikz_backend_example.svg" alt="Example TikZ Backend Output" width="550"/>
    """

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initialize the TikZ backend with plot data and configuration.
        
        Sets up the backend to process the provided plot data and validates
        that the plot type is supported by the TikZ backend.
        
        Args:
            plot: PathPyPlot instance containing graph data, layout, and styling
            show_labels: Whether to display node labels in the generated output
            
        Raises:
            ValueError: If the plot type is not supported by the TikZ backend
            
        Note:
            Currently only static NetworkPlot instances are supported.
            Temporal networks require, e.g. the manim backend instead.
        """
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)  # type: ignore[arg-type]
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Tikz backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save the network visualization to a file in the specified format.
        
        Automatically detects the output format from the file extension and
        performs the necessary compilation steps. Supports TeX (raw LaTeX),
        PDF (compiled document), and SVG (vector graphics) formats.
        
        Args:
            filename: Output file path with extension (.tex, .pdf, or .svg)
            
        Raises:
            NotImplementedError: If the file extension is not supported
            
        Note:
            PDF and SVG compilation requires LaTeX toolchain installation.
            The method handles temporary file creation and cleanup automatically.
        """
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
        """Display the network visualization in the current environment.
        
        Compiles the network to SVG format and displays it either inline
        (in Jupyter notebooks) or opens it in the default web browser.
        The display method is automatically chosen based on the environment.
        
        The method creates temporary files for compilation and cleans them
        up automatically after display.
        
        Environment Detection:
            - **Interactive (Jupyter)**: Displays SVG inline using IPython.display
            - **Non-interactive**: Opens SVG file in default web browser
            
        Note:
            Requires LaTeX toolchain with TikZ and dvisvgm for SVG compilation.
            Temporary files are automatically cleaned up after a brief delay.
        """
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
        """Compile LaTeX source to SVG format using the LaTeX toolchain.
        
        Performs a complete compilation workflow: TeX → DVI → SVG conversion.
        Uses latexmk for robust LaTeX compilation and dvisvgm for high-quality
        SVG conversion with proper text rendering.
        
        Returns:
            tuple: (svg_file_path, temp_directory_path) for the compiled SVG
            
        Raises:
            AttributeError: If LaTeX compilation fails or required tools are missing
            
        Compilation Steps:
            1. Generate temporary directory and save TeX source
            2. Run latexmk to compile TeX → DVI
            3. Use dvisvgm to convert DVI → SVG
            4. Return paths for file access and cleanup
            
        Note:
            Both latexmk and dvisvgm must be available in the system PATH.
        """
        temp_dir, current_dir = prepare_tempfile()
        # save the tex file
        self.save("default.tex")

        # latex compiler
        command = [
            "latexmk",
            "--interaction=nonstopmode",
            "default.tex",
        ]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error("latexmk compiler failed with output:\n%s", e.output.decode())
            raise AttributeError from e

        # dvisvgm command
        command = [
            "dvisvgm",
            "default.dvi",
            "-o",
            "default.svg",
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
        return os.path.join(temp_dir, "default.svg"), temp_dir

    def compile_pdf(self) -> tuple:
        """Compile LaTeX source to PDF format using pdflatex.
        
        Generates a high-quality PDF document suitable for printing and
        publication. Uses latexmk with PDF mode for robust compilation
        and automatic dependency handling.
        
        Returns:
            tuple: (pdf_file_path, temp_directory_path) for the compiled PDF
            
        Raises:
            AttributeError: If LaTeX compilation fails or pdflatex is not available

        Note:
            Requires latexmk and a PDF-capable LaTeX engine (pdflatex, xelatex, etc.).
        """
        temp_dir, current_dir = prepare_tempfile()
        # save the tex file
        self.save("default.tex")

        # latex compiler
        command = [
            "latexmk",
            "--pdf",
            "-shell-escape",
            "--interaction=nonstopmode",
            "default.tex",
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
        return os.path.join(temp_dir, "default.pdf"), temp_dir

    def to_tex(self) -> str:
        """Generate complete LaTeX document with TikZ network visualization.
        
        Combines the network data with a LaTeX template to create a complete
        document ready for compilation. The template includes all necessary
        packages, document setup, and TikZ drawing commands.
        
        Returns:
            str: Complete LaTeX document source code
            
        Process:
            1. **Load template** - Retrieves the appropriate template for the plot type
            2. **Generate TikZ** - Converts network data to TikZ drawing commands  
            3. **Template substitution** - Fills template variables with graph data
            4. **Return final string** - Complete LaTeX document ready for compilation
            
        Template Variables:
            - `$classoptions`: LaTeX class options
            - `$width`, `$height`: Document dimensions
            - `$margin`: Margin around the drawing area
            - `$tikz`: TikZ drawing commands for nodes and edges
            
        Note:
            The generated document is self-contained and includes all necessary
            TikZ packages and configuration for network visualization.
        """
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
            width=unit_str_to_float(self.config.get("width"), "cm"),  # type: ignore[arg-type]
            height=unit_str_to_float(self.config.get("height"), "cm"),  # type: ignore[arg-type]
            margin=self.config.get("margin"),
            tikz=data,
        )

        return tex

    def to_tikz(self) -> str:
        r"""Generate TikZ drawing commands for the network visualization.
        
        Converts the processed graph data (nodes, edges, layout) into TikZ-specific
        drawing commands. Handles node positioning, styling, edge routing, and
        label placement according to the configured visualization parameters.
        
        Returns:
            str: TikZ drawing commands ready for inclusion in LaTeX document
            
        Generated Elements:
            - **Node commands** - `\Vertex` with labels, positions, colors, and sizes
            - **Edge commands** - `\Edge` with styling and optional curvature

        Note:
            The output assumes the tikz-network package is loaded in the template.
            Coordinates are assumed to be normalized to [0, 1] range and scaled
            according to the specified document dimensions.
        """
        tikz = ""
        # generate node strings
        if not self.data["nodes"].empty:
            node_strings: pd.Series = "\\Vertex["
            # show labels if specified
            if self.show_labels:
                node_strings += (
                    "label=$" + self.data["nodes"].index.astype(str).map(self._replace_with_LaTeX_math_symbol) + "$,"
                )
                node_strings += (
                    "fontsize=\\fontsize{" + str(int(0.6 * self.data["nodes"]["size"].mean())) + "}{10}\selectfont,"
                )
            # Convert hex colors to rgb if necessary
            if self.data["nodes"]["color"].str.startswith("#").all():
                self.data["nodes"]["color"] = self.data["nodes"]["color"].map(hex_to_rgb)
                node_strings += "RGB,color={" + self.data["nodes"]["color"].astype(str).str.strip("()") + "},"
            else:
                node_strings += "color=" + self.data["nodes"]["color"] + ","
            # add other options
            node_strings += "size=" + (self.data["nodes"]["size"] * 0.075).astype(str) + ","
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
        if not self.data["edges"].empty:
            edge_strings: pd.Series = "\\Edge["
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
                "(" + self.data["edges"].index.get_level_values("source").astype(str) + ")(" + self.data["edges"].index.get_level_values("target").astype(str) + ")\n"
            )
            tikz += edge_strings.str.cat()

        return tikz

    def _replace_with_LaTeX_math_symbol(self, node_label: str) -> str:
        """Replace certain symbols with LaTeX math symbols."""
        replacements = {
            "->": r"\to ",
            "<-": r"\gets ",
            "<->": r"\leftrightarrow ",
            "=>": r"\Rightarrow ",
            "<=": r"\Leftarrow ",
            "<=>": r"\Leftrightarrow ",
            "!=": r"\neq ",
        }
        if self.config["separator"].strip() in replacements:
            node_label = node_label.replace(
                self.config["separator"],
                replacements[self.config["separator"].strip()],
            )
        return node_label
