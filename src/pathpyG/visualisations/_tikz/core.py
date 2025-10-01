#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : core.py -- Plots with tikz
# Author    : Jürgen Hackl <hackl@ifi.uzh.ch>
# Time-stamp: <Wed 2023-10-25 09:52 juergen>
#
# Copyright (c) 2016-2021 Pathpy Developers
# =============================================================================
from __future__ import annotations

import os
import time
import shutil
import logging
import tempfile
import subprocess
import webbrowser

from typing import Any
from string import Template

from pathpyG.utils.config import config
from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


class TikzPlot(PathPyPlot):
    """Base class for plotting tikz objects."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize plot class."""
        super().__init__()
        if kwargs:
            self.config = kwargs

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    def save(self, filename: str, **kwargs: Any) -> None:
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

        else:
            raise NotImplementedError

    def show(self, **kwargs: Any) -> None:
        """Show the plot on the device."""
        # compile temporary pdf
        temp_file, temp_dir = self.compile_pdf()

        if config["environment"]["interactive"]:
            from IPython.display import IFrame, display

            # open the file in the notebook
            display(IFrame(temp_file, width=600, height=300))
        else:
            # open the file in the webbrowser
            webbrowser.open(r"file:///" + temp_file)

        # Wait for .1 second before temp file is deleted
        time.sleep(0.1)

        # remove the temporal directory
        shutil.rmtree(temp_dir)

    def compile_pdf(self) -> tuple:
        """Compile pdf from tex."""
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
            raise AttributeError
        finally:
            # change back to the current directory
            os.chdir(current_dir)

        # return the name of the folder and temp pdf file
        return (os.path.join(temp_dir, basename + ".pdf"), temp_dir)

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
            classoptions=self.config.get("latex_class_options", ""),
            width=self.config.get("width", "6cm"),
            height=self.config.get("height", "6cm"),
            tikz=data,
        )

        return tex

    def to_tikz(self) -> str:
        """Convert data to tikz."""
        raise NotImplementedError


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
