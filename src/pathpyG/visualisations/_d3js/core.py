#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : core.py -- Plots with d3js
# Author    : Jürgen Hackl <hackl@ifi.uzh.ch>
# Time-stamp: <Wed 2023-12-06 17:07 juergen>
#
# Copyright (c) 2016-2021 Pathpy Developers
# =============================================================================
from __future__ import annotations

import os
import json
import uuid
import logging
import tempfile
import webbrowser

from typing import Any
from string import Template

from pathpyG.utils.config import config
from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


class D3jsPlot(PathPyPlot):
    """Base class for plotting d3js objects."""

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    def save(self, filename: str, **kwargs: Any) -> None:
        """Save the plot to the hard drive."""
        with open(filename, "w+") as new:
            new.write(self.to_html())

    def show(self, **kwargs: Any) -> None:
        """Show the plot on the device."""
        if config["environment"]["interactive"]:
            from IPython.display import display_html, HTML

            display_html(HTML(self.to_html()))
        else:
            # create temporal file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # save html
                self.save(temp_file.name)
                # open the file
                webbrowser.open(r"file:///" + temp_file.name)

    def to_json(self) -> str:
        """Convert data to json."""
        raise NotImplementedError

    def to_html(self) -> str:
        """Convert data to html."""
        # generate unique dom uids
        dom_id = "#x" + uuid.uuid4().hex

        # get path to the pathpy templates
        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            os.path.normpath("_d3js/templates"),
        )

        # get d3js version
        local = self.config.get("d3js_local", False)
        if local:
            d3js = os.path.join(template_dir, "d3.v5.min.js")
        else:
            d3js = "https://d3js.org/d3.v5.min.js"

        # get template files
        with open(os.path.join(template_dir, f"{self._kind}.js")) as template:
            js_template = template.read()

        with open(os.path.join(template_dir, "setup.js")) as template:
            setup_template = template.read()

        with open(os.path.join(template_dir, "styles.css")) as template:
            css_template = template.read()

        # load custom template
        _template = self.config.get("template", None)
        if _template and os.path.isfile(_template):
            with open(_template) as template:
                js_template = template.read()

        # load custom css template
        _template = self.config.get("css", None)
        if _template and os.path.isfile(_template):
            with open(_template) as template:
                css_template += template.read()

        # update config
        self.config["selector"] = dom_id
        data = self.to_json()

        # generate html file
        html = "<style>\n" + css_template + "\n</style>\n"

        # div environment for the plot object
        html += f'\n<div id = "{dom_id[1:]}"> </div>\n'

        # add d3js library
        html += f'<script charset="utf-8" src="{d3js}"></script>\n'

        # start JavaScript
        html += '<script charset="utf-8">\n'

        # add setup code to run d3js in multiple environments
        html += Template(setup_template).substitute(d3js=d3js)

        # start d3 environment
        html += "require(['d3'], function(d3){ //START\n"

        # add data and config
        html += f"const data = {data}\n"
        html += f"const config = {json.dumps(self.config)}\n"

        # add JavaScript
        html += js_template

        # end d3 environment
        html += "\n}); //END\n"

        # end JavaScript
        html += "\n</script>"

        return html


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
