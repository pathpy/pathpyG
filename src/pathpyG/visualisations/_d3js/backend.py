from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
import webbrowser
from copy import deepcopy
from string import Template

from pathpyG.utils.config import config
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot
from pathpyG.visualisations.utils import rgb_to_hex, unit_str_to_float

# create logger
logger = logging.getLogger("root")

SUPPORTED_KINDS = {
    NetworkPlot: "static",
    TemporalNetworkPlot: "temporal",
}


class D3jsBackend(PlotBackend):
    """D3js plotting backend."""

    def __init__(self, plot, show_labels: bool):
        super().__init__(plot, show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(
                f"Plot of type {type(plot)} not supported by D3js backend."
            )
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save the plot to the hard drive."""
        with open(filename, "w+") as new:
            new.write(self.to_html())

    def show(self) -> None:
        """Show the plot on the device."""
        if config["environment"]["interactive"]:
            from IPython.display import display_html, HTML  # noqa I001

            display_html(HTML(self.to_html()))
        else:
            # create temporal file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # save html
                self.save(filename=temp_file.name)
                # open the file
                webbrowser.open(r"file:///" + temp_file.name)

    def _prepare_data(self) -> dict:
        """Prepare the data for json conversion."""
        node_data = self.data["nodes"].copy()
        node_data["uid"] = self.data["nodes"].index
        node_data = node_data.rename(columns={"x": "xpos", "y": "ypos"})
        edge_data = self.data["edges"].copy()
        edge_data["uid"] = self.data["edges"].index.map(lambda x: f"{x[0]}-{x[1]}")
        data_dict = {
            "nodes": node_data.to_dict(orient="records"),
            "edges": edge_data.to_dict(orient="records"),
        }
        return data_dict

    def _prepare_config(self) -> dict:
        """Prepare the config for json conversion."""
        config = deepcopy(self.config)
        config["node"]["color"] = rgb_to_hex(self.config["node"]["color"])
        config["edge"]["color"] = rgb_to_hex(self.config["edge"]["color"])
        config["width"] = unit_str_to_float(self.config["width"], "px")
        config["height"] = unit_str_to_float(self.config["height"], "px")
        config["show_labels"] = self.show_labels
        return config

    def to_json(self) -> tuple[str,str]:
        """Convert data and config to json."""
        data_dict = self._prepare_data()
        config_dict = self._prepare_config()
        return json.dumps(data_dict), json.dumps(config_dict)

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
            d3js = os.path.join(template_dir, "d3.v7.min.js")
        else:
            d3js = "https://d3js.org/d3.v7.min.js"

        js_template = self.get_template(template_dir)

        with open(os.path.join(template_dir, "setup.js")) as template:
            setup_template = template.read()

        with open(os.path.join(template_dir, "styles.css")) as template:
            css_template = template.read()

        # update config
        self.config["selector"] = dom_id
        data_json, config_json = self.to_json()

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
        html += f"const data = {data_json}\n"
        html += f"const config = {config_json}\n"

        # add log print
        html += f"console.log('{self._kind} Network Template');\n"

        # add JavaScript
        html += js_template

        # end d3 environment
        html += "\n}); //END\n"

        # end JavaScript
        html += "\n</script>"

        return html

    def get_template(self, template_dir: str) -> str:
        """Get the JavaScript template for the specific plot type."""
        js_template = ""
        with open(os.path.join(template_dir, "network.js")) as template:
            js_template += template.read()

        with open(os.path.join(template_dir, f"{self._kind}.js")) as template:
            js_template += template.read()

        return js_template