"""D3.js backend for interactive web-based network visualization.

Template-driven HTML generation using D3.js library for rich interactive
visualizations. Supports both static and temporal networks with embedded
JavaScript, and CSS styling.

!!! abstract "Features":
    - Interactive HTML output with drag-and-drop node movement
    - Template-based architecture for extensibility
    - Both static and temporal network support
    - Jupyter notebook integration with inline display
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib
import uuid
import webbrowser
from copy import deepcopy
from string import Template

from pathpyG.utils.config import config
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot
from pathpyG.visualisations.unfolded_network_plot import TimeUnfoldedNetworkPlot
from pathpyG.visualisations.utils import rgb_to_hex, unit_str_to_float

# create logger
logger = logging.getLogger("root")

SUPPORTED_KINDS: dict[type, str] = {
    NetworkPlot: "static",
    TemporalNetworkPlot: "temporal",
    TimeUnfoldedNetworkPlot: "unfolded",
}
_CDN_URL = "https://d3js.org/d3.v7.min.js"


class D3jsBackend(PlotBackend):
    """D3.js backend for interactive web visualization with template system.

    Generates self-contained HTML files with embedded D3.js visualizations
    using modular template architecture. Supports both static and temporal
    networks with rich interactivity and web-standard compatibility.

    Features:
        - Template-driven HTML generation (CSS + JavaScript + data)
        - Multiple output modes: standalone HTML, Jupyter display, browser
        - JSON data serialization with proper type conversion

    Example:
        ```python
        import pathpyG as pp

        # Simple network visualization
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        g = pp.Graph.from_edge_list(edges)
        pp.plot(g)  # Uses d3.js backend by default
        ```
        <iframe src="../../plot/simple_network.html" width="650" height="520"></iframe>

    !!! info "Template Architecture"
        Uses modular templates for extensibility:

        - `styles.css`: Visual styling and responsive design
        - `setup.js`: Environment detection and D3.js loading
        - `network.js`: Core network visualization logic
        - `static.js` / `temporal.js`: Plot-type specific functionality

    !!! note "Web Standards"
        Generates standards-compliant HTML5 with SVG graphics,
        compatible with all modern browsers without plugins.
    """

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initialize D3.js backend with plot validation and configuration.

        Args:
            plot: PathPyPlot instance (NetworkPlot or TemporalNetworkPlot)
            show_labels: Whether to display node labels in visualization

        Raises:
            ValueError: If plot type not supported by D3.js backend

        !!! tip "Supported Plot Types"
            - **NetworkPlot**: Static network visualization
            - **TemporalNetworkPlot**: Animated temporal network evolution
        """
        super().__init__(plot, show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by D3js backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save interactive visualization as standalone HTML file.

        Creates self-contained HTML file with embedded D3.js visualization,
        complete with styling, JavaScript, and data. File can be opened
        in any web browser or served from web servers.

        Args:
            filename: Output HTML file path

        !!! tip "Deployment Ready"
            Generated HTML files are standalone and can be:

            - Opened directly in browsers
            - Served from web servers
            - Embedded in websites or documentation
            - Shared without additional dependencies
        """
        # Default to the CDN version of d3js since browsers may block local scripts
        self.config["d3js_local"] = self.config.get("d3js_local", False)
        with open(filename, "w+") as new:
            new.write(self.to_html())

    def show(self) -> None:
        """Display visualization in appropriate environment.

        Automatically detects environment and displays visualization:
        - Jupyter notebooks: Inline HTML display with IPython widgets
        - Scripts/terminals: Opens temporary HTML file in system browser

        !!! info "Environment Detection"
            Uses pathpyG config to detect interactive environment
            and choose appropriate display method automatically.
        """
        # Default to CDN version if reachable
        # Check if CDN is reachable
        try:
            urllib.request.urlopen(_CDN_URL, timeout=2)
            self.config["d3js_local"] = self.config.get("d3js_local", False)
        except (urllib.error.URLError, urllib.error.HTTPError):
            self.config["d3js_local"] = self.config.get("d3js_local", True)
        
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
        """Transform network data for JSON serialization and D3.js consumption.

        Converts pandas DataFrames to D3.js-compatible format with proper
        node/edge structure. Handles coordinate renaming and unique ID generation
        for JavaScript processing.

        Returns:
            dict: Structured data with 'nodes' and 'edges' arrays

        !!! note "Data Structure"
            **Nodes**: Include uid, coordinates (xpos/ypos), and all attributes

            **Edges**: Include uid, source/target references, and styling
        """
        node_data = self.data["nodes"].copy()
        node_data["uid"] = self.data["nodes"].index.map(lambda x: f"({x[0]},{x[1]})" if isinstance(x, tuple) else str(x))
        node_data = node_data.rename(columns={"x": "xpos", "y": "ypos"})
        if self._kind == "unfolded":
            node_data["ypos"] = 1 - node_data["ypos"]  # Invert y-axis for unfolded layout
        edge_data = self.data["edges"].copy()
        edge_data["uid"] = self.data["edges"].index.map(lambda x: f"{x[0]}-{x[1]}")
        if len(edge_data) > 0:
            edge_data["source"] = edge_data.index.to_frame()["source"].map(lambda x: f"({x[0]},{x[1]})" if isinstance(x, tuple) else str(x))
            edge_data["target"] = edge_data.index.to_frame()["target"].map(lambda x: f"({x[0]},{x[1]})" if isinstance(x, tuple) else str(x))
        data_dict = {
            "nodes": node_data.to_dict(orient="records"),
            "edges": edge_data.to_dict(orient="records"),
        }
        return data_dict

    def _prepare_config(self) -> dict:
        """Transform configuration for JavaScript compatibility.

        Converts pathpyG configuration to web-compatible format with proper
        color conversion, unit normalization, and JavaScript-friendly types.

        Returns:
            dict: Web-compatible configuration object

        !!! info "Configuration Processing"
            - **Colors**: Convert to hex format for CSS compatibility
            - **Units**: Convert to pixels for SVG rendering
            - **Types**: Ensure JSON-serializable data types
        """
        config = deepcopy(self.config)
        config["node"]["color"] = rgb_to_hex(self.config["node"]["color"])
        config["edge"]["color"] = rgb_to_hex(self.config["edge"]["color"])
        config["width"] = unit_str_to_float(self.config["width"], "px")
        config["height"] = unit_str_to_float(self.config["height"], "px")
        config["show_labels"] = self.show_labels
        return config

    def to_json(self) -> tuple[str, str]:
        """Serialize network data and configuration to JSON strings.

        Processes both data and configuration through preparation methods
        and converts to JSON format suitable for JavaScript consumption.

        Returns:
            tuple: (data_json, config_json) string pair for template injection

        !!! tip "Template Integration"
            JSON strings are injected directly into JavaScript templates
            as `const data = {...}` and `const config = {...}` declarations.
        """
        data_dict = self._prepare_data()
        config_dict = self._prepare_config()
        return json.dumps(data_dict), json.dumps(config_dict)

    def to_html(self) -> str:
        """Generate complete standalone HTML visualization.

        Assembles full HTML document using template system with embedded CSS,
        JavaScript, and data. Creates unique DOM IDs to prevent conflicts
        when multiple visualizations exist on same page.

        Returns:
            str: Complete HTML document with embedded visualization

        !!! info "HTML Structure"
            1. **CSS Styles**: Embedded styling
            2. **DOM Container**: Unique div element for visualization
            3. **D3.js Library**: CDN or local library loading
            4. **Setup Code**: Environment detection and module loading
            5. **Data/Config**: JSON-serialized network and configuration
            6. **Visualization**: Plot-specific JavaScript execution

        !!! note "Library Loading"
            Supports both CDN and local (default) D3.js library embedding
            based on `d3js_local` configuration parameter.
        """
        # generate unique dom uids
        dom_id = "#x" + uuid.uuid4().hex

        # get path to the pathpy templates
        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            os.path.normpath("_d3js/templates"),
        )

        # get d3js library path
        if self.config.get("d3js_local", False):
            d3js = os.path.join(template_dir, "d3.v7.min.js")
        else:
            d3js = _CDN_URL

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
        """Load and combine JavaScript templates for visualization.

        Assembles modular JavaScript code by combining core network
        functionality with plot-type specific features. Enables clean
        separation of concerns and extensible template architecture.

        Args:
            template_dir: Directory containing JavaScript template files

        Returns:
            str: Combined JavaScript code for visualization

        !!! info "Template Composition"
            **Core Template** (`network.js`): Base network visualization logic

            **Plot Templates**: Type-specific functionality:

            - `static.js`: Force simulation and interaction for static networks
            - `temporal.js`: Timeline controls and animation for temporal networks

        !!! tip "Extensibility"
            New plot types can be added by creating additional
            JavaScript templates following the established patterns.
        """
        js_template = ""
        with open(os.path.join(template_dir, "network.js")) as template:
            js_template += template.read()

        with open(
            os.path.join(template_dir, "static.js" if self._kind == "unfolded" else f"{self._kind}.js")
        ) as template:
            js_template += template.read()

        return js_template
