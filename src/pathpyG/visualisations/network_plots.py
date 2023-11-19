"""Network plot classes."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_plots.py -- Network plots
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Sun 2023-11-19 15:12 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import logging

from collections import defaultdict
from typing import TYPE_CHECKING, Any
from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.xutils import rgb_to_hex, Colormap
from pathpyG.visualisations.layout import layout as network_layout

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.Graph import Graph
    from pathpyG.core.TemporalGraph import TemporalGraph


# create logger
logger = logging.getLogger("root")


def network_plot(network: Graph, **kwargs: Any) -> NetworkPlot:
    """Plot a static network.

    This function generates a static plot of the network, thereby different
    output can be chosen, including

    - interactive html with d3js
    - tex file with tikz code
    - pdf from the tex source
    - png based on matplotlib

    The appearance of the plot can be modified by keyword arguments which will
    be explained in more detail below.

    Parameters
    ----------
    network : Graph

        A :py:class`Graph` object

    kwargs : keyword arguments, optional (default = no attributes)

        Attributes used to modify the appearance of the plot.
        For details see below.

    Keyword arguments used for the plotting:

    filename : str optional (default = None)

        Filename to save. The file ending specifies the output. i.e. is the
        file ending with '.tex' a tex file will be created; if the file ends
        with '.pdf' a pdf is created; if the file ends with '.html', a html
        file is generated generated. If no ending is defined a temporary html
        file is compiled and shown.


    **Nodes:**

    - ``node_size`` : diameter of the node

    - ``node_color`` : The fill color of the node. Possible values are:

            - A single color string referred to by name, RGB or RGBA code, for
              instance 'red' or '#a98d19' or (12,34,102).

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.


    - ``node_cmap`` : Colormap for node colors. If node colors are given as int
      or float values the color will be assigned based on a colormap. Per
      default the color map goes from red to green. Matplotlib colormaps or
      seaborn color palettes can be used to style the node colors.

    - ``node_opacity`` : fill opacity of the node. The default is 1. The range
      of the number lies between 0 and 1. Where 0 represents a fully
      transparent fill and 1 a solid fill.


    **Edges**

    - ``edge_size`` : width of the edge

    - ``edge_color`` : The line color of the edge. Possible values are:

            - A single color string referred to by name, RGB or RGBA code, for
              instance 'red' or '#a98d19' or (12,34,102).

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.


    - ``edge_cmap`` : Colormap for edge colors. If node colors are given as int
      or float values the color will be assigned based on a colormap. Per
      default the color map goes from red to green. Matplotlib colormaps or
      seaborn color palettes can be used to style the edge colors.

    - ``edge_opacity`` : line opacity of the edge. The default is 1. The range
      of the number lies between 0 and 1. Where 0 represents a fully
      transparent fill and 1 a solid fill.

    **General**

    - ``keep_aspect_ratio``

    - ``margin``

    - ``layout``

    References
    ----------
    .. [tn] https://github.com/hackl/tikz-network

    """
    return NetworkPlot(network, **kwargs)


class NetworkPlot(PathPyPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, network: Graph, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.network = network
        self.config = kwargs
        self.generate()

    def generate(self) -> None:
        """Generate the plot."""
        print("Generate network plot.")
        self._compute_edge_data()
        self._compute_node_data()
        self._compute_layout()
        self._cleanup_data()

    def _compute_node_data(self) -> None:
        """Generate the data structure for the nodes."""
        # initialize values
        nodes: dict = {}
        attributes: set = {"color", "size", "opacity"}
        attr: defaultdict = defaultdict(dict)

        # get attributes categories from pathpyg
        categories = {
            a.replace("node_", "") for a in self.network.node_attrs()
        }.intersection(attributes)

        # add node data to data dict
        self._get_node_data(nodes, attributes, attr, categories)

        # convert needed attributes to useful values
        attr["color"] = self._convert_color(attr["color"], mode="node")
        attr["opacity"] = self._convert_opacity(attr["opacity"], mode="node")
        attr["size"] = self._convert_size(attr["size"], mode="node")

        # update data dict with converted attributes
        for attribute in attr:
            for key, value in attr[attribute].items():
                nodes[key][attribute] = value

        # save node data
        self.data["nodes"] = nodes

    def _get_node_data(
        self,
        nodes: dict,
        attributes: set,
        attr: defaultdict,
        categories: set,
    ) -> None:
        """Extract node data from network."""
        for uid in self.network.nodes:
            nodes[uid] = {"uid": uid}

            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"node_{attribute}", uid].item()
                    if attribute in categories
                    else None
                )

    def _compute_edge_data(self) -> None:
        """Generate the data structure for the edges."""
        # initialize values
        edges: dict = {}
        attributes: set = {"weight", "color", "size", "opacity"}
        attr: defaultdict = defaultdict(dict)

        # get attributes categories from pathpyg
        categories: set = {
            a.replace("edge_", "") for a in self.network.edge_attrs()
        }.intersection(attributes)

        # add edge data to data dict
        self._get_edge_data(edges, attributes, attr, categories)

        # convert needed attributes to useful values
        attr["weight"] = self._convert_weight(attr["weight"], mode="edge")
        attr["color"] = self._convert_color(attr["color"], mode="edge")
        attr["opacity"] = self._convert_opacity(attr["opacity"], mode="edge")
        attr["size"] = self._convert_size(attr["size"], mode="edge")

        # update data dict with converted attributes
        for attribute in attr:
            for key, value in attr[attribute].items():
                edges[key][attribute] = value

        # save edge data
        self.data["edges"] = edges

    def _get_edge_data(
        self,
        edges: dict,
        attributes: set,
        attr: defaultdict,
        categories: set,
    ) -> None:
        """Extract edge data from network."""
        for u, v in self.network.edges:
            uid = f"{u}-{v}"
            edges[uid] = {
                "uid": uid,
                "source": str(u),
                "target": str(v),
            }
            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"edge_{attribute}", u, v].item()
                    if attribute in categories
                    else None
                )

    def _convert_weight(self, weight: dict, mode: str = "node") -> dict:
        """Convert weight to float."""
        # get style from the config
        style = self.config.get(f"{mode}_weight")

        # check if new attribute is a single object
        if isinstance(style, (int, float)):
            weight = {k: style for k in weight}

        # check if new attribute is a dict
        elif isinstance(style, dict):
            weight.update(**{k: v for k, v in style.items() if k in weight})

        # return all weights which are not None
        return {k: v if v is not None else 1 for k, v in weight.items()}

    def _convert_color(self, color: dict, mode: str = "node") -> dict:
        """Convert colors to hex if rgb."""
        # get style from the config
        style = self.config.get(f"{mode}_color")

        # check if new attribute is a single object
        if isinstance(style, (str, int, float, tuple)):
            color = {k: style for k in color}

        # check if new attribute is a dict
        elif isinstance(style, dict):
            color.update(**{k: v for k, v in style.items() if k in color})

        # check if new attribute is a list
        elif isinstance(style, list):
            for i, k in enumerate(color):
                try:
                    color[k] = style[i]
                except IndexError:
                    pass

        # check if numerical values are given
        values = [v for v in color.values() if isinstance(v, (int, float))]

        if values:
            # load colormap to map numerical values to color
            cmap = self.config.get(f"{mode}_cmap", Colormap())
            cdict = {
                values[i]: tuple(c[:3]) for i, c in enumerate(cmap(values, bytes=True))
            }

        # convert colors to hex if not already string
        for key, value in color.items():
            if isinstance(value, tuple):
                color[key] = rgb_to_hex(value)
            elif isinstance(value, (int, float)):
                color[key] = rgb_to_hex(cdict[value])

        # return all colors wich are not None
        return {k: v for k, v in color.items() if v is not None}

    def _convert_opacity(self, opacity: dict, mode: str = "node") -> dict:
        """Convert opacity to float."""
        # get style from the config
        style = self.config.get(f"{mode}_opacity")

        # check if new attribute is a single object
        if isinstance(style, (int, float)):
            opacity = {k: style for k in opacity}

        # check if new attribute is a dict
        elif isinstance(style, dict):
            opacity.update(**{k: v for k, v in style.items() if k in opacity})

        # return all colors wich are not None
        return {k: v for k, v in opacity.items() if v is not None}

    def _convert_size(self, size: dict, mode: str = "node") -> dict:
        """Convert size to float."""
        # get style from the config
        style = self.config.get(f"{mode}_size")

        # check if new attribute is a single object
        if isinstance(style, (int, float)):
            size = {k: style for k in size}

        # check if new attribute is a dict
        elif isinstance(style, dict):
            size.update(**{k: v for k, v in style.items() if k in size})

        # return all colors wich are not None
        return {k: v for k, v in size.items() if v is not None}

    def _compute_layout(self) -> None:
        """Create layout."""
        # get layout form the config
        layout = self.config.get("layout", None)

        # if no layout is considered stop this process
        if layout is None:
            return

        # get layout dict for each node
        if isinstance(layout, str):
            layout = network_layout(self.network, layout=layout)
        elif not isinstance(layout, dict):
            logger.error("The provided layout is not valid!")
            raise AttributeError

        # update x,y position of the nodes
        for uid, (_x, _y) in layout.items():
            self.data["nodes"][uid]["x"] = _x
            self.data["nodes"][uid]["y"] = _y

    def _cleanup_data(self) -> None:
        """Clean up final data structure."""
        self.data["nodes"] = list(self.data["nodes"].values())
        self.data["edges"] = list(self.data["edges"].values())


def temporal_plot(network: TemporalGraph, **kwargs: Any) -> NetworkPlot:
    """Plot a temporal network.

    **Temporal properties:**

    - ``start`` : start time of the simulation

    - ``end`` : end time of the simulation

    - ``delta`` : time needed for progressing one time step

    - ``intervals`` : number of numeric intervals

    """
    return TemporalNetworkPlot(network, **kwargs)


class TemporalNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "temporal"

    def __init__(self, network: TemporalGraph, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__(network, **kwargs)

    def _get_edge_data(
        self, edges: dict, attributes: set, attr: defaultdict, categories: set
    ) -> None:
        """Extract edge data from temporal network."""
        # TODO: Fix typing issue with temporal graphs
        for u, v, t in self.network.temporal_edges:  # type: ignore
            uid = f"{u}-{v}-{t}"
            edges[uid] = {
                "uid": uid,
                "source": str(u),
                "target": str(v),
                "start": int(t),
                "end": int(t) + 1,
            }
            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"edge_{attribute}", u, v].item()
                    if attribute in categories
                    else None
                )

    def _get_node_data(
        self, nodes: dict, attributes: set, attr: defaultdict, categories: set
    ) -> None:
        """Extract node data from temporal network."""
        for uid in self.network.nodes:
            nodes[uid] = {
                "uid": uid,
                "start": int(0),
                "end": int(20),
            }

            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"node_{attribute}", uid].item()
                    if attribute in categories
                    else None
                )


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
