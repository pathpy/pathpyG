"""Static network visualization classes.

Provides comprehensive plotting functionality for static (non-temporal) networks.
Handles data preparation, attribute assignment, layout computation, and backend
integration for Graph objects.

!!! abstract "Key Features"
    - Automatic attribute extraction from network data
    - Flexible node/edge styling (colors, sizes, images)
    - Layout algorithm integration
    - Multi-backend compatibility

!!! note "Attribute Sources"
    Attributes are resolved in order (highest priority to the leftmost): user arguments → network attributes → config defaults
"""

# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_plots.py -- Network plots
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Sat 2024-02-17 15:49 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Sized

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.colors import to_rgb

from pathpyG.visualisations.layout import layout as network_layout
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.utils import image_to_base64, rgb_to_hex

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.graph import Graph


# create logger
logger = logging.getLogger("root")


class NetworkPlot(PathPyPlot):
    """Static network visualization with comprehensive styling options.

    Prepares Graph objects for visualization by extracting node/edge data,
    computing layouts, and processing visual attributes. Supports both 
    simple and higher-order networks with flexible attribute assignment.

    Attributes:
        network: Graph instance being visualized
        node_args: Node-specific styling arguments
        edge_args: Edge-specific styling arguments
        attributes: Standard visual attributes (color, size, opacity, image)

    !!! tip "Attribute Assignment"
        Use `node_color`, `edge_size` etc. for convenient styling.
        Attributes support constants, lists, or node/edge mappings.
    """

    _kind = "network"

    def __init__(self, network: Graph, **kwargs: Any) -> None:
        """Initialize network plot with graph and styling options.

        Processes node/edge arguments, updates configuration, and generates
        plot data structures. Arguments prefixed with 'node_' or 'edge_'
        are automatically assigned to respective components.

        Args:
            network: Graph instance to visualize
            **kwargs: Styling options (node_color, edge_size, layout, etc.)
        """
        super().__init__()
        self.network = network if network.device == torch.device("cpu") else deepcopy(network).to(torch.device("cpu"))
        self.node_args = {}
        self.edge_args = {}
        self.attributes = ["color", "size", "opacity", "image"]
        # extract node and edge specific arguments from kwargs
        for key in kwargs.keys():
            if key.startswith("node_"):
                self.node_args[key[5:]] = kwargs.get(key)
            elif key.startswith("edge_"):
                self.edge_args[key[5:]] = kwargs.get(key)
        # remove node_ and edge_ arguments from kwargs and update config with remaining kwargs
        for node_arg in self.node_args.keys():
            kwargs.pop(f"node_{node_arg}")
        for edge_arg in self.edge_args.keys():
            kwargs.pop(f"edge_{edge_arg}")
        if "node" in kwargs:
            self.config["node"].update(kwargs["node"])
            kwargs.pop("node")
        if "edge" in kwargs:
            self.config["edge"].update(kwargs["edge"])
            kwargs.pop("edge")
        self.config.update(kwargs)
        # generate plot data
        self.generate()

    def generate(self) -> None:
        """Generate complete plot data through processing pipeline.

        Orchestrates data preparation: edges → nodes → layout → post-processing → config.
        Creates final data structures ready for backend rendering.
        """
        self._compute_edge_data()
        self._compute_node_data()
        self._compute_layout()
        self._post_process_node_data()
        self._compute_config()

    def _compute_node_data(self) -> None:
        """Build node DataFrame with visual attributes.

        Creates indexed DataFrame for all nodes, handling higher-order networks
        by converting tuple nodes to string representation. Assigns attributes
        from config defaults, network data, and user arguments.
        """
        # initialize values
        nodes: pd.DataFrame = pd.DataFrame(index=self.network.nodes)
        # if higher-order network, convert node tuples to string representation
        if self.network.order > 1:
            nodes.index = nodes.index.map(lambda x: self.config["separator"].join(map(str, x)))
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("node").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                nodes[attribute] = [self.config.get("node").get(attribute, None)] * len(nodes)  # type: ignore[union-attr]
            else:
                nodes[attribute] = self.config.get("node").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as node attribute
            if f"node_{attribute}" in self.network.node_attrs():
                nodes[attribute] = self.network.data[f"node_{attribute}"]
            # check if attribute is given as argument
            if attribute in self.node_args:
                nodes = self._assign_argument(attribute, self.node_args[attribute], nodes)

        # save node data
        self.data["nodes"] = nodes

    def _post_process_node_data(self) -> None:
        """Finalize node attributes for backend compatibility.

        Converts colors to uniform hex format and loads local images
        to base64 strings for embedding in output formats.
        """
        # convert colors to uniform hex values
        self.data["nodes"]["color"] = self._convert_to_rgb_tuple(self.data["nodes"]["color"])
        self.data["nodes"]["color"] = self.data["nodes"]["color"].map(self._convert_color)

        # load any local images to base64 strings
        if self.data["nodes"]["image"].notna().any():
            self.data["nodes"]["image"] = self.data["nodes"]["image"].map(self._load_image)

    def _compute_edge_data(self) -> None:
        """Build edge DataFrame with visual attributes and deduplication.

        Creates MultiIndex DataFrame for edges, handles higher-order networks,
        assigns attributes, and removes duplicates for undirected graphs.
        Special handling for edge weights as size defaults.

        !!! warning "No support for networks with multiedges"
            For efficiency, duplicate edges are removed.
        """
        # initialize values
        edges: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.network.edges, names=["source", "target"]))
        # if higher-order network, convert node tuples to string representation
        if self.network.order > 1:
            edges.index = edges.index.map(lambda x: (self.config["separator"].join(map(str, x[0])), self.config["separator"].join(map(str, x[1]))))
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("edge").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                edges[attribute] = [self.config.get("edge").get(attribute, None)] * len(edges)  # type: ignore[union-attr]
            else:
                edges[attribute] = self.config.get("edge").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as edge attribute
            if f"edge_{attribute}" in self.network.edge_attrs():
                edges[attribute] = self.network.data[f"edge_{attribute}"]
            # special case for size: If no edge_size is given use edge_weight if available
            elif attribute == "size" and "edge_weight" in self.network.edge_attrs():
                edges[attribute] = self.network.data["edge_weight"]
            # check if attribute is given as argument
            if attribute in self.edge_args:
                edges = self._assign_argument(attribute, self.edge_args[attribute], edges)
            elif attribute == "size" and "weight" in self.edge_args:
                edges = self._assign_argument("size", self.edge_args["weight"], edges)

        # convert attributes to useful values
        edges["color"] = self._convert_to_rgb_tuple(edges["color"])
        edges["color"] = edges["color"].map(self._convert_color)

        # remove duplicate edges for better efficiency
        if not self.network.is_directed():
            # for undirected networks, sort source and target and drop duplicates
            edges = edges.reset_index()
            edges["sorted"] = edges.apply(lambda row: tuple(sorted((row["source"], row["target"]))), axis=1)
            edges = edges.drop_duplicates(subset=["sorted"]).drop(columns=["sorted"])
            edges = edges.set_index(["source", "target"])
        else:
            # for directed networks, remove duplicates based on index
            edges = edges[~edges.index.duplicated(keep="first")]
        
        # save edge data
        self.data["edges"] = edges

    def _assign_argument(self, attr_key: str, attr_value: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Assign user arguments to node/edge attributes flexibly.

        Handles multiple value types: constants, lists/arrays, or dictionaries
        mapping node/edge IDs to values. Special handling for RGB color tuples
        and proper length validation for sequence types.

        Args:
            attr_key: Attribute name (color, size, opacity, image)
            attr_value: Value to assign (constant, list, or dict mapping)
            df: Target DataFrame (nodes or edges)

        Returns:
            Updated DataFrame with assigned attributes

        Raises:
            AttributeError: If list length doesn't match DataFrame size
        """
        if isinstance(attr_value, dict):
            # if dict does not contain values for all edges, only update those that are given
            if attr_key == "color":
                # convert color tuples to hex strings to avoid pandas sequence assignment
                for key in attr_value.keys():
                    value = attr_value[key]
                    if isinstance(value, tuple) and len(value) == 3:
                        attr_value[key] = rgb_to_hex(value)
            new_attrs = df.index.map(attr_value)
            # Check if all values are assigned
            if (~new_attrs.isna()).sum() == df.shape[0]:
                # If all values are assigned, directly set the column to make sure that dtype is correct
                df[attr_key] = new_attrs
            else:
                # Otherwise, only update the values that are not NaN
                df.loc[~new_attrs.isna(), attr_key] = new_attrs[~new_attrs.isna()]
        elif isinstance(attr_value, Sized) and not isinstance(attr_value, str):
            # check if attr_key="color" and given values is an RGB tuple
            if attr_key == "color":
                if isinstance(attr_value, tuple) and len(attr_value) == 3:
                    df[attr_key] = [attr_value] * len(df)
                else:
                    df[attr_key] = attr_value
            elif len(attr_value) != len(df):
                logger.error(f"The provided list for {attr_key} has length {len(attr_value)}, but there are {len(df)} nodes/edges!")
                raise AttributeError
            else:
                df[attr_key] = attr_value
        else:
            df[attr_key] = attr_value
        return df

    def _convert_to_rgb_tuple(self, colors: pd.Series) -> dict:
        """Convert numeric color values to RGB tuples via colormap.

        Maps numerical values to colors using matplotlib colormap when
        colors are provided as numeric data (for value-based coloring).

        Args:
            colors: Series containing color values (numeric or already processed)

        Returns:
            Series with RGB tuple colors or original non-numeric colors
        """
        # check if colors are given as numerical values
        if pd.api.types.is_numeric_dtype(colors):
            # load colormap to map numerical values to color
            cmap_name = self.config.get("cmap")
            cmap = plt.get_cmap(cmap_name)
            # normalize values to [0,1]
            norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
            # map values to colors
            colors = colors.map(lambda x: cmap(norm(x)))
        return colors

    def _convert_color(self, color: tuple[int, int, int]) -> str:
        """Normalize colors to hex format for backend consistency.

        Converts RGB tuples, color names, or existing hex values to
        standardized hex format. Handles matplotlib color names via
        automatic RGB conversion.

        Args:
            color: Color as RGB tuple, hex string, or named color

        Returns:
            Hex color string (e.g., "#ff0000")

        Raises:
            AttributeError: If color format is invalid or unrecognized
        """
        if isinstance(color, tuple):
            return rgb_to_hex(color[:3])
        elif isinstance(color, str):
            if color.startswith("#"):
                return color
            else:
                # try to convert color name to hex
                try:
                    rgb = to_rgb(color)
                    return rgb_to_hex(rgb)
                except ValueError:
                    logger.error(f"The provided color {color} is not valid!")
                    raise AttributeError
        elif not isinstance(color, Sized) and (color is None or pd.isna(color)):
            return pd.NA  # will be filled with self._fill_node_values()
        else:
            logger.error(f"The provided color {color} is not valid!")
            raise AttributeError
        
    def _load_image(self, image_path: str) -> str:
        """Load local images to base64 or pass through URLs.

        Converts local image files to base64 data URLs for embedding
        while preserving existing URLs and data URLs unchanged.

        Args:
            image_path: Local file path, URL, or data URL

        Returns:
            Base64 data URL for local files, original string for URLs

        Raises:
            AttributeError: If local file path doesn't exist
        """
        if image_path.startswith("http://") or image_path.startswith("https://") or image_path.startswith("data:"):
            return image_path  # already a URL or base64 string
        else:
            # check if file exists
            if not os.path.isfile(image_path):
                logger.error(f"The provided image path {image_path} does not exist!")
                raise AttributeError
            return image_to_base64(image_path)

    def _compute_layout(self) -> None:
        """Compute and normalize node positions using layout algorithms.

        Applies layout algorithm from config, normalizes coordinates to [0,1]
        range, and joins position data with node attributes. Handles both
        string layout names and pre-computed position dictionaries.
        """
        # get layout from the config
        layout = self.config.get("layout")

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
        layout_df = pd.DataFrame.from_dict(layout, orient="index", columns=["x", "y"])
        if self.network.order > 1 and not isinstance(layout_df.index[0], str):
            layout_df.index = layout_df.index.map(lambda x: self.config["separator"].join(map(str, x)))
        # scale x and y to [0,1]
        layout_df["x"] = (layout_df["x"] - layout_df["x"].min()) / (layout_df["x"].max() - layout_df["x"].min())
        layout_df["y"] = (layout_df["y"] - layout_df["y"].min()) / (layout_df["y"].max() - layout_df["y"].min())
        # join layout with node data
        self.data["nodes"] = self.data["nodes"].join(layout_df, how="left")

    def _compute_config(self) -> None:
        """Set network-specific visualization configuration.

        Configures directedness, edge curvature, and simulation mode (for `d3.js` backend)
        based on network properties. Directed networks use curved edges,
        simulation mode activates when no layout is specified.
        """
        self.config["directed"] = self.network.is_directed()
        self.config["curved"] = self.network.is_directed()
        self.config["simulation"] = self.config["layout"] is None


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
