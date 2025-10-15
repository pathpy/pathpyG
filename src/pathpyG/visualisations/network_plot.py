"""Network plot classes."""

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

import os
import logging
from typing import TYPE_CHECKING, Any, Sized

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb

from pathpyG.visualisations.layout import layout as network_layout
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.utils import rgb_to_hex, image_to_base64

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.graph import Graph


# create logger
logger = logging.getLogger("root")


class NetworkPlot(PathPyPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, network: Graph, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.network = network
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
        """Generate the plot."""
        self._compute_edge_data()
        self._compute_node_data()
        self._compute_layout()
        self._post_process_node_data()
        self._compute_config()

    def _compute_node_data(self) -> None:
        """Generate the data structure for the nodes."""
        # initialize values
        nodes: pd.DataFrame = pd.DataFrame(index=self.network.nodes)
        # if higher-order network, convert node tuples to string representation
        if self.network.order > 1:
            nodes.index = nodes.index.map(lambda x: self.config["higher_order"]["separator"].join(map(str, x)))
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
                nodes = self.assign_argument(attribute, self.node_args[attribute], nodes)

        # save node data
        self.data["nodes"] = nodes

    def _post_process_node_data(self) -> None:
        """Post-process specific node attributes after constructing the DataFrame."""
        # convert colors to uniform hex values
        self.data["nodes"]["color"] = self._convert_to_rgb_tuple(self.data["nodes"]["color"])
        self.data["nodes"]["color"] = self.data["nodes"]["color"].map(self._convert_color)

        # load any local images to base64 strings
        if self.data["nodes"]["image"].notna().any():
            self.data["nodes"]["image"] = self.data["nodes"]["image"].map(self._load_image)

    def _compute_edge_data(self) -> None:
        """Generate the data structure for the edges."""
        # initialize values
        edges: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.network.edges, names=["source", "target"]))
        # if higher-order network, convert node tuples to string representation
        if self.network.order > 1:
            edges.index = edges.index.map(lambda x: (self.config["higher_order"]["separator"].join(map(str, x[0])), self.config["higher_order"]["separator"].join(map(str, x[1]))))
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
                edges = self.assign_argument(attribute, self.edge_args[attribute], edges)
            elif attribute == "size" and "weight" in self.edge_args:
                edges = self.assign_argument("size", self.edge_args["weight"], edges)

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

    def assign_argument(self, attr_key: str, attr_value: Any, df: pd.DataFrame) -> pd.DataFrame:
        """Assign argument to node or edge attribute.

        Assigns the given value to the specified attribute key in the provided DataFrame.
        `attr_value` can be a constant value, a list of values (of length equal to the number of nodes/edges),
        or a dictionary mapping node/edge identifiers to values.

        Args:
            attr_key (str): Attribute key.
            attr_value (Any): Attribute value.
            df (pd.DataFrame): DataFrame to assign the attribute to (nodes or edges).
        """
        if isinstance(attr_value, dict):
            # if dict does not contain values for all edges, only update those that are given
            new_attrs = df.index.map(attr_value)
            df.loc[~new_attrs.isna(), attr_key] = new_attrs[~new_attrs.isna()]
        elif isinstance(attr_value, Sized) and not isinstance(attr_value, str):
            if len(attr_value) != len(df):
                logger.error(f"The provided list for {attr_key} has length {len(attr_value)}, but there are {len(df)} nodes/edges!")
                raise AttributeError
            df[attr_key] = attr_value
        else:
            df[attr_key] = attr_value
        return df

    def _convert_to_rgb_tuple(self, colors: pd.Series) -> dict:
        """Convert colors to rgb colormap if given as numerical values."""
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
        """Convert color rgb tuple to hex."""
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
        elif color is None or pd.isna(color):
            return pd.NA  # will be filled with self._fill_node_values()
        else:
            logger.error(f"The provided color {color} is not valid!")
            raise AttributeError
        
    def _load_image(self, image_path: str) -> str:
        """Check if image path is a URL or local file and load local files to base64 strings."""
        if image_path.startswith("http://") or image_path.startswith("https://") or image_path.startswith("data:"):
            return image_path  # already a URL or base64 string
        else:
            # check if file exists
            if not os.path.isfile(image_path):
                logger.error(f"The provided image path {image_path} does not exist!")
                raise AttributeError
            return image_to_base64(image_path)

    def _compute_layout(self) -> None:
        """Create layout."""
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
            layout_df.index = layout_df.index.map(lambda x: self.config["higher_order"]["separator"].join(map(str, x)))
        # scale x and y to [0,1]
        layout_df["x"] = (layout_df["x"] - layout_df["x"].min()) / (layout_df["x"].max() - layout_df["x"].min())
        layout_df["y"] = (layout_df["y"] - layout_df["y"].min()) / (layout_df["y"].max() - layout_df["y"].min())
        # join layout with node data
        self.data["nodes"] = self.data["nodes"].join(layout_df, how="left")

    def _compute_config(self) -> None:
        """Add additional configs."""
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
