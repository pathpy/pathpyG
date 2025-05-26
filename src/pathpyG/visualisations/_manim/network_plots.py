"""
Network plots using Manim.

This module provides classes and utilites for visualizing networks using the Manim animation
engine. It includes base classes, custom rendering behaviour and configuration options for styling
and controlling the output.

Classes:
    - NetworkPlot: Base class for network visualizations.
    - StaticNetworkPlot: Static layout and display of a network.
    - TemporalNetworkPlot: Animated plot showing temporal evolution of a network
"""

# =============================================================================
# File      : network_plots.py -- Network plots with manim
# =============================================================================

import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from manim import *
from matplotlib.pyplot import get_cmap
from tqdm import tqdm

import pathpyG as pp
from pathpyG.visualisations._manim.core import ManimPlot

logger = logging.getLogger("root")


class NetworkPlot(ManimPlot):
    """Base class for static and dynamic network plots.

    This class stores the raw input data and configuration arguments,
    serving as a parent for Manim-based visualisations.
    """

    _kind = "network"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """
        Initializes a network plot.

        Args:
            data (dict):  Input network data dictionary
            **kwargs (Any): Optional keyword arguments for configuration
        """
        super().__init__()
        self.data = {}
        self.config = kwargs
        self.raw_data = data


class TemporalNetworkPlot(NetworkPlot, Scene):
    """
    Animated temporal network plot

    This class supports rendering of animations of temporal graphs over time,
    using customizable layout strategies and time-based changes in color and layout.
    """

    _kind = "temporal"

    def __init__(self, data: dict, output_dir: str | Path = None, output_file: str = None, **kwargs) -> None:
        """
        Initialize the temporal network plot.

        Args:
            data (dict): Network data
            output_dir (str | Path, optional): Directory to store output.
            output_file (str, optional): Filename for output.
            **kwargs: Additional keyword arguments to customize the plot. For more detail see below.


        # Keyword Arguments to modify the appearance of the plot
        **General**

        - `delta` (int): Duration of a timestep in milliseconds. Default is `1000`.
        - `start` (int): Start timestep of the animation. Default is `0`.
        - `end` (int): End timestep. Default is `None` (last edge time).
        - `intervals` (int): Number of animation intervals.
        - `dynamic_layout_interval` (int): Steps between layout recomputations. Default is `5`.
        - `background_color` (str): A single color string referred to by name or a RGB, for
            instance `white` or `#a98d19` or Manim Color `WHITE` or `(255,0,0)`. Default is `WHITE`.

        **Nodes:**

        - `node_size` : Sets the radius of the nodes. Can be provided as:
                single float to apply same sizing to all nodes or a dictionary mapping node identifiers to
                individual sizes, e.g., `{'1':1, '2':0.5}`

        - `node_color`: The fill color of the node. Possible values are:

            - A single color string referred to by name, HEX, RGB or RGBA code, for
            instance `red` or `#a98d19`.

            - A sequence of color strings referred to by name, HEX, RGB or RGBA code,
            which will be used for each point's color recursively. For
            instance `['green', 'yellow']` all points will be filled in green or
            yellow, alternatively.

        - `node_cmap` : Colormap for node colors. If node colors are given as int
        or float values the color will be assigned based on a colormap. Per
        default the color map goes from red to green. Matplotlib colormaps
        can be used to style the node colors.

        - `node_opacity` : fill opacity of the node. The default is 1. The range
        of the number lies between 0 and 1. Where 0 represents a fully
        transparent fill and 1 a solid fill. It is also possible to provide a dictionary
        with node identifiers for setting individual opacity, e.g., `{'a': 0.5, 'b'=1}`


        **Edges**

        - `edge_size` : Sets the width of the nodes. Can be provided as:
                single float to apply same sizing to all edges or a dictionary mapping edge identifiers to
                individual sizes, e.g., `{(('a', 'b'), 2): 6}`

        - `edge_color` : The line color of the edge. Possible values are:

            - A single color string referred to by name, HEX, RGB or RGBA code, for
            instance `red` or `#a98d19` or `(12,34,102)`, 'PINK'.

            - A sequence of color strings referred to by name, HEX, RGB or RGBA
            code, which will be used for each point's color recursively. For
            instance `['green','yellow']` all points will be filled in green or
            yellow, alternatively.

        - `edge_cmap` : Colormap for edge colors. If node colors are given as int
        or float values the color will be assigned based on a colormap. Per
        default the color map goes from red to green. Matplotlib colormaps can be used to style the edge colors.

        - `edge_opacity` : line opacity of the edge. The default is 1. The range
        of the number lies between 0 and 1. Where 0 represents a fully
        transparent fill and 1 a solid fill. It is also possible to provide a dictionary
        with edge identifiers for setting individual opacity, e.g., `{(('a', 'b'), 2): 0.2}`



        """
        from manim import config as manim_config

        if output_dir:
            manim_config.media_dir = str(output_dir)
        if output_file:
            manim_config.output_file = output_file

        # Optional config settings
        manim_config.pixel_height = 1080
        manim_config.pixel_width = 1920
        manim_config.frame_rate = 15
        manim_config.quality = "medium_quality"
        manim_config.background_color = kwargs.get("background_color", WHITE)

        self.delta = kwargs.get("delta", 1000)
        self.start = kwargs.get("start", 0)
        self.end = kwargs.get("end", None)
        self.intervals = kwargs.get("intervals", None)
        self.dynamic_layout_interval = kwargs.get("dynamic_layout_interval", 5)
        self.node_color = kwargs.get("node_color", BLUE)
        self.edge_color = kwargs.get("edge_color", GREY)
        self.node_cmap = kwargs.get("nodes_cmap", get_cmap())
        self.edge_cmap = kwargs.get("edge_cmap", get_cmap())
        self.node_opacity = kwargs.get("node_opacity", 1)
        self.node_size = kwargs.get("node_size", 0.4)
        self.node_label = kwargs.get("node_label", {})
        self.node_label_size = kwargs.get("node_label_size", 8)
        self.edge_label = kwargs.get("edge_label", {})
        self.edge_size = kwargs.get("edge_size", 0.4)
        self.edge_opacity = kwargs.get("edge_opacity", 1)

        NetworkPlot.__init__(
            self,
            data,
            **kwargs,
        )
        Scene.__init__(self)
        self.data = data

    def compute_edge_index(self) -> tuple:
        """
        Convert input data into edge tuples and compute maximum time value.

        Returns:
            tuple:
                A tuple containing:

                - `tedges` (list of tuple): A list of temporal edges, where each edge is represented as
                `(source, target, timestamp)`.
                - `max_time` (int): The maximum timestamp found in the edge data.
        """

        tedges = [(d["source"], d["target"], d["start"]) for d in self.data["edges"]]
        max_time = max(d["start"] for d in self.data["edges"])
        return tedges, max_time

    def get_layout(self, graph: pp.TemporalGraph, type: str = "fr", time_window: tuple = None) -> dict:
        """
        Compute spatial layout for network nodes using pathpy layout functions.

        Args:
            graph (pp.TemporalGraph): Graph for which to compute layout.
            type (str, optional): Layout algorithm to use (e.g., "fr", "random").
            time_window (tuple, optional): Optional (start, end) for subgraph

        Returns:
            dict: Mapping from node IDs to 3D positions (x , y , z)
        """
        layout_style = {}
        layout_style["layout"] = type

        layout = pp.layout(graph.to_static_graph(time_window), **layout_style)
        for key in layout.keys():
            layout[key] = np.append(
                layout[key], 0.0
            )  # manim works in 3 dimensions, not 2 --> add zeros as third dimension to every node coordinate

        layout_array = np.array(list(layout.values()))
        mins = layout_array.min(axis=0)  # compute the mins and maxs of the 3 dimensions
        maxs = layout_array.max(axis=0)
        center = (mins + maxs) / 2  # compute the center of the network
        scale = (
            4.0 / (maxs - mins).max() if (maxs - mins).max() != 0 else 1.0
        )  # compute scale, so that every node fits into a 2 x 2 box

        for k in layout:
            layout[k] = (layout[k] - center) * scale  # scale the position of each node

        return layout

    def get_colors(self, g: pp.TemporalGraph) -> dict:
        """
        Compute colors for nodes and edges based on user input and colormaps.

        Args:
            g (pp.TemporalGraph): Input temporal graph.

        Returns:
            dict: Dictionary mapping node/edge identifiers to colors.
        """
        color_dict = {}
        if isinstance(self.node_color, str):
            for node in g.nodes:
                color_dict[node] = self.node_color

        elif (
            isinstance(self.node_color, tuple)
            and len(self.node_color) == 3
            and all(isinstance(c, (int, float)) for c in self.node_color)
        ):
            rbg_norm = tuple(x / 255 for x in self.node_color)
            for node in g.nodes:

                color_dict[node] = mcolors.to_hex(rbg_norm)

        elif self.node_cmap is not None and isinstance(self.node_color, (int, float)):
            node_color = self.node_cmap(self.node_color)[:3]
            node_color = mcolors.to_hex(node_color)
            for node in g.nodes:
                color_dict[node] = node_color

        elif isinstance(self.node_color, list) and all(isinstance(item, str) for item in self.node_color):
            for i, node in enumerate(g.nodes):
                color_dict[node] = self.node_color[i % len(self.node_color)]

        elif (
            isinstance(self.node_color, list)
            and all(isinstance(item, (int, float)) for item in self.node_color)
            and self.node_cmap is not None
        ):
            color_list = []
            for color in self.node_color:
                color_list.append(mcolors.to_hex(self.node_cmap(color)[:3]))
            node_color = color_list
            for i, node in enumerate(g.nodes):
                color_dict[node] = node_color[i % len(node_color)]

        elif isinstance(self.node_color, list) and all(isinstance(item, tuple) for item in self.node_color):
            for node, t, color in self.node_color:
                if isinstance(color, (int, float)) and self.node_cmap != None:

                    if t == self.start:  # node gets initialized with the right color
                        color_dict[node] = color
                    else:
                        color_dict[(node, t)] = color

        # colors of edges
        if isinstance(self.edge_color, str):
            for edge in g.temporal_edges:
                v, w, t = edge
                edge = v, w
                color_dict[(edge, t)] = self.edge_color

        elif (
            isinstance(self.edge_color, tuple)
            and len(self.edge_color) == 3
            and all(isinstance(c, (int, float)) for c in self.edge_color)
        ):
            rbg_norm = tuple(x / 255 for x in self.edge_color)
            for edge in g.temporal_edges:
                v, w, t = edge
                edge = v, w
                color_dict[(edge, t)] = mcolors.to_hex(rbg_norm)

        elif self.edge_cmap is not None and isinstance(self.edge_color, (int, float)):
            edge_color = self.edge_cmap(self.edge_color)[:3]
            edge_color = mcolors.to_hex(edge_color)
            print(edge_color)
            for edge in g.temporal_edges:
                v, w, t = edge
                edge = v, w
                color_dict[(edge, t)] = edge_color

        elif isinstance(self.edge_color, list) and all(isinstance(item, str) for item in self.edge_color):
            for i, temporal_edge in enumerate(g.temporal_edges):
                v, w, t = temporal_edge
                edge = (v, w)
                color_dict[(edge, t)] = self.edge_color[i % len(self.edge_color)]

        elif (
            isinstance(self.edge_color, list)
            and all(isinstance(item, (int, float)) for item in self.edge_color)
            and self.node_cmap is not None
        ):
            color_list = []
            for color in self.edge_color:
                color_list.append(mcolors.to_hex(self.node_cmap(color)[:3]))
            edge_color = color_list
            for i, temporal_edge in enumerate(g.temporal_edges):
                v, w, t = temporal_edge
                edge = (v, w)
                color_dict[(edge, t)] = edge_color[i % len(edge_color)]

        elif isinstance(self.edge_color, list) and all(isinstance(item, tuple) for item in self.edge_color):
            for edge, t, color in self.edge_color:
                if isinstance(color, (int, float)) and self.node_cmap != None:
                    color = mcolors.to_hex(self.node_cmap(color)[:3])
                print(edge)
                color_dict[(edge, t)] = color
        return color_dict

    def construct(self):
        """
        Construct and animate the network scene using Manim.

        This method:
            - Adds nodes using `Graph`
            - Draws and removes temporal edges frame-by-frame
            - Recomputes layout dynamically (if specified)
            - Displays timestamps

        """

        edge_list, end_time = self.compute_edge_index()
        g = pp.TemporalGraph.from_edge_list(edge_list)  # create ppG Graph

        start = self.start  # start time of the simulation
        end = end_time if self.end is None else self.end  # end time of the simulation
        delta = self.delta  # time needed for progressing one time step
        intervals = (
            self.intervals
        )  # number of numeric intervals, if None --> intervals = num of timesteps (end - start)
        dynamic_layout_interval = (
            self.dynamic_layout_interval
        )  # specifies after how many time steps a new layout is computed

        # if intervals is not specified, every timestep is an interval
        if intervals is None:
            intervals = end - start

        delta /= 1000  # convert milliseconds to seconds

        # colors of nodes
        color_dict = self.get_colors(g)

        layout = self.get_layout(g, "random" if dynamic_layout_interval != None else "fr")

        time_stamps = g.data["time"]
        time_stamps = [timestamp.item() for timestamp in time_stamps]
        time_stamp_dict = dict((time, []) for time in time_stamps)
        for v, w, t in g.temporal_edges:
            time_stamp_dict[t].append((v, w))

        graph = Graph(
            g.nodes,
            [],
            layout=layout,
            labels=False,
            vertex_config={
                v: {
                    "radius": (self.node_size.get(v, 0.4) if isinstance(self.node_size, dict) else self.node_size),
                    "fill_color": color_dict[v] if isinstance(color_dict, dict) and v in color_dict else BLUE,
                    "fill_opacity": (
                        self.node_opacity.get(v, 1) if isinstance(self.node_opacity, dict) else self.node_opacity
                    ),
                }
                for v in g.nodes
            },
        )
        self.add(graph)  # create initial nodes

        # add labels
        for node, label_text in self.node_label.items():
            label = Text(label_text, font_size=self.node_label_size).set_color(BLACK)
            label.next_to(graph[node], UP, buff=0.05)
            self.node_label[node] = label
            self.add(label)

        step_size = int((end - start + 1) / intervals)  # step size based on the number of intervals
        time_window = range(start, end + 1, step_size)

        change = False
        for time_step in tqdm(time_window):
            range_stop = time_step + step_size
            range_stop = range_stop if range_stop < end + 1 else end + 1

            if step_size == 1 or time_step == end:
                text = Text(f"T = {time_step}").set_color(BLACK)
            else:
                text = Text(f"T = {time_step} to T = {range_stop - 1}").set_color(BLACK)
            text.to_corner(UL)
            self.add(text)

            for step in range(time_step, range_stop, 1):
                # dynamic layout change
                if (
                    dynamic_layout_interval != None
                    and (step - start) % dynamic_layout_interval == 0
                    and step - start != 0
                    and change
                ):  # change the layout based on the edges since the last change until the current timestep and only if there were edges in the last interval
                    change = False
                    new_layout = self.get_layout(g, time_window=(step - dynamic_layout_interval, step))

                    animations = []
                    for node in g.nodes:
                        if node in new_layout:
                            new_pos = new_layout[node]
                            animations.append(graph[node].animate.move_to(new_pos))
                            # also change the positions of the labels
                            if node in self.node_label:
                                label = self.node_label[node]
                                animations.append(label.animate.move_to(new_pos + (0, 0.125, 0)))

                    self.play(*animations, run_time=delta)

                # color change
                for node in g.nodes:
                    if (node, step) in color_dict:
                        graph[node].set_fill(color_dict[(node, step)])

            lines = []
            for step in range(time_step, range_stop, 1):  # generate Lines for all the timesteps in the current interval
                if step in time_stamp_dict:
                    for edge in time_stamp_dict[step]:
                        u, v = edge
                        sender = graph[u].get_center()
                        receiver = graph[v].get_center()
                        stroke_width = (
                            self.edge_size.get((edge, step), 0.4)
                            if isinstance(self.edge_size, dict)
                            else self.edge_size
                        )
                        stroke_opacity = (
                            self.edge_opacity.get((edge, step), 1)
                            if isinstance(self.edge_opacity, dict)
                            else self.edge_opacity
                        )

                        color = color_dict[(edge, step)] if (edge, step) in color_dict else GRAY
                        line = Line(
                            sender,
                            receiver,
                            stroke_width=stroke_width,
                            color=color,
                            stroke_opacity=stroke_opacity,
                        )
                        lines.append(line)
            if len(lines) > 0:
                change = True
                self.add(*lines)
                self.wait(delta)
                self.remove(*lines)
            else:
                self.wait(delta)

            self.remove(text)


class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a static network."""

    _kind = "static"
