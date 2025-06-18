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

import numpy as np
from manim import BLACK, BLUE, GRAY, UL, UP, WHITE, Graph, Line, Scene, Text
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
            **kwargs: Additional keyword arguments to customize the plot.

        """
        from manim import config as manim_config

        NetworkPlot.__init__(
            self,
            data,
            **kwargs,
        )

        if output_dir:
            manim_config.media_dir = str(output_dir)
        if output_file:
            manim_config.output_file = output_file

        # Optional config settings
        manim_config.pixel_height = 1080
        manim_config.pixel_width = 1920
        manim_config.frame_rate = 15
        manim_config.quality = "high_quality"
        manim_config.background_color = self.config.get("background_color", WHITE)

        self.delta = self.config.get("delta", 1000)
        self.start = self.config.get("start", 0)
        self.end = self.config.get("end", None)
        self.intervals = self.config.get("intervals", None)
        self.dynamic_layout_interval = self.config.get("dynamic_layout_interval", None)
        self.font_size = self.config.get("font_size", 8)
        self.look_behind = self.config.get("look_behind", 5)
        self.look_forward = self.config.get("look_forward", 3)

        # defaults
        self.node_size = 0.4
        self.node_opacity = 1
        self.edge_size = 0.4
        self.edge_opacity = 1

        self.node_label: dict[Any, Any] = {}

        Scene.__init__(self)

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

        tedges = [(d["source"], d["target"], d["start"]) for d in self.raw_data["edges"]]
        max_time = max(d["start"] for d in self.raw_data["edges"])
        return tedges, max_time

    def get_layout(self, graph: pp.TemporalGraph, layout_type: str = "fr", time_window: tuple = None) -> dict:
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
        layout_style["layout"] = layout_type
        try:
            layout = pp.layout(
                graph.get_window(*time_window).to_static_graph() if time_window != None else graph.to_static_graph(),
                **layout_style,
                seed=0,
            )
        

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
        except IndexError as e:
                    layout = None
        return layout

    def get_color_at_time(self, node_data: dict, time_step: int):
        """Return Color from Dictionary that provides the color changes per node

        Args:
            node_data (dict): holds all information for a specific node
            time_step (int): timestep for which a color change might occur

        Returns:
            The color the node changes to if any.
        """
        if "color_change" not in node_data:
            return node_data.get("color", BLUE)

        changes = [c for c in node_data["color_change"] if c["time"] <= time_step]
        if not changes:
            return node_data.get("color", BLUE)

        latest_change = max(changes, key=lambda c: c["time"])
        return latest_change["color"]

    def construct(self):
        """
        Construct and animate the network scene using Manim.

        This method:
            - Adds nodes using `Graph`
            - Draws and removes temporal edges frame-by-frame
            - Recomputes layout dynamically (if specified) based on on the temporal edges in the time window between the current step - look_behind and the current step + look_forward
            - Displays timestamps
        """

        nodes_data = self.raw_data["nodes"]
        edges_data = self.raw_data["edges"]
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

        look_behind = self.look_behind
        look_forward = self.look_forward

        delta /= 1000  # convert milliseconds to seconds
        layout = self.get_layout(g, "random" if dynamic_layout_interval is not None else "fr")

        time_stamps = g.data["time"]
        time_stamps = [timestamp.item() for timestamp in time_stamps]
        time_stamp_dict = dict((time, []) for time in time_stamps)
        for v, w, t in g.temporal_edges:
            time_stamp_dict[t].append((v, w))

        graph = Graph(
            [str(v["uid"]) for v in nodes_data],
            [],
            layout=layout,
            labels=False,
            vertex_config={
                str(v["uid"]): {
                    "radius": v.get("size", self.node_size),
                    "fill_color": v.get("color", BLUE),
                    "fill_opacity": (v.get("opacity", self.node_opacity)),
                }
                for v in nodes_data
            },
        )
        self.add(graph)  # create initial nodes

        # add labels
        for node_data in nodes_data:
            node_id = str(node_data["uid"])
            label_text = node_data.get("label", None)
            if label_text is not None:
                label = Text(label_text, font_size=self.font_size).set_color(BLACK)
                label.next_to(graph[node_id], UP, buff=0.05)
                self.node_label[node_id] = label
                self.add(label)

        step_size = int((end - start + 1) / intervals)  # step size based on the number of intervals
        time_window = range(start, end + 1, step_size)

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
                    dynamic_layout_interval is not None
                    and (step - start) % dynamic_layout_interval == 0
                    and step - start != 0
                ):  # change the layout based on the edges since the last change until the current timestep
                    # and only if there were edges in the last interval
                    new_layout = self.get_layout(g, time_window=(step - look_behind, step + look_forward))
                    if new_layout != None:
                        animations = []
                        for node in g.nodes:
                            if node in new_layout:
                                new_pos = new_layout[node]
                                animations.append(graph[node].animate.move_to(new_pos))
                                # also change the positions of the labels
                                if node in self.node_label:
                                    label = self.node_label[node]
                                    offset = graph[node].height / 2 + label.height / 2 + 0.05
                                    animations.append(label.animate.move_to(new_pos + offset * UP))

                        self.play(*animations, run_time=delta)

                # color change
                for node in g.nodes:
                    node_info = next(nd for nd in nodes_data if str(nd["uid"]) == node)
                    color = self.get_color_at_time(node_info, step)
                    graph[node].set_fill(color)

            lines = []
            for step in range(time_step, range_stop, 1):  # generate Lines for all the timesteps in the current interval
                if step in time_stamp_dict:
                    for edge in time_stamp_dict[step]:
                        u, v = edge
                        sender = graph[u].get_center()
                        receiver = graph[v].get_center()

                        s_to_r_vec = receiver - sender  # vector from receiver to sender
                        r_to_s_vec = sender - receiver  # vector from sender to reiceiver
                        # normalize vectors
                        s_to_r_vec = 1 / np.linalg.norm(s_to_r_vec) * s_to_r_vec
                        r_to_s_vec = 1 / np.linalg.norm(r_to_s_vec) * r_to_s_vec

                        node_u_data = next((node for node in nodes_data if str(node.get("uid")) == u), {})
                        node_u_size = node_u_data.get("size", self.node_size)
                        node_v_data = next((node for node in nodes_data if str(node.get("uid")) == v), {})
                        node_v_size = node_v_data.get("size", self.node_size)

                        sender = graph[u].get_center() + (s_to_r_vec * node_u_size)
                        receiver = graph[v].get_center() + (r_to_s_vec * node_v_size)

                        edge_info = next(
                            (
                                e
                                for e in edges_data
                                if e["source"] == f'{u}' and e["target"] == f'{v}' and e["start"] <= step <= e["end"]
                            ),
                            None,
                        )
                        if edge_info:
                            stroke_width = edge_info.get("size", self.edge_size)
                            stroke_opacity = edge_info.get("opacity", self.edge_opacity)
                            color = edge_info.get("color", GRAY)
                        else:
                            stroke_width = self.edge_size
                            stroke_opacity = self.edge_opacity
                            color = GRAY

                        line = Line(
                            sender,
                            receiver,
                            stroke_width=stroke_width,
                            color=color,
                            stroke_opacity=stroke_opacity,
                        )
                        lines.append(line)
            if len(lines) > 0:
                self.add(*lines)
                self.wait(delta)
                self.remove(*lines)
            else:
                self.wait(delta)

            self.remove(text)


class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a static network."""

    _kind = "static"
