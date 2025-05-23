"""Network plots with manim."""

# =============================================================================
# File      : network_plots.py -- Network plots with manim
# =============================================================================

from typing import Any
import numpy as np
import logging
import pathpyG as pp
from manim import *
from pathlib import Path
from tqdm import tqdm
from pathpyG.visualisations._manim.core import ManimPlot
from matplotlib.pyplot import get_cmap
import matplotlib.colors as mcolors

logger = logging.getLogger("root")


class NetworkPlot(ManimPlot):
    """Network plot class for a network."""

    _kind = "network"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.data = {}
        self.config = kwargs
        self.raw_data = data


class TemporalNetworkPlot(NetworkPlot, Scene):
    """Network plot class for a temporal network.
    **Temporal properties:**

    - ``start`` : start time of the simulation

    - ``end`` : end time of the simulation

    - ``delta`` : time needed for progressing one time step

    - ``intervals`` : number of numeric intervals

    -``dynamic_layout_intervals``: specifies after how many time steps a new layout is computed
    """

    _kind = "temporal"

    def __init__(self, data: dict, output_dir: str | Path = None, output_file: str = None, **kwargs) -> None:
        """Initialize network plot class."""
        from manim import config as manim_config

        if output_dir:
            manim_config.media_dir = str(Path(output_dir).resolve())
        if output_file:
            manim_config.output_file = output_file

        # Optional config settings
        manim_config.pixel_height = 1080
        manim_config.pixel_width = 1920
        manim_config.frame_rate = 15
        manim_config.quality = "medium_quality"
        manim_config.background_color = DARK_GREY

        self.delta = kwargs.get("delta", 1000)
        self.start = kwargs.get("start", 0)
        self.end = kwargs.get("end", None)
        self.intervals = kwargs.get("intervals", None)
        self.dynamic_layout_interval = kwargs.get("dynamic_layout_interval", 5)
        self.node_color = kwargs.get("node_color", BLUE)
        self.edge_color = kwargs.get("edge_color", GRAY)
        self.node_cmap = kwargs.get("nodes_cmap", get_cmap())
        self.edge_cmap = kwargs.get("edge_cmap", get_cmap())
        self.node_opacity = kwargs.get("node_opacity", None)
        self.node_size = kwargs.get("node_size", None)
        self.node_label = kwargs.get("node_label", None)
        self.edge_label = kwargs.get("edge_label", None)
        self.edge_size = kwargs.get("edge_size", None)
        self.edge_opacity = kwargs.get("edge_opacity", None)

        NetworkPlot.__init__(self, data, **kwargs)
        Scene.__init__(self)
        self.data = data

    def compute_edg_index(self):
        """Compute Edge Index from data for ppG graph"""
        tedges = [(d["source"], d["target"], d["start"]) for d in self.data["edges"]]
        max_time = max(d["start"] for d in self.data["edges"])
        return tedges, max_time

    def get_layout(self, graph: pp.TemporalGraph, type: str = "fr", time_window: tuple = None):
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

    def get_colors(self, g: pp.TemporalGraph):
        color_dict = {}
        if isinstance(self.node_color, str):
            for node in g.nodes:
                color_dict[node] = self.node_color

        elif self.node_cmap != None and isinstance(self.node_color, (int, float)):
            node_color = self.node_cmap(self.node_color)[:3]
            #node_color = mcolors.to_hex(node_color)
            for node in g.nodes:
                color_dict[node] = node_color

        elif isinstance(self.node_color, list) and all(isinstance(item, str) for item in self.node_color):
            for i, node in enumerate(g.nodes):
                color_dict[node] = self.node_color[i % len(self.node_color)]

        elif (
            isinstance(self.node_color, list)
            and all(isinstance(item, (int, float)) for item in self.node_color)
            and self.node_cmap != None
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

        elif self.edge_cmap != None and isinstance(self.edge_color, (int, float)):
            edge_color = self.edge_cmap(self.edge_color)[:3]
            #edge_color = mcolors.to_hex(edge_color)
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
            and self.node_cmap != None
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
        """Construct Manim Scene from Graph Data"""

        edge_list, end_time = self.compute_edg_index()
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
                v: {"radius": 0.4, "fill_color": color_dict[v] if v in color_dict else BLUE} for v in g.nodes
            },
        )
        self.add(graph)  # create initial nodes
        step_size = int((end - start + 1) / intervals)  # step size based on the number of intervals
        time_window = range(start, end + 1, step_size)

        change = False
        for time_step in tqdm(time_window):
            range_stop = time_step + step_size
            range_stop = range_stop if range_stop < end + 1 else end + 1

            if step_size == 1 or time_step == end:
                text = Text(f"T = {time_step}")
            else:
                text = Text(f"T = {time_step} to T = {range_stop - 1}")
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
                        line = Line(
                            sender,
                            receiver,
                            stroke_width=0.4,
                            color=color_dict[(edge, step)] if (edge, step) in color_dict else GRAY,
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
    """Network plot class for a temporal network."""

    _kind = "static"
