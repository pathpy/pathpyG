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
    """Network plot class for a temporal network."""

    _kind = "temporal"

    def __init__(self, data: dict, output_dir: str | Path = None, output_file: str = None, **kwargs) -> None:
        """Initialize network plot class."""
        from manim import config as manim_config

        # Set the directory and filename *once* here
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

        NetworkPlot.__init__(self, data, **kwargs)
        Scene.__init__(self)
        self.data = data

    def compute_edg_index(self):
        """Compute Edge Index from data for ppG graph"""
        tedges = [(d["source"], d["target"], d["start"]) for d in self.data["edges"]]
        return tedges

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

    def construct(self):
        """Construct Manim Scene from Graph Data"""

        edge_list = self.compute_edg_index()
        g = pp.TemporalGraph.from_edge_list(edge_list)  # create ppG Graph

        start = 0  # start time of the simulation
        end = 31  # end time of the simulation
        delta = 1000  # time needed for progressing one time step
        intervals = None  # number of numeric intervals, if None --> intervals = num of timesteps (end - start)
        dynamic_layout_interval = 5  # specifies after how many time steps a new layout is computed

        # if intrervals is not specified, every timestep is an interval
        if intervals == None:
            intervals = end - start

        delta /= 1000  # convert milliseconds to seconds

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
            vertex_config={v: {"radius": 0.08, "fill_color": BLUE_A} for v in g.nodes},
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
                text = Text(f"T = {time_step} to {range_stop - 1}")
            text.to_corner(UL)
            self.add(text)

            # dynamic layout change
            if (
                not dynamic_layout_interval == None
                and (time_step - start) % dynamic_layout_interval == 0
                and time_step - start != 0
                and change
            ):  # change the layout based on the edges since the last change until the current timestep and only if there were edges in the last interval
                change = False
                new_layout = self.get_layout(g, time_window=(time_step - dynamic_layout_interval, time_step))

                animations = []
                for node in g.nodes:
                    if node in new_layout:
                        new_pos = new_layout[node]
                        animations.append(graph[node].animate.move_to(new_pos))
                self.play(*animations, run_time=delta)

            lines = []
            for step in range(time_step, range_stop, 1):  # generate Lines for all the timesteps in the current interval
                if step in time_stamp_dict:
                    for edge in time_stamp_dict[step]:
                        u, v = edge
                        sender = graph[u].get_center()
                        receiver = graph[v].get_center()
                        line = Line(sender, receiver, stroke_width=0.8, color=GRAY)
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
