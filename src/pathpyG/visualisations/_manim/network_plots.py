"""Network plots with manim."""

# =============================================================================
# File      : network_plots.py -- Network plots with manim
# =============================================================================

from typing import Any
import numpy as np
import logging
from manim import Scene
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

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        NetworkPlot.__init__(self, data, **kwargs)
        Scene.__init__(self)
        
        self.data = data
        self.config = kwargs

 
    def compute_edg_index(self):
        """Compute Edge Index from data for ppG graph"""
        tedges = [(d['source'], d['target'], d['start']) for d in self.data["edges"]]
        return tedges
    
    def construct(self):
        import pathpyG as pp

        edge_list = self.compute_edg_index() 
        g =  pp.TemporalGraph.from_edge_list(edge_list) # create ppG Graph
           
        layout_style = {}
        layout_style['layout'] = 'Fruchterman-Reingold'
        fr_layout = pp.layout(g.to_static_graph(), **layout_style)
        for key in fr_layout.keys():
            fr_layout[key] = np.append(fr_layout[key], 0.0) # manim works in 3 dimensions, not 2 --> add zeros as third dimension to every node coordinate

        layout_array = np.array(list(fr_layout.values()))
        mins = layout_array.min(axis=0)
        maxs = layout_array.max(axis=0)
        center = (mins + maxs) / 2
        scale = 2.0 / (maxs - mins).max()

        for k in fr_layout:
            fr_layout[k] = (fr_layout[k] - center) * scale

        
        time_stamps = g.data['time']
        time_diffs = time_stamps[1:] - time_stamps[:-1]
        time_diffs = time_diffs + 1
        
        time_diffs = [timediff.item() for timediff in time_diffs]
        run_time = 10
        time_diff_sum = np.sum(time_diffs)
        time_diffs = (time_diffs/time_diff_sum) * run_time
        
        graph = Graph(
            g.nodes, [],
            layout=fr_layout,
            labels=False,
            vertex_config={
                v: {"radius": 0.01, "fill_color": BLUE_A} for v in g.nodes

            }
        )

        self.play(Create(graph))  # create initial nodes
        
        # animate edges individually
        edge_animations = []
        edges = []
        for u, v in g.edges:
            start = graph[u].get_center()
            end = graph[v].get_center()
            edge = Line(start, end, stroke_width = 0.2, color = GRAY)
            edges.append(edge)
            # self.play(Create(edge), run_time=run_time)
            # self.wait(time_diffs[i])
            # i += 1
        print(len(edges))
        edges = edges[:9828]

        for edge, delay in tqdm(zip(edges, time_diffs), total=len(edges)):
            
            line = edge
            self.add(line)
            self.wait(delay)         
            self.remove(line)
        




class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "static"
