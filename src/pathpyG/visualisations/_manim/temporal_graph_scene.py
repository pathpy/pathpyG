import logging

import pandas as pd
from manim import Arrow, Create, Graph, Scene, Uncreate, GrowArrow

# set manim log level to warning
logging.getLogger("manim").setLevel(logging.WARNING)


class TemporalGraphScene(Scene):
    def __init__(self, data: dict, config: dict, show_labels: bool):
        super().__init__()
        self.data = data
        self.data["edges"].index = pd.MultiIndex.from_tuples(
            self.data["edges"][["source", "target"]].itertuples(index=False)
        )
        self.config = config
        self.show_labels = show_labels

    def construct(self):
        """Constructs the Manim scene for the temporal graph."""
        self.data["nodes"]["size"] *= 0.025  # scale sizes down
        vertex_config = (
            self.data["nodes"][["size", "color", "opacity"]]
            .rename(columns={"size": "radius", "color": "fill_color", "opacity": "fill_opacity"})
            .to_dict(orient="index")
        )
        g = Graph(
            vertices=self.data["nodes"].index.tolist(),
            edges=[],
            vertex_config=vertex_config,
            labels=self.show_labels,
        )
        self.play(Create(g))
        self.wait()
        for t in range(self.data["edges"]["end"].max() + 1):
            # Gather all new edges to be added
            animations = []
            new_edges = self.data["edges"][self.data["edges"]["start"] == t]
            new_edge_config = (
                new_edges[["color", "opacity", "size"]]
                .rename(columns={"color": "stroke_color", "opacity": "stroke_opacity", "size": "stroke_width"})
                .to_dict(orient="index")
            )
            if not new_edges.empty:
                edge_list = [(row[0], row[1]) for row in new_edges[["source", "target"]].itertuples(index=False)]
                animations.append(g.animate(animation=GrowArrow).add_edges(*edge_list, edge_config=new_edge_config, edge_type=Arrow))

            # Gather all old edges to be removed
            old_edges = self.data["edges"][self.data["edges"]["end"] == t]
            if not old_edges.empty:
                edge_list = [(row[0], row[1]) for row in old_edges[["source", "target"]].itertuples(index=False)]
                removed_edges = g.remove_edges(*edge_list)
                animations.extend([removed_edge.animate.scale(0, scale_tips=True, about_point=removed_edge.get_end()) for removed_edge in removed_edges])

            # play all animations
            if animations:
                self.play(*animations)
        self.wait()
        self.play(Uncreate(g))
