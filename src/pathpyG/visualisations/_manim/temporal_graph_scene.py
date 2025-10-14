import logging

import numpy as np
from manim import BLACK, RIGHT, UP, Arrow, Create, Dot, GrowArrow, LabeledDot, Scene, Text, Transform, Uncreate

from pathpyG.visualisations.layout import Layout

# set manim log level to warning
logging.getLogger("manim").setLevel(logging.WARNING)
# set root logger
logger = logging.getLogger("root")


class TemporalGraphScene(Scene):
    def __init__(self, data: dict, config: dict, show_labels: bool):
        super().__init__()
        self.data = data
        self.data["nodes"]["size"] *= 0.025  # scale sizes down
        self.data["nodes"] = self.data["nodes"].rename(
            columns={"size": "radius", "color": "fill_color", "opacity": "fill_opacity"}
        )
        if "x" in self.data["nodes"] and "y" in self.data["nodes"]:
            self.data["nodes"][["x", "y"]] = (self.data["nodes"][["x", "y"]] - 0.5) * 5  # scale layout
        self.data["edges"] = self.data["edges"].rename(
            columns={"color": "stroke_color", "opacity": "stroke_opacity", "size": "stroke_width"}
        )
        self.config = config
        self.show_labels = show_labels

    def construct(self):
        """Constructs the Manim scene for the temporal graph."""
        # Add initial nodes
        start_node_df = self.data["nodes"][self.data["nodes"]["start"] == 0]
        if "x" in self.data["nodes"] and "y" in self.data["nodes"]:
            layout = {node: np.concatenate([pos.values, [0]]) for node, pos in start_node_df[["x", "y"]].iterrows()}
        else:
            # Use random layout if no positions are given
            layout = Layout(nodes=start_node_df.index.tolist()).generate_layout()
            # add z coordinate for manim and scale layout
            layout = {node: (np.concatenate([pos, [0]]) - 0.5) * 5 for node, pos in layout.items()}
        vertex_config = start_node_df[["radius", "fill_color", "fill_opacity"]].to_dict(orient="index")
        if self.show_labels:
            nodes = {node: LabeledDot(label=str(node), point=layout[node], **vertex_config[node]) for node in vertex_config}
        else:
            nodes = {node: Dot(point=layout[node], **vertex_config[node]) for node in vertex_config}
        self.play(*[Create(node) for node in nodes.values()])

        # Iterate over time steps and update nodes and edges
        time_text = Text(f"Time: {0}", font_size=24, color=BLACK).to_corner(UP + RIGHT)
        for t in range(self.data["edges"]["end"].max() + 1):
            # Add time step text
            self.play(Transform(time_text, Text(f"Time: {t}", font_size=24, color=BLACK).to_corner(UP + RIGHT)), run_time=0.02)

            # Add edges for current time step
            new_edge_df = self.data["edges"][(self.data["edges"]["start"] == t)]
            # drop duplicate edges
            if new_edge_df.index.duplicated().any():
                logger.warning(f"Dropping duplicate edges at time {t}.")
                new_edge_df = new_edge_df[~new_edge_df.index.duplicated(keep='first')]
            new_edge_config = new_edge_df[["stroke_color", "stroke_opacity", "stroke_width"]].to_dict(orient="index")
            if not new_edge_df.empty:
                arrows = {
                    (source, target): Arrow(
                        start=self.get_boundary_point(
                            center=layout[source],
                            direction=layout[target] - layout[source],
                            radius=nodes[source].radius/2,
                        ),
                        end=self.get_boundary_point(
                            center=layout[target],
                            direction=layout[source] - layout[target],
                            radius=nodes[target].radius/2,
                        ),
                        **new_edge_config[(source, target)],
                    )
                    for source, target in new_edge_df.index
                }
                self.play(*[GrowArrow(arrow) for arrow in arrows.values()], run_time=self.config["temporal"]["delta"]/(4*1000))
            else:
                self.wait(self.config["temporal"]["delta"]/(4*1000))

            # Update node positions for the next time step
            new_nodes = self.data["nodes"][self.data["nodes"]["start"] == (t + 1)]
            if not new_nodes.empty:
                new_vertex_config = new_nodes[["radius", "fill_color", "fill_opacity"]].to_dict(orient="index")
                if "x" in new_nodes and "y" in new_nodes:
                    layout.update({node: np.concatenate([pos.values, [0]]) for node, pos in new_nodes[["x", "y"]].iterrows()})

                if self.show_labels:
                    new_nodes = {
                        node: LabeledDot(label=str(node), point=layout[node], **new_vertex_config[node])
                        for node in new_vertex_config
                    }
                else:
                    new_nodes = {node: Dot(point=layout[node], **new_vertex_config[node]) for node in new_vertex_config}
                movement_animations = [Transform(nodes[node], new_nodes[node]) for node in new_nodes]

                # Update edge positions with moving nodes
                if not new_edge_df.empty:
                    new_arrows = {
                        (source, target): Arrow(
                            start=self.get_boundary_point(
                                center=layout[source],
                                direction=layout[target] - layout[source],
                                radius=new_nodes[source].radius/2,
                            ),
                            end=self.get_boundary_point(
                                center=layout[target],
                                direction=layout[source] - layout[target],
                                radius=new_nodes[target].radius/2,
                            ),
                            **new_edge_config[(source, target)],
                        )
                        for source, target in new_edge_df.index
                        if (source, target) in arrows
                    }
                    movement_animations.extend([Transform(arrows[index], new_arrows[index]) for index in new_arrows])
                self.play(*movement_animations, run_time=self.config["temporal"]["delta"]/(2*1000) - 0.02) # 0.02 for time text update
            else:
                self.wait(self.config["temporal"]["delta"]/(2*1000) - 0.02) # 0.02 for time text update

            # Gather all old edges to be removed
            if not new_edge_df.empty:
                self.play(
                    *[arrow.animate.scale(0, scale_tips=True, about_point=arrow.get_end()) for arrow in arrows.values()],
                    run_time=self.config["temporal"]["delta"]/(4*1000)
                )
            else:
                self.wait(self.config["temporal"]["delta"]/(4*1000))

        self.play(Uncreate(node) for node in nodes.values())

    def get_boundary_point(self, center, direction, radius):
        """Calculate the boundary point of a circle in a given direction."""
        distance = np.linalg.norm(direction)
        if distance == 0:
            return center  # Avoid division by zero
        direction = direction / distance
        return center + direction * radius
