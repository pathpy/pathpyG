from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection, PathCollection
from matplotlib.path import Path

from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.utils import unit_str_to_float

logger = logging.getLogger("root")

SUPPORTED_KINDS = {
    NetworkPlot: "static",
}


class MatplotlibBackend(PlotBackend):
    """Matplotlib plotting backend."""

    def __init__(self, plot, show_labels: bool):
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Matplotlib backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save the plot to the hard drive."""
        fig, ax = self.to_fig()
        fig.savefig(filename)

    def show(self) -> None:
        """Show the plot on the device."""
        fig, ax = self.to_fig()
        plt.show()

    def to_fig(self) -> tuple[plt.Figure, plt.Axes]:
        """Convert data to figure."""
        size_factor = 1 / 200  # scale node size to reasonable values
        fig, ax = plt.subplots(
            figsize=(unit_str_to_float(self.config["width"], "in"), unit_str_to_float(self.config["height"], "in")),
            dpi=150,
        )
        ax.set_axis_off()

        # get source and target coordinates for edges
        source_coords = self.data["nodes"].loc[self.data["edges"]["source"], ["x", "y"]].values
        target_coords = self.data["nodes"].loc[self.data["edges"]["target"], ["x", "y"]].values

        if self.config["directed"]:
            self.add_directed_edges(source_coords, target_coords, ax, size_factor)
        else:
            self.add_undirected_edges(source_coords, target_coords, ax, size_factor)

        # plot nodes
        # We use EllipseCollection instead of scatter because there you can specify the radius of each circle in the unit of the data coordinates
        # https://stackoverflow.com/a/33095224
        ax.add_collection(
            EllipseCollection(
                widths=self.data["nodes"]["size"] * size_factor,
                heights=self.data["nodes"]["size"] * size_factor,
                angles=0,
                units="xy",
                offsets=self.data["nodes"][["x", "y"]].values,
                transOffset=ax.transData,
                facecolors=self.data["nodes"]["color"],
                edgecolors="black",
                linewidths=0.5,
                alpha=self.data["nodes"]["opacity"],
                zorder=2,
            )
        )

        # add node labels
        if self.show_labels:
            for label in self.data["nodes"].index:
                x, y = self.data["nodes"].loc[label, ["x", "y"]]
                # Annotate the node label with text above the node
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(0, 0.75 * self.data["nodes"].loc[label, "size"]),
                    textcoords="offset points",
                    fontsize=0.75 * self.data["nodes"]["size"].mean(),
                )

        # set limits
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        return fig, ax

    def add_undirected_edges(self, source_coords, target_coords, ax, size_factor):
        """Add undirected edges to the plot based on LineCollection."""
        # shorten edges so they don't overlap with nodes
        vec = target_coords - source_coords
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        direction = vec / dist
        source_coords += direction * (self.data["nodes"].loc[self.data["edges"]["source"], ["size"]].values * (size_factor / 2))  # /2 because we use radius instead of diameter
        target_coords -= direction * (self.data["nodes"].loc[self.data["edges"]["target"], ["size"]].values * (size_factor / 2))
        
        # create and add lines
        edge_lines = list(zip(source_coords, target_coords))
        ax.add_collection(
            LineCollection(
                edge_lines,
                colors=self.data["edges"]["color"],
                alpha=self.data["edges"]["opacity"],
                linewidths=self.data["edges"]["size"],
                zorder=1,
            )
        )

    def add_directed_edges(self, source_coords, target_coords, ax, size_factor):
        """Add directed edges with arrowheads to the plot based on Bezier curves."""
        # get bezier curve vertices and codes
        head_length = 0.02
        vertices, codes = self.get_bezier_curve(
            source_coords,
            target_coords,
            source_node_size=self.data["nodes"].loc[self.data["edges"]["source"], ["size"]].values
            * (size_factor / 2),  # /2 because we use radius instead of diameter
            target_node_size=self.data["nodes"].loc[self.data["edges"]["target"], ["size"]].values
            * (size_factor / 2),
            head_length=head_length,
        )
        ax.add_collection(
            PathCollection(
                [
                    Path(
                        v,
                        codes,
                    )
                    for v in zip(*vertices)
                ],
                facecolor="none",
                edgecolor=self.data["edges"]["color"],
                alpha=self.data["edges"]["opacity"],
                linewidth=self.data["edges"]["size"],
                zorder=1,
            )
        )

        # add arrowheads
        arrow_vertices, arrow_codes = self.get_arrowhead(vertices, head_length=head_length)
        ax.add_collection(
            PathCollection(
                [Path(v, arrow_codes) for v in zip(*arrow_vertices)],
                facecolor=self.data["edges"]["color"],
                edgecolor=self.data["edges"]["color"],
                alpha=self.data["edges"]["opacity"],
                zorder=1,
            )
        )

    def get_bezier_curve(
        self,
        source_coords,
        target_coords,
        source_node_size,
        target_node_size,
        head_length,
        curvature=0.2,
        shorten=0.005,
    ):
        """Calculates the vertices and codes for a quadratic Bézier curve path.

        Args:
            source_coords (np.array): Start points (x, y) for all edges.
            target_coords (np.array): End points (x, y) for all edges.
            source_node_size (np.array): Size of the source nodes to adjust the curve shortening.
            target_node_size (np.array): Size of the target nodes to adjust the curve shortening.
            head_length (float): Length of the arrowhead to adjust the curve shortening.
            curvature (float): A value controlling the curve's bend.
            shorten (float): Amount to shorten the curve at both ends to avoid overlap with nodes.
                Will shorten double at the target end to make space for the arrowhead.

        Returns:
            tuple: A tuple containing (vertices, codes) for the Path object.
        """
        # Start and end points for the Bézier curve
        P0 = source_coords
        P2 = target_coords

        # Calculate distance and direction vector
        mid_point = (P0 + P2) / 2
        vec = P2 - P0
        dist = np.linalg.norm(vec, axis=1, keepdims=True)

        # Perpendicular vector
        perp_vec = np.array([-vec[:, 1], vec[:, 0]]).T / dist

        # Calculate control points
        P1 = mid_point + perp_vec * dist * curvature

        # Shorten the curve to avoid overlap with nodes
        direction_P0_P1 = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1, keepdims=True)
        direction_P2_P1 = (P1 - P2) / np.linalg.norm(P1 - P2, axis=1, keepdims=True)
        P0 += direction_P0_P1 * (shorten + source_node_size)
        P2 += direction_P2_P1 * (shorten + target_node_size + head_length)

        vertices = [P0, P1, P2]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.MOVETO,
        ]
        return vertices, codes

    def get_arrowhead(self, vertices, head_length=0.01, head_width=0.02):
        """Calculates the vertices and codes for a triangular arrowhead path.

        Args:
            vertices (list): List of vertices from the Bézier curve.
            head_length (float): Length of the arrowhead.
            head_width (float): Width of the arrowhead.

        Returns:
            tuple: A tuple containing (vertices, codes) for the Path object.
        """
        # Extract the last segment of the Bézier curve
        P1, P2 = vertices[1], vertices[2]
        # 1. Calculate the tangent vector (direction of the curve at the end)
        # For a quadratic curve, this is the vector from the control point to the end point.
        tangent = P2 - P1
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

        # 2. Calculate the perpendicular vector for the width
        perp = np.array([-tangent[:, 1], tangent[:, 0]]).T

        # 3. Define the three points of the arrowhead triangle
        base_center = P2
        tip = P2 + tangent * head_length
        wing1 = base_center + perp * head_width / 2
        wing2 = base_center - perp * head_width / 2

        vertices = [wing1, tip, wing2, wing1]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,  # Close the shape to make it fillable
        ]
        return vertices, codes
