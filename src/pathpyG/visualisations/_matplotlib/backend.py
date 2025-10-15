"""Matplotlib backend for raster graphics network visualization.

High-performance matplotlib implementation with optimized collections for
efficient rendering. Supports both directed and undirected networks with
curved edges, proper arrowheads, and comprehensive styling options.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection, PathCollection
from matplotlib.path import Path

from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.utils import unit_str_to_float

logger = logging.getLogger("root")

SUPPORTED_KINDS = {
    NetworkPlot: "static",
}


class MatplotlibBackend(PlotBackend):
    """Matplotlib backend for network visualization with optimized rendering.

    Uses matplotlib collections (EllipseCollection, LineCollection, PathCollection)
    for efficient batch rendering of network elements. Provides high-quality
    output with proper edge-node intersection handling and curved edge support.

    Features:
        - Batch rendering via matplotlib collections
        - Bezier curves for directed edges
        - Automatic edge shortening to avoid node overlap

    !!! note "Performance Optimization"
        Uses collections instead of individual plot calls for 10-100x
        faster rendering on networks with many edges.
    """

    def __init__(self, plot: PathPyPlot, show_labels: bool):
        """Initialize matplotlib backend with plot validation.

        Args:
            plot: PathPyPlot instance containing network data
            show_labels: Whether to display node labels

        Raises:
            ValueError: If plot type not supported by matplotlib backend
        """
        super().__init__(plot, show_labels=show_labels)
        self._kind = SUPPORTED_KINDS.get(type(plot), None)  # type: ignore[arg-type]
        if self._kind is None:
            logger.error(f"Plot of type {type(plot)} not supported by Matplotlib backend.")
            raise ValueError(f"Plot of type {type(plot)} not supported.")

    def save(self, filename: str) -> None:
        """Save plot to file with automatic format detection.

        Args:
            filename: Output file path (format inferred from extension)
        """
        fig, ax = self.to_fig()
        fig.savefig(filename)

    def show(self) -> None:
        """Display plot in interactive matplotlib window.

        Opens plot in default matplotlib backend for interactive exploration.
        """
        fig, ax = self.to_fig()
        plt.show()

    def to_fig(self) -> tuple[plt.Figure, plt.Axes]:
        """Generate complete matplotlib figure with network visualization.

        Creates figure with proper sizing, renders edges and nodes using optimized
        collections, adds labels if enabled, and sets appropriate axis limits.

        Returns:
            tuple: (Figure, Axes) matplotlib objects ready for display/saving

        !!! info "Rendering Pipeline"
            1. **Setup**: Create figure with configured dimensions and DPI
            2. **Edges**: Render using LineCollection (undirected) or PathCollection (directed)
            3. **Nodes**: Render using EllipseCollection for precise sizing
            4. **Labels**: Add text annotations at node centers
            5. **Layout**: Set axis limits with margin configuration
        """
        size_factor = 1 / 200  # scale node size to reasonable values
        fig, ax = plt.subplots(
            figsize=(unit_str_to_float(self.config["width"], "in"), unit_str_to_float(self.config["height"], "in")),
            dpi=150,
        )
        ax.set_axis_off()

        # get source and target coordinates for edges
        source_coords = self.data["nodes"].loc[self.data["edges"].index.get_level_values("source"), ["x", "y"]].values
        target_coords = self.data["nodes"].loc[self.data["edges"].index.get_level_values("target"), ["x", "y"]].values

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
                # Annotate the node label with text in the center of the node
                ax.annotate(
                    label,
                    (x, y),
                    fontsize=0.4 * self.data["nodes"]["size"].mean(),
                    ha="center",
                    va="center",
                )

        # set limits
        ax.set_xlim(-1 * self.config["margin"], 1 + (1*self.config["margin"]))
        ax.set_ylim(-1 * self.config["margin"], 1 + (1*self.config["margin"]))
        return fig, ax

    def add_undirected_edges(self, source_coords, target_coords, ax, size_factor):
        """Render undirected edges using LineCollection for efficiency.

        Computes edge shortening to prevent overlap with nodes and renders
        all edges in a single matplotlib LineCollection for optimal performance.

        Args:
            source_coords: Source node coordinates array
            target_coords: Target node coordinates array  
            ax: Matplotlib axes for rendering
            size_factor: Scaling factor for node size calculations

        !!! tip "Edge Shortening"
            Automatically shortens edges by node radius to create clean
            visual separation between edges and node boundaries.
        """
        # shorten edges so they don't overlap with nodes
        vec = target_coords - source_coords
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        direction = vec / dist
        source_coords += direction * (self.data["nodes"].loc[self.data["edges"].index.get_level_values("source"), ["size"]].values * (size_factor / 2))  # /2 because we use radius instead of diameter
        target_coords -= direction * (self.data["nodes"].loc[self.data["edges"].index.get_level_values("target"), ["size"]].values * (size_factor / 2))

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
        """Render directed edges using Bezier curves with arrowheads.

        Creates curved edges using quadratic Bezier curves and adds proportional
        arrowheads. Handles edge shortening and automatic fallback to straight
        edges when curves would be too short.

        Args:
            source_coords: Source node coordinates array
            target_coords: Target node coordinates array
            ax: Matplotlib axes for rendering  
            size_factor: Scaling factor for node size calculations

        !!! warning "Curve Limitations"
            Falls back to straight edges when arrowheads would be too large
            relative to edge length to maintain visual clarity.
        """
        # get bezier curve vertices and codes
        head_length = 0.02
        vertices, codes = self.get_bezier_curve(
            source_coords,
            target_coords,
            source_node_size=self.data["nodes"].loc[self.data["edges"].index.get_level_values("source"), ["size"]].values
            * (size_factor / 2),  # /2 because we use radius instead of diameter
            target_node_size=self.data["nodes"].loc[self.data["edges"].index.get_level_values("target"), ["size"]].values
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
        shorten=0.005,
    ):
        """Generate quadratic Bezier curve paths for directed edges.

        Computes control points for smooth curved edges with automatic shortening
        to accommodate node sizes and arrowheads. Uses perpendicular offset for
        curve control points based on curvature configuration.

        Args:
            source_coords: Start points (x, y) for all edges
            target_coords: End points (x, y) for all edges  
            source_node_size: Source node radii for edge shortening
            target_node_size: Target node radii for edge shortening
            head_length: Arrowhead length for target-end shortening
            shorten: Additional shortening amount to prevent visual overlap

        Returns:
            tuple: (vertices, codes) for matplotlib Path objects

        !!! info "Bezier Curve Mathematics"
            Uses quadratic Bezier curves with control point positioned
            perpendicular to edge midpoint. Curvature parameter controls
            the distance of control point from edge midpoint.

        !!! note "Fallback Behavior" 
            Returns straight line paths when curves would be too short
            for proper arrowhead placement.
        """
        # Start and end points for the Bézier curve
        P0 = source_coords
        P2 = target_coords

        # Calculate distance and direction vector
        mid_point = (P0 + P2) / 2
        vec = P2 - P0
        dist = np.linalg.norm(vec, axis=1, keepdims=True)
        # Avoid division by zero
        dist[dist == 0] = 1e-6

        # Perpendicular vector
        perp_vec = np.array([-vec[:, 1], vec[:, 0]]).T / dist

        # Calculate control points
        P1 = mid_point + perp_vec * dist * self.config["curvature"]

        # Shorten the curve to avoid overlap with nodes
        distance_P0_P1 = np.linalg.norm(P1 - P0, axis=1, keepdims=True)
        distance_P0_P1[distance_P0_P1 == 0] = 1e-6
        distance_P2_P1 = np.linalg.norm(P1 - P2, axis=1, keepdims=True)
        distance_P2_P1[distance_P2_P1 == 0] = 1e-6
        direction_P0_P1 = (P1 - P0) / distance_P0_P1
        direction_P2_P1 = (P1 - P2) / distance_P2_P1
        P0_offset_dist = shorten + source_node_size
        P2_offset_dist = shorten + target_node_size + (head_length * self.data["edges"]["size"].values[:, np.newaxis])
        if np.any(distance_P2_P1/2 < P2_offset_dist):
            logger.warning("Arrowhead length is too long for some edges. Please reduce the edge size. Using non-curved edges instead.")
            direction_P0_P2 = vec / dist
            P0 += direction_P0_P2 * P0_offset_dist
            P2 -= direction_P0_P2 * P2_offset_dist
            return [P0, P2], [Path.MOVETO, Path.LINETO]
        
        P0 += direction_P0_P1 * P0_offset_dist
        P2 += direction_P2_P1 * P2_offset_dist

        vertices = [P0, P1, P2]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.MOVETO,
        ]
        return vertices, codes

    def get_arrowhead(self, vertices, head_length=0.01, head_width=0.02):
        """Generate triangular arrowhead paths for directed edges.

        Creates proportional arrowheads at curve endpoints using tangent vectors
        for proper orientation. Arrowhead size scales with edge width for
        consistent visual appearance across different edge weights.

        Args:
            vertices: Bezier curve vertices list for tangent calculation
            head_length: Base arrowhead length (scaled by edge size)
            head_width: Base arrowhead width (scaled by edge size)

        Returns:
            tuple: (vertices, codes) for matplotlib Path objects

        !!! tip "Proportional Scaling"
            Arrowhead dimensions automatically scale with edge width
            to maintain consistent visual proportions across different
            edge weights in the same network.
        """
        # Extract the last segment of the Bézier curve
        P1, P2 = vertices[-2], vertices[-1]
        # 1. Calculate the tangent vector (direction of the curve at the end)
        # For a quadratic curve, this is the vector from the control point to the end point.
        tangent = P2 - P1
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        # Avoid division by zero
        tangent[tangent == 0] = 1e-6

        # 2. Calculate the perpendicular vector for the width
        perp = np.array([-tangent[:, 1], tangent[:, 0]]).T

        # 3. Define the three points of the arrowhead triangle
        base_center = P2
        tip = P2 + tangent * head_length * self.data["edges"]["size"].values[:, np.newaxis]
        wing1 = base_center + perp * head_width / 2 * self.data["edges"]["size"].values[:, np.newaxis]
        wing2 = base_center - perp * head_width / 2 * self.data["edges"]["size"].values[:, np.newaxis]

        vertices = [wing1, tip, wing2, wing1]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,  # Close the shape to make it fillable
        ]
        return vertices, codes
