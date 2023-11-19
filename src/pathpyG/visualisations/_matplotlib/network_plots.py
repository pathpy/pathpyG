"""Network plots with matplotlib."""
# =============================================================================
# File      : network_plots.py -- Network plots with matplotlib
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Sun 2023-11-19 15:39 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from typing import Any

import logging
from pathpyG.visualisations._matplotlib.core import MatplotlibPlot

# create logger
logger = logging.getLogger("root")


class NetworkPlot(MatplotlibPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.data = data
        self.config = kwargs
        self.generate()

    def generate(self) -> None:
        """Clen up data."""
        self._compute_node_data()
        self._compute_edge_data()

    def _compute_node_data(self) -> None:
        """Generate the data structure for the nodes."""
        default = {
            "uid": None,
            "x": 0,
            "y": 0,
            "size": 30,
            "color": "blue",
            "opacity": 1.0,
        }

        nodes: dict = {key: [] for key in default}

        for node in self.data["nodes"]:
            for key, value in default.items():
                nodes[key].append(node.get(key, value))

        self.data["nodes"] = nodes

    def _compute_edge_data(self) -> None:
        """Generate the data structure for the edges."""
        default = {
            "uid": None,
            "size": 5,
            "color": "red",
            "opacity": 1.0,
        }

        edges: dict = {**{key: [] for key in default}, **{"line": []}}

        for edge in self.data["edges"]:
            source = self.data["nodes"]["uid"].index(edge.get("source"))
            target = self.data["nodes"]["uid"].index(edge.get("target"))
            edges["line"].append(
                [
                    (self.data["nodes"]["x"][source], self.data["nodes"]["x"][target]),
                    (self.data["nodes"]["y"][source], self.data["nodes"]["y"][target]),
                ]
            )

            for key, value in default.items():
                edges[key].append(edge.get(key, value))

        self.data["edges"] = edges

    def to_fig(self) -> Any:
        """Convert data to figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_axis_off()

        # plot edges
        for i in range(len(self.data["edges"]["uid"])):
            ax.plot(
                *self.data["edges"]["line"][i],
                color=self.data["edges"]["color"][i],
                alpha=self.data["edges"]["opacity"][i],
                zorder=1,
            )

        # plot nodes
        ax.scatter(
            self.data["nodes"]["x"],
            self.data["nodes"]["y"],
            s=self.data["nodes"]["size"],
            c=self.data["nodes"]["color"],
            alpha=self.data["nodes"]["opacity"],
            zorder=2,
        )
        return plt


class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a static network."""

    _kind = "static"


class TemporalNetworkPlot(NetworkPlot):
    """Network plot class for a static network."""

    _kind = "temporal"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        raise NotImplementedError


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
