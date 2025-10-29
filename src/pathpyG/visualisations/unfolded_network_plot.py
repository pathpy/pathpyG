"""Time-unfolded temporal network visualisation module.

Prepares temporal graphs for visualization as time-unfolded networks
by assigning the node positions in a grid.
"""

import numpy as np
import pandas as pd

from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot


class TimeUnfoldedNetworkPlot(TemporalNetworkPlot):
    """Time-unfolded temporal network visualisation class.

    Prepares temporal graphs for visualization as time-unfolded networks
    by assigning the node positions in a grid.

    Inherits from TemporalNetworkPlot.
    """

    _kind = "unfolded"
    network: TemporalGraph

    def _compute_edge_data(self):
        super()._compute_edge_data()
        self.data["edges"].index = pd.MultiIndex.from_arrays(
            [
                list(zip(self.data["edges"].index.get_level_values("source"), self.data["edges"]["start"])),
                list(zip(self.data["edges"].index.get_level_values("target"), self.data["edges"]["end"])),
            ],
            names=["source", "target"],
        )

    def _post_process_node_data(self):
        super()._post_process_node_data()

        self.data["nodes"].index = pd.Index(
            list(zip(self.data["nodes"].index, self.data["nodes"]["start"])),
            name="uid",
            tupleize_cols=False
        )

    def _compute_layout(self) -> None:
        """Compute time-unfolded node layout.

        For each node, assign positions in a grid based on time steps.
        Depending on orientation, x (left/right) or y (up/down) coordinates represent time steps
        and the other coordinate represents node identity.
        """
        num_nodes = self.network.n
        max_time = int(
            max(self.data["nodes"].index.get_level_values("time").max() + 1, self.data["edges"]["end"].max() + 1)
        )
        orientation = self.config.get("orientation")

        # Determine coordinate assignment based on orientation
        if orientation in ["left", "right"]:
            time_coord = "x"
            node_coord = "y"
            if orientation == "left":
                sign = -1
            else:
                sign = 1
        elif orientation in ["up", "down"]:
            time_coord = "y"
            node_coord = "x"
            if orientation == "down":
                sign = -1
            else:
                sign = 1
        else:
            raise ValueError("Invalid orientation option. Choose from 'left', 'right', 'up', or 'down'.")

        # Create a DataFrame for the grid layout
        node_ids = np.repeat(self.data["nodes"].index.get_level_values("uid").unique(), max_time)
        node_values = np.repeat(np.arange(num_nodes), max_time)
        time_values = np.tile(np.arange(max_time), num_nodes)
        layout_df = pd.DataFrame(
            {
                "uid": node_ids,
                "time": time_values,
                time_coord: (sign * time_values).astype(float),
                node_coord: node_values.astype(float),
            }
        ).set_index(["uid", "time"])

        # Scale coordinates between 0 and 1
        layout_df[time_coord] = (layout_df[time_coord] - layout_df[time_coord].min()) / (
            layout_df[time_coord].max() - layout_df[time_coord].min()
        )
        layout_df[node_coord] = (layout_df[node_coord] - layout_df[node_coord].min()) / (
            layout_df[node_coord].max() - layout_df[node_coord].min()
        )

        # Join the layout DataFrame with the existing node data
        self.data["nodes"] = self.data["nodes"].join(layout_df, how="outer")

    def _compute_config(self) -> None:
        """Set temporal-specific visualization configuration.

        Forces directed=True and curved=False for temporal networks.
        Enables simulation mode (for `d3js` backend) when no layout algorithm is specified.
        """
        self.config["directed"] = True
        self.config["curved"] = False
        self.config["simulation"] = False
