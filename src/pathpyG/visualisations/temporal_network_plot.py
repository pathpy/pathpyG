"""Temporal network visualization module.

Prepares temporal graphs for visualization, handling time-based
node and edge dynamics, windowed layout computation, and
attribute interpolation.
"""

from __future__ import annotations

import logging
from math import ceil
from typing import TYPE_CHECKING, Any

import pandas as pd

from pathpyG.visualisations.layout import layout as network_layout
from pathpyG.visualisations.network_plot import NetworkPlot

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.temporal_graph import TemporalGraph

# create logger
logger = logging.getLogger("root")


class TemporalNetworkPlot(NetworkPlot):
    """Temporal network visualization with time-based node and edge dynamics.

    Extends NetworkPlot to handle temporal graphs where edges appear at
    fixed times. Provides windowed layout computation and
    time-aware attribute interpolation.

    !!! info "Temporal Features"
        - Node lifetime tracking (start/end times)
        - Windowed layout computation
        - Time-based attribute interpolation
    """

    _kind = "temporal"
    network: TemporalGraph

    def __init__(self, network: TemporalGraph, **kwargs: Any) -> None:
        """Initialize temporal network plot.

        Args:
            network: TemporalGraph instance to visualize
            **kwargs: Additional plotting parameters
        """
        super().__init__(network, **kwargs)

    def _compute_node_data(self) -> None:
        """Generate temporal node data with time-based attributes.

        Creates multi-index DataFrame with (node_id, time) structure.
        Handles node appearance times and attribute assignment from
        network data, config defaults, and user arguments.
        """
        # initialize values with index `node-0` to indicate time step 0
        start_nodes: pd.DataFrame = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(node, 0) for node in self.network.nodes], names=["uid", "time"])
        )
        new_nodes: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["uid", "time"]))
        # add attributes to start nodes and new nodes if given as dictionary
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("node").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                start_nodes[attribute] = [self.config.get("node").get(attribute, None)] * len(start_nodes)  # type: ignore[union-attr]
            else:
                start_nodes[attribute] = self.config.get("node").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as node attribute
            if f"node_{attribute}" in self.network.node_attrs():
                start_nodes[attribute] = self.network.data[f"node_{attribute}"]
            # check if attribute is given as argument
            if attribute in self.node_args:
                if isinstance(self.node_args[attribute], dict):
                    # check if entry is tuple or string
                    for key in self.node_args[attribute].keys():  # type: ignore[union-attr]
                        if isinstance(key, tuple):
                            # add node attribute according to node-time keys
                            new_nodes.loc[key, attribute] = self.node_args[attribute][key]  # type: ignore[index]
                        else:
                            # add node attributes to start nodes according to node keys
                            start_nodes.loc[(key, 0), attribute] = self.node_args[attribute][key]  # type: ignore[index]
                else:
                    start_nodes[attribute] = self.node_args[attribute]

        # save node data and combine start nodes with new nodes by making sure start nodes are overwritten
        self.data["nodes"] = new_nodes.combine_first(start_nodes)

    def _post_process_node_data(self) -> pd.DataFrame:
        """Add node lifetime information and forward-fill attributes.

        Computes start/end times for each node appearance and fills
        missing attribute values using forward-fill within node groups.

        Returns:
            Processed DataFrame with start/end time columns
        """
        # Post-processing from parent class
        super()._post_process_node_data()

        # Fill all NaN/None values with the previous value and add start/end time columns.
        nodes = self.data["nodes"]
        nodes = nodes.sort_values(by=["uid", "time"]).groupby("uid", sort=False).ffill()
        nodes["start"] = nodes.index.get_level_values("time")
        nodes = nodes.droplevel("time")
        # add end time step with the start the node appears the next time or max time step + 1
        nodes["end"] = nodes.groupby("uid")["start"].shift(-1)
        max_node_time = nodes["start"].max() + 1
        if self.network.data.time.size(0) > 0 and max_node_time < self.network.data.time[-1].item() + 1:
            max_node_time = self.network.data.time[-1].item() + 1
        nodes["end"] = nodes["end"].fillna(max_node_time)
        self.data["nodes"] = nodes

    def _compute_edge_data(self) -> None:
        """Generate temporal edge data with time-based attributes.

        Creates edge DataFrame with temporal index (source, target, time).
        Handles edge attributes from network data, config defaults, and
        user arguments. Adds start/end time columns for edge lifetime.
        """
        # initialize values
        edges: pd.DataFrame = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(self.network.temporal_edges, names=["source", "target", "time"])
        )
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("edge").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                edges[attribute] = [self.config.get("edge").get(attribute, None)] * len(edges)  # type: ignore[union-attr]
            else:
                edges[attribute] = self.config.get("edge").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as edge attribute
            if f"edge_{attribute}" in self.network.edge_attrs():
                edges[attribute] = self.network.data[f"edge_{attribute}"]
            # special case for size: If no edge_size is given use edge_weight if available
            elif attribute == "size" and "edge_weight" in self.network.edge_attrs():
                edges[attribute] = self.network.data["edge_weight"]
            # check if attribute is given as argument
            if attribute in self.edge_args:
                edges = self._assign_argument(attribute, self.edge_args[attribute], edges)
            elif attribute == "size" and "weight" in self.edge_args:
                edges = self._assign_argument("size", self.edge_args["weight"], edges)

        # convert needed attributes to useful values
        edges["color"] = self._convert_to_rgb_tuple(edges["color"])
        edges["color"] = edges["color"].map(self._convert_color)
        edges["start"] = edges.index.get_level_values("time").astype(int)
        edges["end"] = edges["start"] + 1  # assume all edges last for one time step
        edges.index = edges.index.droplevel("time")

        # save edge data
        self.data["edges"] = edges

    def _compute_layout(self) -> None:
        """Compute time-aware node layout using sliding window approach.

        Uses configurable time windows to create smooth layout transitions.
        For each time step, considers edges from surrounding time steps
        based on layout_window_size configuration.

        !!! tip "Window Configuration"
            - Integer: symmetric window around current time
            - [past, future]: asymmetric window sizes
            - Negative values: use all past/future time steps
        """
        # get layout from the config
        layout_type = self.config.get("layout")

        # if no layout is considered or the graph is empty stop this process
        if layout_type is None or len(self.data["nodes"]) == 0:
            return

        max_time = int(
            max(self.data["nodes"].index.get_level_values("time").max() + 1, self.data["edges"]["end"].max())
        )
        window_size = self.config.get("layout_window_size")
        if isinstance(window_size, int):
            # if uneven window size, add one to the future time steps since the end time step is exclusive
            window_size = [window_size // 2, ceil(window_size / 2)]
        elif isinstance(window_size, list | tuple):
            if window_size[0] < 0:
                # use all previous time steps
                window_size[0] = max_time  # type: ignore[index]
            if window_size[1] < 0:
                # use all following time steps
                window_size[1] = max_time  # type: ignore[index]
        elif not isinstance(window_size, (list, tuple)):
            logger.error("The provided layout_window_size is not valid!")
            raise AttributeError

        pos = network_layout(self.network, layout="random")  # initial layout
        num_steps = max(max_time - window_size[1], 0)
        layout_df = pd.DataFrame()
        for step in range(num_steps + 1):
            start_time = max(0, step - window_size[0])
            end_time = step + window_size[1] + 1
            # only compute layout if there are edges in the current window, otherwise use the previous layout
            if ((start_time <= self.network.data.time) & (self.network.data.time <= end_time)).sum() > 0:
                # get subgraph for the current time step
                sub_graph = self.network.get_window(start_time=start_time, end_time=end_time)

                # get layout dict for each node
                if isinstance(layout_type, str):
                    pos = network_layout(sub_graph, layout=layout_type, pos=pos)
                elif not isinstance(layout_type, dict):
                    logger.error("The provided layout is not valid!")
                    raise AttributeError

            # update x,y position of the nodes
            new_layout_df = pd.DataFrame.from_dict(pos, orient="index", columns=["x", "y"])
            if self.network.order > 1 and not isinstance(new_layout_df.index[0], str):
                new_layout_df.index = new_layout_df.index.map(lambda x: self.config["separator"].join(map(str, x)))
            # scale x and y to [0,1]
            new_layout_df["x"] = (new_layout_df["x"] - new_layout_df["x"].min()) / (
                new_layout_df["x"].max() - new_layout_df["x"].min()
            )
            new_layout_df["y"] = (new_layout_df["y"] - new_layout_df["y"].min()) / (
                new_layout_df["y"].max() - new_layout_df["y"].min()
            )
            # add time for the layout
            new_layout_df["time"] = step
            # append to layout df
            layout_df = pd.concat([layout_df, new_layout_df])
        # join layout with node data
        layout_df = layout_df.reset_index().rename(columns={"index": "uid"}).set_index(["uid", "time"])
        self.data["nodes"] = self.data["nodes"].join(layout_df, how="outer")

    def _compute_config(self) -> None:
        """Set temporal-specific visualization configuration.

        Forces directed=True and curved=False for temporal networks.
        Enables simulation mode (for `d3js` backend) when no layout algorithm is specified.
        """
        self.config["directed"] = True
        self.config["curved"] = False
        self.config["simulation"] = self.config["layout"] is None
