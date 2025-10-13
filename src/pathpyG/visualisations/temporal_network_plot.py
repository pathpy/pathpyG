from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from pathpyG.visualisations.network_plot import NetworkPlot

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.temporal_graph import TemporalGraph


class TemporalNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "temporal"
    network: TemporalGraph

    def __init__(self, network: TemporalGraph, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__(network, **kwargs)

    def generate(self) -> None:
        """Generate the plot."""
        self._compute_edge_data()
        self._compute_node_data()
        # self._compute_layout()
        self._compute_config()

    def _compute_node_data(self) -> None:
        """Generate the data structure for the nodes."""
        # initialize values with index `node-0` to indicate time step 0
        start_nodes: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_tuples([(node, 0) for node in self.network.nodes], names=["uid", "time"]))
        new_nodes: pd.DataFrame = pd.DataFrame()
        # add attributes to start nodes and new nodes if given as dictionary
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("node").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                start_nodes[attribute] = [self.config.get("node").get(attribute, None)] * len(start_nodes)  # type: ignore[union-attr]
            else:
                start_nodes[attribute] = self.config.get("node").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as argument
            if attribute in self.node_args:
                if isinstance(self.node_args[attribute], dict):
                    # check if dict contains node or node-time keys
                    if "-" in next(iter(self.node_args[attribute].keys())):  # type: ignore[union-attr]
                        # add node attribute according to node-time keys
                        new_nodes = new_nodes.join(
                            pd.DataFrame.from_dict(self.node_args[attribute], orient="index", columns=[attribute]),
                            how="outer",
                        )
                    else:
                        # add node attributes to start nodes according to node keys
                        start_nodes[attribute] = start_nodes.index.get_level_values("uid").map(self.node_args[attribute])
                else:
                    start_nodes[attribute] = self.node_args[attribute]
            # check if attribute is given as node attribute
            elif f"node_{attribute}" in self.network.node_attrs():
                start_nodes[attribute] = self.network.data[f"node_{attribute}"]

        # combine start nodes and new nodes
        new_nodes = new_nodes.set_index(new_nodes.index.map(lambda x: (x.split("-")[0], int(x.split("-")[1]))))
        new_nodes.index.set_names(["uid", "time"], inplace=True)
        nodes = pd.concat([start_nodes, new_nodes])
        # fill missing values with last known value
        nodes = nodes.sort_values(by=["uid", "time"]).groupby("uid", sort=False).ffill()
        nodes["start"] = nodes.index.get_level_values("time")
        nodes = nodes.droplevel("time")
        # add end time step with the start the node appears the next time or max time step + 1
        nodes["end"] = nodes.groupby("uid")["start"].shift(-1)
        max_node_time = nodes["start"].max() + 1
        if max_node_time < self.network.data.time[-1].item():
            max_node_time = self.network.data.time[-1].item() + 1
        nodes["end"] = nodes["end"].fillna(max_node_time)

        # convert attributes to useful values
        nodes["color"] = self._convert_to_rgb_tuple(nodes["color"])
        nodes["color"] = nodes["color"].map(self._convert_color)

        # save node data
        self.data["nodes"] = nodes

    def _compute_edge_data(self) -> None:
        """Generate the data structure for the edges."""
        # initialize values
        edges: pd.DataFrame = pd.DataFrame(index=pd.MultiIndex.from_tuples(self.network.temporal_edges, names=["source", "target", "time"]))
        for attribute in self.attributes:
            # set default value for each attribute based on the pathpyG.toml config
            if isinstance(self.config.get("edge").get(attribute, None), list | tuple):  # type: ignore[union-attr]
                edges[attribute] = [self.config.get("edge").get(attribute, None)] * len(edges)  # type: ignore[union-attr]
            else:
                edges[attribute] = self.config.get("edge").get(attribute, None)  # type: ignore[union-attr]
            # check if attribute is given as argument
            if attribute in self.edge_args:
                if isinstance(self.edge_args[attribute], dict):
                    # if dict does not contain values for all edges, only update those that are given
                    new_attrs = edges.index.map(lambda x: f"{x[0]}-{x[1]}-{x[2]}").map(self.edge_args[attribute])
                    edges.loc[~new_attrs.isna(), attribute] = new_attrs[~new_attrs.isna()]
                else:
                    edges[attribute] = self.edge_args[attribute]
            # check if attribute is given as edge attribute
            elif f"edge_{attribute}" in self.network.edge_attrs():
                edges[attribute] = self.network.data[f"edge_{attribute}"]
            # special case for size: If no edge_size is given use edge_weight if available
            elif attribute == "size":
                if "edge_weight" in self.network.edge_attrs():
                    edges[attribute] = self.network.data["edge_weight"]
                elif "weight" in self.edge_args:
                    if isinstance(self.edge_args["weight"], dict):
                        new_attrs = edges.index.map(lambda x: f"{x[0]}-{x[1]}-{x[2]}").map(
                            self.edge_args["weight"]
                        )
                        edges.loc[~new_attrs.isna(), attribute] = new_attrs[~new_attrs.isna()]
                    else:
                        edges[attribute] = self.edge_args["weight"]


        # convert needed attributes to useful values
        edges["color"] = self._convert_to_rgb_tuple(edges["color"])
        edges["color"] = edges["color"].map(self._convert_color)
        edges["start"] = edges.index.get_level_values("time").astype(int)
        edges["end"] = edges["start"] + 1  # assume all edges last for one time step
        edges.index = edges.index.droplevel("time")

        # save edge data
        self.data["edges"] = edges

    def _compute_config(self) -> None:
        """Add additional configs."""
        self.config["directed"] = True
        self.config["curved"] = False
        self.config["simulation"] = self.config["layout"] is None
