from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.utils import Colormap, rgb_to_hex

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

    def _get_edge_data(self, edges: dict, attributes: set, attr: defaultdict, categories: set) -> None:
        """Extract edge data from temporal network."""
        for u, v, t in self.network.temporal_edges:
            uid = f"{u}-{v}-{t}"
            edges[uid] = {
                "uid": uid,
                "source": str(u),
                "target": str(v),
                "start": int(t),
                "end": int(t) + 1,
            }
            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"edge_{attribute}", u, v].item() if attribute in categories else None
                )

    def _compute_node_data(self):
        """_summary_"""
        super()._compute_node_data()

        raw_color_attr = self.config.get("node_color", {})
        if not isinstance(raw_color_attr, dict):
            return

        color_changes_by_node = defaultdict(list)
        for key, color in raw_color_attr.items():
            if "-" not in key:
                continue

            try:
                node_id, time_str = key.rsplit("-", 1)
                time = float(time_str)
            except ValueError as exc:
                raise ValueError(f"Invalid time-encoded node_color key: '{key}'") from exc

            if isinstance(color, (int, float)):
                cmap = self.config.get("node_cmap", Colormap())
                rgb = cmap([color])[0]
                color = rgb_to_hex(rgb[:3])

            elif isinstance(color, tuple):
                color = rgb_to_hex(color)

            color_changes_by_node[node_id].append({"time": time, "color": color})

        for node_id, changes in color_changes_by_node.items():
            if node_id in self.data.get("nodes", {}):
                self.data["nodes"][node_id]["color_change"] = sorted(changes, key=lambda x: x["time"])

    def _get_node_data(self, nodes: dict, attributes: set, attr: defaultdict, categories: set) -> None:
        """Extract node data from temporal network."""

        time = {e[2] for e in self.network.temporal_edges}

        if self.config.get("end", None) is None:
            self.config["end"] = int(max(time) + 1)

        if self.config.get("start", None) is None:
            self.config["start"] = int(min(time) - 1)

        for uid in self.network.nodes:
            nodes[uid] = {
                "uid": str(uid),
                "start": int(min(time) - 1),
                "end": int(max(time) + 1),
            }

            # add edge attributes if needed
            for attribute in attributes:
                attr[attribute][uid] = (
                    self.network[f"node_{attribute}", uid].item() if attribute in categories else None
                )

    def _compute_config(self) -> None:
        """Add additional configs."""
        pass