"""Event graph representation of a temporal graph and related operations."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.temporal_graph import TemporalGraph


class EventGraph(Graph):
    """A directed acyclic graph whose nodes are time-stamped events."""

    def __init__(
        self,
        data: Data,
        delta: int,
        first_order_mapping: IndexMap | None = None,
        n_first_order: int | None = None,
        mapping: IndexMap | None = None,
    ) -> None:
        """Create an EventGraph from a `Data` object carrying per-event `node_time`."""
        if "node_time" not in data:
            raise ValueError("EventGraph requires a per-event `node_time` node attribute.")

        super().__init__(data, mapping=mapping)

        self.delta = delta
        self.first_order_mapping = first_order_mapping if first_order_mapping is not None else IndexMap()
        if n_first_order is not None:
            self._n_first_order = int(n_first_order)
        else:
            self._n_first_order = int(self.data.node_sequence.max().item()) + 1

        ei = self.data.edge_index
        self.data.edge_delta = self.data.node_time[ei[1]] - self.data.node_time[ei[0]]

        self._temporal_graph: TemporalGraph | None = None

    @staticmethod
    def build_edge_index(g: TemporalGraph, delta: float | int = 1):
        """Build the event-graph edge index by lifting a temporal graph to second order.

        Each temporal edge of `g` becomes an event (node); two events are connected
        when the second can continue the first within the time window `delta`.

        Args:
            g: Temporal graph to lift.
            delta: Maximum time difference between events to consider them connected.

        Returns:
            ho_index: Edge index of the second-order temporal event graph.
        """
        # first-order edge index
        edge_index, timestamps = g.data.edge_index, g.data.time

        delta = torch.tensor(delta, device=edge_index.device)  # type: ignore[assignment]
        indices = torch.arange(0, edge_index.size(1), device=edge_index.device)

        unique_t = torch.unique(timestamps, sorted=True)
        second_order = []

        # lift order: find possible continuations for edges in each time stamp
        for t in tqdm(unique_t):
            # find indices of all source edges that occur at unique timestamp t
            src_time_mask = timestamps == t
            src_edge_idx = indices[src_time_mask]

            # find indices of all edges that can possibly continue edges occurring at time t for the given delta
            dst_time_mask = (timestamps > t) & (timestamps <= t + delta)
            dst_edge_idx = indices[dst_time_mask]

            if dst_edge_idx.size(0) > 0 and src_edge_idx.size(0) > 0:
                # compute second-order edges between src and dst idx
                # for all edges where dst in src_edges (edge_index[1, x[:, 0]]) matches src in dst_edges (edge_index[0, x[:, 1]])
                x = torch.cartesian_prod(src_edge_idx, dst_edge_idx)
                ho_edge_index = x[edge_index[1, x[:, 0]] == edge_index[0, x[:, 1]]]
                second_order.append(ho_edge_index)

        ho_index = torch.cat(second_order, dim=0).t().contiguous()
        return ho_index

    @classmethod
    def from_temporal_graph(cls, g: TemporalGraph, delta: int = 1) -> "EventGraph":
        """Build an EventGraph from a temporal graph by lifting its edges into events."""
        ho_index = cls.build_edge_index(g, delta)
        m = g.data.time.size(0)  # number of events (== number of first-order edges)
        node_sequence = g.data.edge_index.as_tensor().t().contiguous()  # [m, 2]
        node_time = g.data.time.clone()  # [m]

        # Build an event mapping with IDs of the form "a->b@t" for each edge node
        event_ids = [
            f"{g.mapping.to_id(u)}->{g.mapping.to_id(v)}@{t}"
            for (u, v), t in zip(node_sequence.tolist(), node_time.tolist())
        ]
        mapping = IndexMap(event_ids)

        data = Data(
            edge_index=ho_index,
            num_nodes=m,
            node_sequence=node_sequence,
            node_time=node_time,
        )
        eg = cls(data, delta=delta, first_order_mapping=g.mapping, n_first_order=g.n, mapping=mapping)

        # Attach a clone of the temporal graph since we already have it
        eg._temporal_graph = TemporalGraph(g.data.clone(), mapping=g.mapping)

        return eg

    def __str__(self) -> str:
        """Return a human-readable summary listing the delta and all events."""
        return (
            f"EventGraph (delta={self.delta})\n" +
            "\n".join(f"{self.mapping.to_id(i)}" for i in range(self.n))
        )

    def __len__(self):
        """Return the number of events in the graph."""
        return self.n

    def __getitem__(self, key):
        """Return the (u, v, t) endpoints for an integer key, else delegate to `Graph`."""
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            u, v = self.data.node_sequence[key].tolist()
            return self.first_order_mapping.to_id(u), self.first_order_mapping.to_id(v), self.data.node_time[key].item()
        return super().__getitem__(key)

    def to(self, device: torch.device) -> "EventGraph":
        """Move the event graph and its underlying temporal graph to the given device."""
        super().to(device)
        if self._temporal_graph is not None:
            self._temporal_graph.to(device)
        return self

    def to_temporal_graph(self) -> TemporalGraph:
        """Return the underlying temporal graph, reconstructing it if necessary."""
        if self._temporal_graph is None:
            edge_index = self.data.node_sequence.t().contiguous()  # [2, num_events]
            self._temporal_graph = TemporalGraph(
                Data(
                    edge_index=edge_index,
                    time=self.data.node_time.clone(),
                    num_nodes=self.n_first_order,
                ),
                mapping=self.first_order_mapping,
            )
        return self._temporal_graph

    @property
    def n_first_order(self) -> int:
        """Number of distinct first-order nodes underlying the events."""
        return self._n_first_order

    @property
    def num_events(self) -> int:
        """Number of events (nodes) in the event graph."""
        return self.n

    def event_time(self, i: int) -> int:
        """Return the timestamp of the i-th event."""
        return self.data.node_time[i].item()

    def shortest_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return first-order shortest-path distances and predecessors respecting delta."""
        from pathpyG.algorithms.temporal import temporal_shortest_paths

        return temporal_shortest_paths(g=None, delta=self.delta, eg=self)