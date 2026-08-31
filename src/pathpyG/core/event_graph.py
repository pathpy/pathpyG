"""Event graph representation of a temporal graph and related operations."""
from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.temporal_graph import TemporalGraph

logger = logging.getLogger("root")


def _copy_attr(value: Any) -> Any:
    """Return an independent copy of an attribute value, leaving immutable values as they are."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


class EventGraph(Graph):
    """A directed acyclic graph whose nodes are time-stamped events."""

    # Attributes that are constructed explicitly when lifting a temporal graph and that
    # must therefore not be overwritten by propagated attributes.
    _RESERVED_ATTRS = frozenset({"edge_index", "num_nodes", "node_sequence", "node_time"})

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
    def build_edge_index(temporal_graph: TemporalGraph, delta: float | int = 1):
        """Build the event-graph edge index by lifting a temporal graph to second order.

        Each temporal edge of `temporal_graph` becomes an event (node); two events are
        connected when the second can continue the first within the time window `delta`.

        Args:
            temporal_graph: Temporal graph to lift.
            delta: Maximum time difference between events to consider them connected.

        Returns:
            ho_index: Edge index of the second-order temporal event graph.
        """
        # first-order edge index
        edge_index, timestamps = temporal_graph.data.edge_index, temporal_graph.data.time

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

    @staticmethod
    def lift_attrs(temporal_graph: TemporalGraph) -> dict[str, Any]:
        """Map the attributes of a temporal graph to attributes of the lifted event graph.

        Since every temporal edge becomes an event, edge attributes of the temporal graph
        become node attributes of the event graph and are renamed from `edge_x` to `node_x`
        accordingly. Node attributes refer to first-order nodes, which have no counterpart
        among the events, so they are dropped. Graph attributes are kept unchanged.

        Attributes that are constructed explicitly while lifting (`edge_index`, `num_nodes`,
        `node_sequence` and the timestamps in `time`) are excluded.

        Args:
            temporal_graph: Temporal graph whose attributes shall be lifted.

        Returns:
            dict: mapping from attribute names in the event graph to their values.
        """
        attrs: dict[str, Any] = {}
        for key in temporal_graph.data.keys():
            if key in EventGraph._RESERVED_ATTRS or key == "time":
                continue
            if key.startswith("node_"):
                # attributes of first-order nodes have no counterpart in the event graph
                continue
            if key.startswith("edge_"):
                event_key = "node_" + key[len("edge_") :]
                if event_key in EventGraph._RESERVED_ATTRS:
                    logger.error("Edge attribute %s cannot be lifted to reserved attribute %s", key, event_key)
                    raise ValueError(f"edge attribute '{key}' would be lifted to reserved attribute '{event_key}'")
                attrs[event_key] = _copy_attr(temporal_graph.data[key])
            else:
                # graph-level attributes are kept as they are
                attrs[key] = _copy_attr(temporal_graph.data[key])
        return attrs

    @classmethod
    def from_temporal_graph(cls, temporal_graph: TemporalGraph, delta: int = 1) -> "EventGraph":
        """Build an EventGraph from a temporal graph by lifting its edges into events.

        Attributes of the temporal graph are propagated to the event graph as described in
        [`lift_attrs`][pathpyG.core.event_graph.EventGraph.lift_attrs].
        """
        ho_index = cls.build_edge_index(temporal_graph, delta)
        m = temporal_graph.data.time.size(0)  # number of events (== number of first-order edges)
        node_sequence = temporal_graph.data.edge_index.as_tensor().t().contiguous()  # [m, 2]
        node_time = temporal_graph.data.time.clone()  # [m]

        # Build an event mapping with IDs of the form "a->b@t" for each edge node
        event_ids = [
            f"{temporal_graph.mapping.to_id(u)}->{temporal_graph.mapping.to_id(v)}@{t}"
            for (u, v), t in zip(node_sequence.tolist(), node_time.tolist())
        ]
        mapping = IndexMap(event_ids)

        data = Data(
            edge_index=ho_index,
            num_nodes=m,
            node_sequence=node_sequence,
            node_time=node_time,
        )
        for key, value in cls.lift_attrs(temporal_graph).items():
            data[key] = value

        event_graph = cls(data, delta=delta, first_order_mapping=temporal_graph.mapping, n_first_order=temporal_graph.n, mapping=mapping)

        # Attach a clone of the temporal graph since we already have it
        event_graph._temporal_graph = TemporalGraph(temporal_graph.data.clone(), mapping=temporal_graph.mapping)

        return event_graph

    def _summary(self) -> str:
        """Return a one-line summary of the event graph."""
        return "Event Graph (delta={0}) with {1} first-order nodes, {2} events and {3} edges in [{4}, {5}]".format(
            self.delta,
            self.n_first_order,
            self.num_events,
            self.m,
            self.start_time,
            self.end_time,
        )

    def __len__(self):
        """Return the number of events in the graph."""
        return self.n

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

    @property
    def start_time(self) -> Union[int, float]:
        """Return the timestamp of the first event in the event graph."""
        return self.data.node_time.min().item()

    @property
    def end_time(self) -> Union[int, float]:
        """Return the timestamp of the last event in the event graph."""
        return self.data.node_time.max().item()

    def event_time(self, i: int) -> int:
        """Return the timestamp of the i-th event."""
        return self.data.node_time[i].item()

    def edge_delta_map(self) -> dict[tuple[int, int], int]:
        """Return a mapping from each transition edge (src, dst) to its time delta."""
        return {
            tuple(c): d
            for c, d in zip(
                self.data.edge_index.as_tensor().t().tolist(),
                self.data.edge_delta.tolist(),
            )
        }

    def shortest_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return first-order shortest-path distances and predecessors respecting delta."""
        from pathpyG.algorithms.temporal import temporal_shortest_paths

        return temporal_shortest_paths(temporal_graph=None, delta=self.delta, event_graph=self)

    def reduce_delta(self, decrement: int = 1) -> "EventGraph":
        """Return a new EventGraph with a reduced time window `delta - decrement`.

        The events are unchanged, so node and graph attributes are carried over as they are,
        while edge attributes are restricted to the edges that remain for the smaller `delta`.
        """
        new_delta = self.delta - decrement
        if new_delta < 0:
            raise ValueError(
                f"decrement={decrement} exceeds current delta={self.delta}"
            )

        ei = self.data.edge_index
        edge_delta = self.data.node_time[ei[1]] - self.data.node_time[ei[0]]
        mask = edge_delta <= new_delta
        new_edge_index = ei[:, mask].contiguous()

        data = Data(
            edge_index=new_edge_index,
            num_nodes=self.n,
            node_sequence=self.data.node_sequence.clone(),
            node_time=self.data.node_time.clone(),
        )
        for key in self.data.keys():
            if key in self._RESERVED_ATTRS:
                continue
            if key.startswith("edge_"):
                data[key] = self.data[key][mask]
            else:
                data[key] = _copy_attr(self.data[key])

        return EventGraph(
            data,
            delta=new_delta,
            first_order_mapping=self.first_order_mapping,
            n_first_order=self.n_first_order,
            mapping=self.mapping,
        )