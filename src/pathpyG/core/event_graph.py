from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from pathpyG.algorithms.temporal import lift_order_temporal, temporal_shortest_paths
from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.temporal_graph import TemporalGraph


class EventGraph(Graph):
    def __init__(
        self,
        data: Data,
        delta: Union[int, float],
        fo_mapping: IndexMap | None = None,
        num_fo_nodes: int | None = None,
        mapping: IndexMap | None = None,
    ) -> None:

        if "node_time" not in data:
            raise ValueError("EventGraph requires a per-event `node_time` node attribute.")

        super().__init__(data, mapping=mapping)

        self.delta = delta
        self.fo_mapping = fo_mapping if fo_mapping is not None else IndexMap()
        if num_fo_nodes is not None:
            self._num_fo_nodes = int(num_fo_nodes)
        else:
            self._num_fo_nodes = int(self.data.node_sequence.max().item()) + 1

        ei = self.data.edge_index
        self.data.edge_delta = self.data.node_time[ei[1]] - self.data.node_time[ei[0]]

        self._temporal_graph: TemporalGraph | None = None

    @classmethod
    def from_temporal_graph(cls, g: TemporalGraph, delta: Union[int, float] = 1) -> "EventGraph":
        ho_index = lift_order_temporal(g, delta)
        m = g.data.time.size(0)  # number of events (== number of first-order edges)
        node_sequence = g.data.edge_index.as_tensor().t().contiguous()  # [m, 2]
        node_time = g.data.time.clone()  # [m]

        data = Data(
            edge_index=ho_index,
            num_nodes=m,
            node_sequence=node_sequence,
            node_time=node_time,
        )
        eg = cls(data, delta=delta, fo_mapping=g.mapping, num_fo_nodes=g.n)

        # Attach a clone of the temporal graph since we already have it
        eg._temporal_graph = TemporalGraph(g.data.clone(), mapping=g.mapping)

        return eg

    def __str__(self) -> str:
        events_str = ""
        for i in range(self.n):
            u_id, v_id, t = self.event_endpoints(i)
            events_str += f"\n{u_id}->{v_id}@{t}"
        return (
            f"EventGraph (delta={self.delta})"
            f"{events_str}"
        )

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self.event_endpoints(int(key))
        return super().__getitem__(key)

    def to(self, device: torch.device) -> "EventGraph":
        super().to(device)
        if self._temporal_graph is not None:
            self._temporal_graph.to(device)
        return self

    def to_temporal_graph(self) -> TemporalGraph:
        if self._temporal_graph is None:
            edge_index = self.data.node_sequence.t().contiguous()  # [2, num_events]
            self._temporal_graph = TemporalGraph(
                Data(
                    edge_index=edge_index,
                    time=self.data.node_time.clone(),
                    num_nodes=self.num_fo_nodes,
                ),
                mapping=self.fo_mapping,
            )
        return self._temporal_graph

    @property
    def num_fo_nodes(self) -> int:
        return self._num_fo_nodes

    @property
    def num_events(self) -> int:
        return self.n

    def event_time(self, i: int) -> Union[int, float]:
        return self.data.node_time[i].item()

    def event_endpoints(self, i: int) -> Tuple:
        u, v = self.data.node_sequence[i].tolist()
        return self.fo_mapping.to_id(u), self.fo_mapping.to_id(v), self.data.node_time[i].item()

    def continuations(self, i: int) -> list:
        out = []
        for nxt in self.get_successors(i):
            nxt = int(nxt.item())
            out.append((nxt, self.data.edge_delta[self.edge_to_index[(i, nxt)]].item()))
        return out

    def shortest_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: This is wasteful, since we already have the lifted edge index
        # Modify `temporal_shortest_paths` to take in an optional pre-computed
        # edge_index?
        return temporal_shortest_paths(self.to_temporal_graph(), self.delta)