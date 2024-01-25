from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import TemporalData
from torch import IntTensor

from pathpyG import Graph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.utils.config import config


class TemporalGraph(Graph):
    def __init__(self, data: TemporalData, mapping: IndexMap = None) -> None:
        """Creates an instance of a temporal graph with given edge index and timestamps"""

        # sort edges by timestamp
        t_sorted, sort_index = torch.sort(data.t)

        # reorder temporal data
        self.data = TemporalData(
            src=data.src[sort_index],
            dst=data.dst[sort_index],
            t=t_sorted
        )

        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()

        # create mapping between edge index and edge tuples
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        self.start_time = t_sorted.min()
        self.end_time = t_sorted.max()

        # initialize adjacency matrix
        self._sparse_adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(
            self.data.edge_index
        ).tocsr()

    @staticmethod
    def from_edge_list(edge_list) -> TemporalGraph:
        sources = []
        targets = []
        ts = []

        index_map = IndexMap()

        for v, w, t in edge_list:
            index_map.add_id(v)
            index_map.add_id(w)
            sources.append(index_map.to_idx(v))
            targets.append(index_map.to_idx(w))
            ts.append(t)

        return TemporalGraph(
            data = TemporalData(src=torch.Tensor(sources),
                         dst = torch.Tensor(targets),
                         t = torch.Tensor(ts)),
            mapping = index_map
        )

    @staticmethod
    def from_csv(file) -> TemporalGraph:
        tedges = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split(",")
                tedges.append((fields[0], fields[1], int(fields[2])))
        return TemporalGraph.from_edge_list(tedges)

    @property
    def temporal_edges(self):
        i = 0
        for e in self.data.edge_index.t():
            yield self.mapping.to_id(e[0].item()), self.mapping.to_id(e[1].item()), self.data.t[i].item()  # type: ignore
            i += 1
    
    def shuffle_time(self) -> None:
        """Randomly shuffles the temporal order of edges by randomly permuting timestamps."""
        self.data['t'] = self.data['t'][torch.randperm(len(self.data['t']))]
        # t_sorted, indices = torch.sort(torch.tensor(t).to(config["torch"]["device"]))
        # self.data['src'] = self.data['src']
        # self.data['dst'] = self.data['dst']
        # self.data['t'] = t_sorted

    def to_static_graph(self) -> Graph:
        """Return instance of [`Graph`][pathpyG.Graph] that represents the static, time-aggregated network.
        """
        return Graph(self.data.edge_index, self.mapping)

    def get_window(self, start: int, end: int) -> TemporalGraph:
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given window defined by start and (non-inclusive) end, where start
        and end refer to the number of events"""

        #idx = torch.tensor([self.data['src'][start:end].numpy(), self.data['dst'][start:end].numpy()]).to(config["torch"]["device"])
        #max_idx = torch.max(idx).item()

        return TemporalGraph(
            data = TemporalData(
                src=self.data.src[start:end],
                dst=self.data.dst[start:end],
                t=self.data.t[start:end]
            ),
            mapping = self.mapping
        )


    def __str__(self) -> str:
        """
        Returns a string representation of the graph
        """
        s = "Temporal Graph with {0} nodes, {1} unique edges and {2} events in [{3}, {4}]\n".format(
            self.data.num_nodes,
            self.data.edge_index.unique(dim=1).size(dim=1),
            self.data.num_events,
            self.start_time,
            self.end_time,
        )

        attr_types = Graph.attr_types(self.data.to_dict())

        if len(self.data.node_attrs()) > 0:
            s += "\nNode attributes\n"
            for a in self.data.node_attrs():
                s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.edge_attrs()) > 1:
            s += "\nEdge attributes\n"
            for a in self.data.edge_attrs():
                if a != "edge_index":
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.keys()) > len(self.data.edge_attrs()) + len(
            self.data.node_attrs()
        ):
            s += "\nGraph attributes\n"
            for a in self.data.keys():
                if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        return s
