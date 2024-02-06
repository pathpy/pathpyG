from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import TemporalData
from torch import IntTensor

import datetime
from time import mktime

from pathpyG import Graph
from pathpyG.utils.config import config


class TemporalGraph(Graph):
    def __init__(self, edge_index, t, node_id=[], **kwargs):
        """Creates an instance of a temporal graph with given edge index and timestamps of edges"""

        assert len(node_id) == len(set(node_id)), "node_id entries must be unique"

        # sort edges by timestamp and reorder edge_index accordingly
        t_sorted, indices = torch.sort(torch.tensor(t).to(config["torch"]["device"]))

        if len(node_id) == 0:
            self.data = TemporalData(
                src=edge_index[0][indices],
                dst=edge_index[1][indices],
                t=t_sorted,
                node_id=[],
                **kwargs,
            )
        else:
            self.data = TemporalData(
                src=edge_index[0][indices],
                dst=edge_index[1][indices],
                t=t_sorted,
                node_id=node_id,
                num_nodes=len(node_id),
                **kwargs,
            )

        self.data["edge_index"] = edge_index[:, indices]

        # create mappings between node ids and node indices
        self.node_index_to_id = dict(enumerate(node_id))
        self.node_id_to_index = {v: i for i, v in enumerate(node_id)}

        # create mapping between edge index and edge tuples
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in edge_index.t()])
        }

        self.start_time = t_sorted.min()
        self.end_time = t_sorted.max()

        # initialize adjacency matrix
        self._sparse_adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(
            self.data.edge_index
        ).tocsr()

    @staticmethod
    def from_edge_list(edge_list):
        sources = []
        targets = []
        ts = []

        nodes_index = dict()
        index_nodes = dict()

        n = 0
        for v, w, t in edge_list:
            if v not in nodes_index:
                nodes_index[v] = n
                index_nodes[n] = v
                n += 1
            if w not in nodes_index:
                nodes_index[w] = n
                index_nodes[n] = w
                n += 1
            sources.append(nodes_index[v])
            targets.append(nodes_index[w])
            ts.append(t)

        return TemporalGraph(
            edge_index=torch.LongTensor([sources, targets]).to(
                config["torch"]["device"]
            ),
            t=ts,
            node_id=[index_nodes[i] for i in range(n)],
        )

    @staticmethod
    def from_csv(file, timestamp_format='%Y-%m-%d %H:%M:%S', time_rescale=1) -> TemporalGraph:
        tedges = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split(",")
                timestamp = fields[2]
                if timestamp.isdigit():
                    t = int(timestamp)
                else:
                    # if it is a string, we use the timestamp format to convert
                    # it to a UNIX timestamp
                    x = datetime.datetime.strptime(timestamp, timestamp_format)
                    t = int(mktime(x.timetuple()))
                tedges.append((fields[0], fields[1], int(t/time_rescale)))
        return TemporalGraph.from_edge_list(tedges)

    @property
    def temporal_edges(self):
        if len(self.node_index_to_id) > 0:
            i = 0
            for e in self.data.edge_index.t():
                yield self.node_index_to_id[e[0].item()], self.node_index_to_id[e[1].item()], self.data.t[i].item()  # type: ignore
                i += 1
        else:
            i = 0
            for e in self.data.edge_index.t():
                yield e[0].item(), e[1].item(), self.data.t[i].item()  # type: ignore
                i += 1

    @staticmethod
    def from_pyg_data(d: TemporalData, node_id=[]):
        x = d.to_dict()

        del x["src"]
        del x["dst"]
        del x["t"]
        if "edge_index" in d:
            del x["edge_index"]
        if "node_index" in d:
            del x["node_id"]

        g = TemporalGraph(
            edge_index=torch.tensor([d["src"], d["dst"]]).to(config["torch"]["device"]),
            t=d["t"],
            node_id=node_id,
            **x,
        )

        return g
    
    def shuffle_time(self) -> None:
        """Randomly shuffles the temporal order of edges by randomly permuting the time stamps."""
        self.data['t'] = self.data['t'][torch.randperm(len(self.data['t']))]
        # t_sorted, indices = torch.sort(torch.tensor(t).to(config["torch"]["device"]))
        # self.data['src'] = self.data['src']
        # self.data['dst'] = self.data['dst']
        # self.data['t'] = t_sorted

    def to_static_graph(self) -> Graph:
        """Return instance of [`Graph`][pathpyG.Graph] that represents the static, time-aggregated network.
        """
        node_id = [self.node_index_to_id[i] for i in range(self.N)]
        return Graph(self.data.edge_index, node_id)

    def to_pyg_data(self) -> TemporalData:
        """
        Returns an instance of [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) containing the
        `edge_index` as well as node, edge, and graph attributes
        """
        return self.data


    def get_window(self, start, end):
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given window defined by start and (non-inclusive) end, where start
        and end refer to the number of events"""

        idx = torch.tensor([self.data['src'][start:end].numpy(), self.data['dst'][start:end].numpy()]).to(config["torch"]["device"])
        max_idx = torch.max(idx).item()

        return TemporalGraph(
            edge_index = idx,
            t = self.data.t[start:end],
            node_id = self.data.node_id[:max_idx+1]
        )


    def __str__(self):
        """
        Returns a string representation of the graph
        """
        s = "Temporal Graph with {0} nodes {1} edges and {2} time-stamped events in [{3}, {4}]\n".format(
            self.data.num_nodes,
            self.data["edge_index"].unique(dim=1).size(dim=1),
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
