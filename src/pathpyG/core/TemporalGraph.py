from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import TemporalData, Data
from torch_geometric import EdgeIndex
from torch import IntTensor

import datetime
from time import mktime

from pathpyG import Graph
from pathpyG.core.IndexMap import IndexMap


class TemporalGraph(Graph):
    def __init__(self, data: TemporalData, mapping: IndexMap = None) -> None:
        """Creates an instance of a temporal graph from a `TemporalData` object.
        
        
        Example:
            ```py
            from pytorch_geometric.data import TemporalData
            import pathpyG as pp

            d = TemporalData(src=[0,0,1], dst=[1,2,2], t=[0,1,2])
            t = pp.TemporalGraph(d, mapping)
            print(t)
            ```
        """

        # sort edges by timestamp
        # Note: function sort_by_time mentioned in pyG documentation does not exist
        t_sorted, sort_index = torch.sort(data.t)

        # reorder temporal data
        self.data = TemporalData(
            src=data.src[sort_index],
            dst=data.dst[sort_index],
            t=t_sorted
        ).to(data.edge_index.device)

        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()

        # create mapping between edge index and edge tuples
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        self.start_time = t_sorted.min().item()
        self.end_time = t_sorted.max().item()

        # # initialize adjacency matrix
        # self._sparse_adj_matrix = torch_geometric.utils.to_scipy_sparse_matrix(
        #     self.data.edge_index
        # ).tocsr()

    @staticmethod
    def from_edge_list(edge_list, device: Optional[torch.device] = None) -> TemporalGraph:
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
            data=TemporalData(
                        src=torch.Tensor(sources, device=device).long(),
                        dst=torch.Tensor(targets, device=device).long(),
                        t=torch.Tensor(ts, device=device)),
            mapping=index_map
        )

    @staticmethod
    def from_csv(file, timestamp_format='%Y-%m-%d %H:%M:%S', time_rescale=1, device: Optional[torch.device] = None) -> TemporalGraph:
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
        return TemporalGraph.from_edge_list(tedges, device=device)

    @property
    def temporal_edges(self) -> Generator[Tuple[int, int, int], None, None]:
        """Iterator that yields each edge as a tuple of source and destination node as well as the corresponding timestamp."""
        i = 0
        for e in self.data.edge_index.t():
            yield self.mapping.to_id(e[0].item()), self.mapping.to_id(e[1].item()), self.data.t[i].item()  # type: ignore
            i += 1

    def to(self, device: torch.device) -> Graph:
        """Moves all attributes to the given device. """
        self.data.to(device)
        return self
    
    def shuffle_time(self) -> None:
        """Randomly shuffles the temporal order of edges by randomly permuting timestamps."""
        self.data['t'] = self.data['t'][torch.randperm(len(self.data['t']))]
        # t_sorted, indices = torch.sort(torch.tensor(t).to(config["torch"]["device"]))
        # self.data['src'] = self.data['src']
        # self.data['dst'] = self.data['dst']
        # self.data['t'] = t_sorted

    def to_static_graph(self, weighted=False, time_window: Optional[Tuple[int,int]]=None) -> Graph:
        """Return weighted time-aggregated instance of [`Graph`][pathpyG.Graph] graph.
        """
        if time_window is not None:
            idx = (self.data.t >= time_window[0]).logical_and(self.data.t < time_window[1]).nonzero().ravel()
            edge_index = torch.stack((self.data.src[idx], self.data.dst[idx]))
        else:
            edge_index = torch.stack((self.data.src, self.data.dst))

        n = edge_index.max().item()+1

        if weighted:
            i, w = torch_geometric.utils.coalesce(edge_index, torch.ones(edge_index.size(1), device=self.data.edge_index.device))
            return Graph(Data(edge_index=EdgeIndex(data=i, sparse_size=(n,n)), edge_weight=w), self.mapping)
        else:
            return Graph.from_edge_index(EdgeIndex(data=edge_index, sparse_size=(n,n)), self.mapping)

    def to_undirected(self) -> TemporalGraph:
        """
        Returns an undirected version of a directed graph.

        This method transforms the current graph instance into an undirected graph by
        adding all directed edges in opposite direction. It applies [`ToUndirected`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.ToUndirected.html#torch_geometric.transforms.ToUndirected)
        transform to the underlying [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object, which automatically
        duplicates edge attributes for newly created directed edges.

        Example:
            ```py
            import pathpyG as pp
            g = pp.TemporalGraph.from_edge_list([('a', 'b', 1), ('b', 'c', 2), ('c', 'a', 3)])
            g_u = g.to_undirected()
            print(g_u)
            ```
        """        
        rev_edge_index = self.data.edge_index.flip([0])
        edge_index = torch.cat([self.data.edge_index, rev_edge_index], dim=1)
        times = torch.cat([self.data.t, self.data.t])
        return TemporalGraph(
            data=TemporalData(
                src=edge_index[0],
                dst=edge_index[1],
                t=times
            ),
            mapping=self.mapping
        )        

    def get_window(self, start: int, end: int) -> TemporalGraph:
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given window defined by start and (non-inclusive) end, where start
        and end refer to the number of events"""

        #idx = torch.tensor([self.data['src'][start:end].numpy(), self.data['dst'][start:end].numpy()]).to(config["torch"]["device"])
        #max_idx = torch.max(idx).item()

        # return TemporalGraph(
        #     edge_index = idx,
        #     t = self.data.t[start:end],
        #     node_id = self.data.node_id[:max_idx+1]
        #     )
        return TemporalGraph(
            data=TemporalData(
                src=self.data.src[start:end],
                dst=self.data.dst[start:end],
                t=self.data.t[start:end]
            ),
            mapping=self.mapping
        )
    

    def get_snapshot(self, start: int, end: int) -> TemporalGraph:
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given time window defined by start and (non-inclusive) end, where start
        and end refer to the time stamps"""

        #idx = torch.tensor([self.data['src'][start:end].numpy(), self.data['dst'][start:end].numpy()]).to(config["torch"]["device"])
        #max_idx = torch.max(idx).item()

        return TemporalGraph(
            data=TemporalData(
                src=self.data.src[start:end],
                dst=self.data.dst[start:end],
                t=self.data.t[start:end]
            ).to(self.data.edge_index.device),
            mapping=self.mapping
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
