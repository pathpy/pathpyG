from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric import EdgeIndex

from pathpyG import Graph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.utils.config import config

class TemporalGraph(Graph):
    def __init__(self, data: Data, mapping: IndexMap = None) -> None:
        """Creates an instance of a temporal graph from a `TemporalData` object.
        
        
        Example:
            ```py
            from pytorch_geometric.data import TemporalData
            import pathpyG as pp

            d = Data(edge_index=[[0,0,1], [1,2,2]], time=[0,1,2])
            t = pp.TemporalGraph(d, mapping)
            print(t)
            ```
        """
        if not isinstance(data.edge_index, EdgeIndex):
            data.edge_index = data.edge_index = EdgeIndex(data=data.edge_index, sparse_size=(data.num_nodes, data.num_nodes))

        # reorder temporal data
        self.data = data.sort_by_time()

        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()

        # create mapping between edge index and edge tuples
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        self.start_time = self.data.time[0].item()
        self.end_time = self.data.time[-1].item()

    @staticmethod
    def from_edge_list(edge_list, num_nodes: Optional[int] = None) -> TemporalGraph:
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

        if not num_nodes:
            num_nodes = len(set(sources+targets))

        return TemporalGraph(
            data=Data(
                edge_index=torch.stack((torch.Tensor(sources), torch.Tensor(targets))).long(),
                time=torch.Tensor(ts),
                num_nodes=num_nodes
            ),
            mapping=index_map
        )

    @staticmethod
    def from_csv(filename: str, sep: str = '', header: bool = True, is_undirected: bool = False, timestamp_format='%Y-%m-%d %H:%M:%S', time_rescale=1) -> TemporalGraph:
        """Read temporal graph from csv file, using pandas module"""
        from pathpyG.io.pandas import read_csv_temporal_graph
        return read_csv_temporal_graph(filename, sep=sep, header=header, is_undirected=is_undirected, timestamp_format=timestamp_format, time_rescale=time_rescale)

    @property
    def temporal_edges(self) -> Generator[Tuple[int, int, int], None, None]:
        """Iterator that yields each edge as a tuple of source and destination node as well as the corresponding timestamp."""
        i = 0
        for e in self.data.edge_index.t():
            yield self.mapping.to_id(e[0].item()), self.mapping.to_id(e[1].item()), self.data.time[i].item()  # type: ignore
            i += 1
    
    def shuffle_time(self) -> None:
        """Randomly shuffles the temporal order of edges by randomly permuting timestamps."""
        self.data.time = self.data.time[torch.randperm(len(self.data.time))]

    def to_static_graph(self, weighted=False, time_window: Optional[Tuple[int,int]]=None) -> Graph:
        """Return weighted time-aggregated instance of [`Graph`][pathpyG.Graph] graph.
        """
        if time_window is not None:
            idx = (self.data.time >= time_window[0]).logical_and(self.data.time < time_window[1]).nonzero().ravel()
            edge_index = self.data.edge_index[:, idx]
        else:
            edge_index = self.data.edge_index

        n = edge_index.max().item()+1

        if weighted:
            i, w = torch_geometric.utils.coalesce(edge_index.as_tensor(), torch.ones(edge_index.size(1), device=self.data.edge_index.device))
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
        times = torch.cat([self.data.time, self.data.time])
        return TemporalGraph(
            data=Data(
                edge_index=edge_index,
                time=times
            ),
            mapping=self.mapping
        )        

    def get_window(self, start: int, end: int) -> TemporalGraph:
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given window defined by start and (non-inclusive) end, where start
        and end refer to the index of the first and last event in the time-ordered list of events."""

        return TemporalGraph(
            data=Data(
                edge_index=self.data.edge_index[:, start:end],
                time=self.data.time[start:end]
            ),
            mapping=self.mapping
        )

    def get_snapshot(self, start: int, end: int) -> TemporalGraph:
        """Returns an instance of the TemporalGraph that captures all time-stamped 
        edges in a given time window defined by start and (non-inclusive) end, where start
        and end refer to the time stamps"""

        return TemporalGraph(
            data=self.data.snapshot(start, end),
            mapping=self.mapping
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the graph
        """
        s = "Temporal Graph with {0} nodes, {1} unique edges and {2} events in [{3}, {4}]\n".format(
            self.data.num_nodes,
            self.data.edge_index.unique(dim=1).size(dim=1),
            self.data.edge_index.size(1),
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
                s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.keys()) > len(self.data.edge_attrs()) + len(
            self.data.node_attrs()
        ):
            s += "\nGraph attributes\n"
            for a in self.data.keys():
                if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a) and a != 'src' and a != 'dst' and a != 't':
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        return s
