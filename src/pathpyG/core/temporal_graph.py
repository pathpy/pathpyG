from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import numpy as np

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric import EdgeIndex

from pathpyG import Graph
from pathpyG.core.index_map import IndexMap


class TemporalGraph(Graph):
    def __init__(self, data: Data, mapping: IndexMap | None = None) -> None:
        """Creates an instance of a temporal graph from a `TemporalData` object.

        Args:
            data: xxx
            mapping: xxx

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
            data.edge_index = data.edge_index = EdgeIndex(
                data=data.edge_index.contiguous(), sparse_size=(data.num_nodes, data.num_nodes)
            )

        # reorder temporal data
        self.data = data.sort_by_time()

        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()

        # create mapping between edge index and edge tuples
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        self.start_time = self.data.time[0].item()
        self.end_time = self.data.time[-1].item()

    @staticmethod
    def from_edge_list(edge_list, num_nodes: Optional[int] = None) -> TemporalGraph:
        edge_array = np.array(edge_list)
        ts = edge_array[:, 2].astype(np.number)

        index_map = IndexMap(np.unique(edge_array[:, :2]))
        edge_index = index_map.to_idxs(edge_array[:, :2].T)

        if not num_nodes:
            num_nodes = index_map.num_ids()

        return TemporalGraph(
            data=Data(
                edge_index=edge_index,
                time=torch.Tensor(ts),
                num_nodes=num_nodes,
            ),
            mapping=index_map,
        )

    @property
    def temporal_edges(self) -> Generator[Tuple[int, int, int], None, None]:
        """Iterator that yields each edge as a tuple of source and destination node as well as the corresponding timestamp."""
        return [(*self.mapping.to_ids(e), t.item()) for e, t in zip(self.data.edge_index.t(), self.data.time)]

    @property
    def order(self) -> int:
        """Return order 1, since all temporal graphs must be order one."""
        return 1

    def shuffle_time(self) -> None:
        """Randomly shuffle the temporal order of edges by randomly permuting timestamps."""
        self.data.time = self.data.time[torch.randperm(len(self.data.time))]

    def to_static_graph(self, weighted: bool = False, time_window: Optional[Tuple[int, int]] = None) -> Graph:
        """Return weighted time-aggregated instance of [`Graph`][pathpyG.Graph] graph.

        Args:
            weighted: whether or not to return a weighted time-aggregated graph
            time_window: A tuple with start and end time of the aggregation window

        Returns:
            Graph: A static graph object
        """
        if time_window is not None:
            idx = (self.data.time >= time_window[0]).logical_and(self.data.time < time_window[1]).nonzero().ravel()
            edge_index = self.data.edge_index[:, idx]
        else:
            edge_index = self.data.edge_index

        n = edge_index.max().item() + 1

        if weighted:
            i, w = torch_geometric.utils.coalesce(
                edge_index.as_tensor(), torch.ones(edge_index.size(1), device=self.data.edge_index.device)
            )
            return Graph(Data(edge_index=EdgeIndex(data=i, sparse_size=(n, n)), edge_weight=w), self.mapping)
        else:
            return Graph.from_edge_index(EdgeIndex(data=edge_index, sparse_size=(n, n)), self.mapping)

    def to_undirected(self) -> TemporalGraph:
        """Return an undirected version of a directed graph.

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
        return TemporalGraph(data=Data(edge_index=edge_index, time=times), mapping=self.mapping)

    def get_batch(self, start_idx: int, end_idx: int) -> TemporalGraph:
        """Return an instance of the TemporalGraph that captures all time-stamped
        edges in a given batch defined by start and (non-inclusive) end, where start
        and end refer to the index of the first and last event in the time-ordered list of events."""

        return TemporalGraph(
            data=Data(edge_index=self.data.edge_index[:, start_idx:end_idx], time=self.data.time[start_idx:end_idx]),
            mapping=self.mapping,
        )

    def get_window(self, start_time: int, end_time: int) -> TemporalGraph:
        """Return an instance of the TemporalGraph that captures all time-stamped
        edges in a given time window defined by start and (non-inclusive) end, where start
        and end refer to the time stamps"""

        return TemporalGraph(data=self.data.snapshot(start_time, end_time), mapping=self.mapping)

    def __str__(self) -> str:
        """
        Return a string representation of the graph
        """
        s = "Temporal Graph with {0} nodes, {1} unique edges and {2} events in [{3}, {4}]\n".format(
            self.data.num_nodes,
            self.data.edge_index.unique(dim=1).size(dim=1),
            self.data.edge_index.size(1),
            self.start_time,
            self.end_time,
        )

        attr = self.data.to_dict()
        attr_types = {}
        for k in attr:
            t = type(attr[k])
            if t == torch.Tensor:
                attr_types[k] = str(t) + " -> " + str(attr[k].size())
            else:
                attr_types[k] = str(t)

        from pprint import pformat

        attribute_info = {"Node Attributes": {}, "Edge Attributes": {}, "Graph Attributes": {}}
        for a in self.node_attrs():
            attribute_info["Node Attributes"][a] = attr_types[a]
        for a in self.edge_attrs():
            attribute_info["Edge Attributes"][a] = attr_types[a]
        for a in self.data.keys():
            if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                attribute_info["Graph Attributes"][a] = attr_types[a]
        s += pformat(attribute_info, indent=4, width=160)
        return s
