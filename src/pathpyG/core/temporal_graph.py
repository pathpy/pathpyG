"""Temporal Graph class for handling time-stamped edges."""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric import EdgeIndex
from torch_geometric.data import Data

from pathpyG import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.utils import to_numpy


class TemporalGraph(Graph):
    """Class representing a temporal graph with time-stamped edges.

    Info:
        The [`data`][torch_geometric.data.Data] attribute is a PyG Data object that contains the following attributes:

        - `edge_index`: [Edge index][torch_geometric.EdgeIndex] tensor of shape `(2, num_edges)` representing directed edges.
        - `time`: [Tensor][torch.Tensor] of shape `(num_edges,)` containing timestamps for each edge.

    Attributes:
        data (Data): PyG Data object containing temporal edges and attributes.
        mapping (IndexMap): Mapping from node IDs to indices.
        edge_to_index (dict): Mapping from edge tuples to their indices.
        tedge_to_index (dict): Mapping from temporal edge tuples to their indices.
    """

    def __init__(self, data: Data, mapping: IndexMap | None = None) -> None:
        """Creates an instance of a temporal graph from a `TemporalData` object.

        Args:
            data: PyG `Data` object containing edges saved in `edge_index` and timestamps in `time`.
            mapping: Optional mapping from node IDs to indices.

        Example:
            ```py
            from pytorch_geometric.data import TemporalData
            import pathpyG as pp

            d = Data(edge_index=[[0, 0, 1], [1, 2, 2]], time=[0, 1, 2])
            t = pp.TemporalGraph(d, mapping)
            print(t)
            ```
        """
        self.data = data
        if not isinstance(self.data.edge_index, EdgeIndex):
            self.data.edge_index = EdgeIndex(
                data=self.data.edge_index.contiguous(), sparse_size=(self.data.num_nodes, self.data.num_nodes)
            )

        # reorder temporal data
        # Note that we do not use `torch_geometric.self.data.Data.sort_by_time` because it cannot sort numpy arrays`
        sorted_idx = torch.argsort(self.data.time)
        for edge_attr in set(self.data.edge_attrs()).union(set(["time"])):
            if edge_attr == "edge_index":
                self.data.edge_index = self.data.edge_index[:, sorted_idx]
            else:
                self.data[edge_attr] = self.data[edge_attr][sorted_idx]

        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = IndexMap()

        # create mapping between edge index and edge tuples
        self.edge_to_index = {(e[0].item(), e[1].item()): i for i, e in enumerate(self.data.edge_index.t())}
        self.tedge_to_index = {
            (e[0].item(), e[1].item(), t.item()): i
            for i, (e, t) in enumerate(zip([e for e in self.data.edge_index.t()], self.data.time))
        }

    @staticmethod
    def from_edge_list(  # type: ignore[override]
        edge_list, num_nodes: Optional[int] = None, device: Optional[torch.device] = None
    ) -> "TemporalGraph":
        """Create a temporal graph from a list of tuples containing edges with timestamps.

        Args:
            edge_list: A list of tuples in the format (source, destination, timestamp).
            num_nodes: Optional number of nodes in the graph. If not provided, it will be inferred.
            device: The device on which to create the tensors (CPU or GPU).

        Returns:
            TemporalGraph: An instance of the TemporalGraph class.

        Examples:
            Create a temporal graph from an edge list:

            >>> import pathpyG as pp
            >>> edge_list = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
            >>> g = pp.TemporalGraph.from_edge_list(edge_list)
        """
        if len(edge_list) == 0:
            return TemporalGraph(
                data=Data(
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                    time=torch.empty((0,), dtype=torch.long, device=device),
                    num_nodes=num_nodes,
                ),
            )

        edge_array = np.array(edge_list)

        # Convert timestamps to tensor
        if isinstance(edge_list[0][2], int):
            ts = torch.tensor(edge_array[:, 2].astype(np.int_), device=device)
        else:
            ts = torch.tensor(edge_array[:, 2].astype(np.double), device=device)

        index_map = IndexMap(np.unique(edge_array[:, :2]))
        edge_index = index_map.to_idxs(edge_array[:, :2].T, device=device)

        if not num_nodes:
            num_nodes = index_map.num_ids()

        return TemporalGraph(
            data=Data(
                edge_index=edge_index,
                time=ts,
                num_nodes=num_nodes,
            ),
            mapping=index_map,
        )

    @property
    def temporal_edges(self) -> list:
        """Return all temporal edges as a list of tuples (source, destination, timestamp).

        Returns:
            list: A list of tuples representing temporal edges in the format (source, destination, timestamp).

        Examples:
            Get the list of temporal edges:

            >>> import pathpyG as pp
            >>> g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "a", 3)])
            >>> print(g.temporal_edges)
            [('a', 'b', 1), ('b', 'c', 2), ('c', 'a', 3)]

            Iterate over temporal edges:
            >>> for edge in g.temporal_edges:
            ...     print(edge)
            ('a', 'b', 1)
            ('b', 'c', 2)
            ('c', 'a', 3)
        """
        edge_ids = self.mapping.to_ids(self.data.edge_index)
        if isinstance(edge_ids, torch.Tensor):
            edge_ids = to_numpy(edge_ids)
        edge_ids = edge_ids.tolist()
        times = to_numpy(self.data.time).tolist()
        return list(zip(edge_ids[0], edge_ids[1], times))

    def to(self, device: torch.device) -> "TemporalGraph":
        """Moves all graph data to the specified device (CPU or GPU).

        Args:
            device: The target device to move the graph data to.

        Returns:
            TemporalGraph: A new TemporalGraph instance with data on the specified device.
        """
        self.data.edge_index = self.data.edge_index.to(device)
        self.data.time = self.data.time.to(device)
        for attr in self.node_attrs():
            if isinstance(self.data[attr], torch.Tensor):
                self.data[attr] = self.data[attr].to(device)
        for attr in self.edge_attrs():
            if isinstance(self.data[attr], torch.Tensor):
                self.data[attr] = self.data[attr].to(device)
        return self

    @property
    def order(self) -> int:
        """Return order 1, since all temporal graphs must be order one."""
        return 1

    @property
    def start_time(self) -> Union[int, float]:
        """Return the timestamp of the first event in the temporal graph."""
        return self.data.time.min().item()

    @property
    def end_time(self) -> Union[int, float]:
        """Return the timestamp of the last event in the temporal graph."""
        return self.data.time.max().item()

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

    def to_undirected(self) -> "TemporalGraph":
        """Return an undirected version of a directed graph.

        This method transforms the current graph instance into an undirected graph by
        adding all directed edges in opposite direction.

        Warning:
            This method duplicates all temporal edges in the graph, which can lead to duplicated
            edges if the original graph already contains bidirectional edges. As of now, edge attributes will
            **not** be duplicated for the new edges.

        Example:
            ```py
            import pathpyG as pp

            g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "a", 3)])
            g_u = g.to_undirected()
            print(g_u)
            ```
        """
        # TODO: Handle edge attributes for new edges
        rev_edge_index = self.data.edge_index.flip([0])
        edge_index = torch.cat([self.data.edge_index, rev_edge_index], dim=1)
        times = torch.cat([self.data.time, self.data.time])
        return TemporalGraph(data=Data(edge_index=edge_index, time=times), mapping=self.mapping)

    def get_batch(self, start_idx: int, end_idx: int) -> "TemporalGraph":
        """Return a batch of temporal edges based on start and end indices.

        Return an instance of the TemporalGraph that captures all time-stamped
        edges in a given batch defined by start and (non-inclusive) end, where start
        and end refer to the index of the first and last event in the time-ordered list of events.

        Args:
            start_idx: The starting index of the batch (inclusive).
            end_idx: The ending index of the batch (exclusive).

        Examples:
            Get a batch of temporal edges:

            >>> import pathpyG as pp
            >>> g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "a", 3)])
            >>> batch = g.get_batch(0, 2)
            >>> print(batch.temporal_edges)
            [('a', 'b', 1), ('b', 'c', 2)]
        """
        # Create new Data object with the selected batch of edges and times
        data = Data(edge_index=self.data.edge_index[:, start_idx:end_idx], time=self.data.time[start_idx:end_idx])

        # Copy all node attributes
        for node_attr in self.node_attrs():
            data[node_attr] = self.data[node_attr]
        # Copy only edge attributes for the selected batch
        for edge_attr in self.edge_attrs():
            data[edge_attr] = self.data[edge_attr][start_idx:end_idx]

        return TemporalGraph(
            data=data,
            mapping=self.mapping,
        )

    def get_window(self, start_time: int, end_time: int) -> "TemporalGraph":
        """Return a time window of temporal edges based on start and end timestamps.

        Return an instance of the TemporalGraph that captures all time-stamped
        edges in a given time window defined by start and (non-inclusive) end, where start
        and end refer to the time stamps.

        Args:
            start_time: The starting timestamp of the window (inclusive).
            end_time: The ending timestamp of the window (exclusive).

        Examples:
            Get a time window of temporal edges:

            >>> import pathpyG as pp
            >>> g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "a", 3)])
            >>> window = g.get_window(0, 2)
            >>> print(window.temporal_edges)
            [('a', 'b', 1)]
        """
        # While there is a PyG function `Data.snapshot`,
        # we do it manually since it cannot handle numpy arrays as edge attributes.
        edge_mask = (self.data.time >= start_time).logical_and(self.data.time < end_time)
        # Create a new Data object with the selected edges and times
        data = Data(
            edge_index=self.data.edge_index[:, edge_mask],
            time=self.data.time[edge_mask],
        )
        # Copy all node attributes
        for node_attr in self.node_attrs():
            data[node_attr] = self.data[node_attr]
        # Copy only edge attributes for the selected edges
        for edge_attr in self.edge_attrs():
            data[edge_attr] = self.data[edge_attr][edge_mask]

        return TemporalGraph(data=data, mapping=self.mapping)

    def __getitem__(self, key: Union[tuple, str]) -> Any:
        """Return node, edge, temporal edge, or graph attribute.

        Args:
            key: name of attribute to be returned
        """
        if not isinstance(key, tuple):
            if key in self.data.keys():
                return self.data[key]
            else:
                raise KeyError(key + " is not a graph attribute")
        elif key[0] in self.node_attrs():
            return self.data[key[0]][self.mapping.to_idx(key[1])]
        elif key[0] in self.edge_attrs():
            # TODO: Get item for non-temporal edges will only return the last occurence of the edge
            #       This is a limitation and should be fixed in the future.
            if len(key) == 3:
                return self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]]
            else:
                return self.data[key[0]][
                    self.tedge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2]), key[3]]
                ]
        else:
            raise KeyError(key[0] + " is not a node or edge attribute")

    def __str__(self) -> str:
        """Return a string representation of the graph."""
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

        attribute_info: dict[str, dict[str, Any]] = {
            "Node Attributes": {},
            "Edge Attributes": {},
            "Graph Attributes": {},
        }
        for a in self.node_attrs():
            attribute_info["Node Attributes"][a] = attr_types[a]
        for a in self.edge_attrs():
            attribute_info["Edge Attributes"][a] = attr_types[a]
        for a in self.data.keys():
            if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                attribute_info["Graph Attributes"][a] = attr_types[a]
        s += pformat(attribute_info, indent=4, width=160)
        return s
