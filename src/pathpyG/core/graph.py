from __future__ import annotations
from typing import (
    Dict,
    Iterable,
    Tuple,
    List,
    Union,
    Any,
    Optional,
)

import numpy as np

import torch

import torch_geometric
import torch_geometric.utils
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_undirected

from pathpyG.core.index_map import IndexMap


class Graph:
    """
    A graph object storing nodes, edges, and attributes.

    An object than be be used to store directed or undirected graphs with node
    and edge attributes. Data on nodes and edges are stored in an underlying instance of
    [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).
    """

    def __init__(self, data: Data, mapping: Optional[IndexMap] = None):
        """Generate graph instance from a pyG `Data` object.

        Generate a Graph instance from a `torch_geometric.Data` object that contains an EdgeIndex as well as
        optional node-, edge- or graph-level attributes. An optional mapping can be used to transparently map
        node indices to string identifiers.

        Args:
            data: A pyG Data object containing an EdgeIndex and additional attributes
            mapping: `IndexMap` object that maps node indices to string identifiers

        Example:
            ```py
            import pathpyG as pp
            from torch_geometric.data import Data
            from torch_geometric import EdgeIndex

            data = Data(edge_index=EdgeIndex([[1,1,2],[0,2,1]], sparse_size=(3,3)))
            g = pp.Graph(data)

            g = pp.Graph(data, mapping=pp.IndexMap(['a', 'b', 'c']))
            ```
        """
        if mapping is None:
            self.mapping = IndexMap()
        else:
            self.mapping = mapping

        # set num_nodes property
        if "num_nodes" not in data and "edge_index" in data:
            data.num_nodes = data.edge_index.max().item() + 1

        # turn edge index tensor into EdgeIndex object
        if not isinstance(data.edge_index, EdgeIndex):
            data.edge_index = EdgeIndex(data=data.edge_index, sparse_size=(data.num_nodes, data.num_nodes))

        if (
            data.edge_index.get_sparse_size(dim=0) != data.num_nodes
            or data.edge_index.get_sparse_size(dim=1) != data.num_nodes
        ):
            raise Exception("sparse size of EdgeIndex should match number of nodes!")

        self.data = data

        # sort EdgeIndex and validate
        data.edge_index, sorted_idx = data.edge_index.sort_by("row")
        for edge_attr in self.edge_attrs():
            data[edge_attr] = self.data[edge_attr][sorted_idx]

        data.edge_index.validate()

        # create mapping between edge tuples and edge indices
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

        ((self.row_ptr, self.col), _) = self.data.edge_index.get_csr()
        ((self.col_ptr, self.row), _) = self.data.edge_index.get_csc()

        # create node_sequence mapping for higher-order graphs
        if "node_sequence" not in self.data:
            self.data.node_sequence = torch.arange(data.num_nodes).reshape(-1, 1)

    @staticmethod
    def from_edge_index(edge_index: torch.Tensor, mapping: Optional[IndexMap] = None, num_nodes: int = None) -> Graph:
        """Construct a graph from a torch Tensor containing an edge index. An optional mapping can
        be used to transparently map node indices to string identifiers.

        Args:
            edge_index:  torch.Tensor or torch_geometric.EdgeIndex object containing an edge_index
            mapping: `IndexMap` object that maps node indices to string identifiers
            num_nodes: optional number of nodes (default: None). If None, the number of nodes will be
                inferred based on the maximum node index in the edge index, i.e. there will be no isolated nodes.

        Examples:
            You can create a graph from an edge index tensor as follows:

            >>> import torch
            >>> import pathpyG as pp
            >>> g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]))
            >>> print(g)
            Directed graph with 3 nodes and 3 edges ...

            You can also include a mapping of node IDs:

            >>> g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
            >>>                              mapping=pp.IndexMap(['a', 'b', 'c']))
            >>> print(g.mapping)
            a -> 0
            b -> 1
            c -> 2
        """

        if not num_nodes:
            d = Data(edge_index=edge_index)
        else:
            d = Data(edge_index=edge_index, num_nodes=num_nodes)
        return Graph(d, mapping=mapping)

    @staticmethod
    def from_edge_list(
        edge_list: Iterable[Tuple[str, str]],
        is_undirected: bool = False,
        mapping: Optional[IndexMap] = None,
        num_nodes: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Graph:
        """Generate a Graph based on an edge list.

        Edges can be given as string or integer tuples. If strings are used and no mapping is given,
        a mapping of node IDs to indices will be automatically created based on a lexicographic ordering of
        node IDs.

        Args:
            edge_list: Iterable of edges represented as tuples
            is_undirected: Whether the edge list contains all bidorectional edges
            mapping: optional mapping of string IDs to node indices
            num_nodes: optional number of nodes (useful in case not all nodes have incident edges)
            device: optional torch device where tensors shall be stored

        Examples:
            >>> import pathpyG as pp
            >>> l = [('a', 'b'), ('a', 'c'), ('b', 'c')]
            >>> g = pp.Graph.from_edge_list(l)
            >>> print(list(g.edges))
            [('a', 'b'), ('a', 'c'), ('b', 'c')]
        """

        # handle empty graph
        if len(edge_list) == 0:
            return Graph(
                Data(edge_index=torch.tensor([[], []], dtype=torch.int32, device=device), num_nodes=0),
                mapping=IndexMap(),
            )

        if mapping is None:
            edge_array = np.array(edge_list)
            node_ids = np.unique(edge_array)
            if np.issubdtype(node_ids.dtype, str) and np.char.isnumeric(node_ids).all():
                node_ids = np.sort(node_ids.astype(int)).astype(str)
            mapping = IndexMap(node_ids)

        if num_nodes is None:
            num_nodes = mapping.num_ids()

        edge_index = EdgeIndex(
            mapping.to_idxs(edge_list, device=device).T.contiguous(),
            sparse_size=(num_nodes, num_nodes),
            is_undirected=is_undirected,
        )
        return Graph(Data(edge_index=edge_index, num_nodes=num_nodes), mapping=mapping)

    def to_undirected(self) -> Graph:
        """Return an undirected version of this directed graph.

        This method creates a new undirected Graph from the current graph instance by
        adding all directed edges in opposite direction.

        Examples:
            >>> import pathpyG as pp
            >>> g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a')])
            >>> g_u = g.to_undirected()
            >>> print(g_u)
            Undirected graph with 3 nodes and 6 (directed) edges
        """
        # create undirected edge index by coalescing the directed edges and keep
        # track of the original edge index for the edge attributes
        attr_idx = torch.arange(self.data.num_edges, device=self.data.edge_index.device)
        edge_index, attr_idx = to_undirected(
            self.data.edge_index,
            edge_attr=attr_idx,
            num_nodes=self.data.num_nodes,
            reduce="min",
        )

        data = Data(
            edge_index=EdgeIndex(
                data=edge_index, sparse_size=(self.data.num_nodes, self.data.num_nodes), is_undirected=True
            ),
            num_nodes=self.data.num_nodes,
        )
        # Note that while the torch_geometric.transforms.ToUndirected function would do this automatically,
        # we do it manually since the transform cannot handle numpy arrays as edge attributes.
        # make sure to copy all node and (undirected) edge attributes
        for node_attr in self.node_attrs():
            data[node_attr] = self.data[node_attr]
        for edge_attr in self.edge_attrs():
            if edge_attr != "edge_index":
                data[edge_attr] = self.data[edge_attr][attr_idx]

        return Graph(data, self.mapping)

    def to_weighted_graph(self) -> Graph:
        """Coalesces multi-edges to single-edges with an additional weight attribute

        If the graph contains multiple edges between the same nodes, this method will coalesce
        them into a single edge with an additional weight attribute called `edge_weight` that
        contains the number of coalesced edges. The method returns a new graph instance with
        the coalesced edges.

        Returns:
            Graph: Graph with coalesced edges
        """
        i, w = torch_geometric.utils.coalesce(
            self.data.edge_index.as_tensor(), torch.ones(self.m, device=self.data.edge_index.device)
        )
        return Graph(Data(edge_index=i, edge_weight=w, num_nodes=self.data.num_nodes), mapping=self.mapping)

    def to(self, device: torch.device) -> Graph:
        """Move all tensors to the given device.
        
        Args:
            device: torch device to which all tensors shall be moved

        Returns:
            Graph: self
        """
        self.data.edge_index = self.data.edge_index.to(device)
        self.data.node_sequence = self.data.node_sequence.to(device)
        for attr in self.node_attrs():
            if isinstance(self.data[attr], torch.Tensor):
                self.data[attr] = self.data[attr].to(device)
        for attr in self.edge_attrs():
            if isinstance(self.data[attr], torch.Tensor):
                self.data[attr] = self.data[attr].to(device)

        self.row = self.row.to(device)
        self.row_ptr = self.row_ptr.to(device)
        self.col = self.col.to(device)
        self.col_ptr = self.col_ptr.to(device)

        return self

    def node_attrs(self) -> List[str]:
        """
        Return a list of node attributes.

        This method returns a list containing the names of all node-level attributes,
        ignoring the special `node_sequence` attribute.

        Returns:
            list: list of node attributes
        """
        attrs = []
        for k in self.data.keys():
            if k != "node_sequence" and k.startswith("node_"):
                attrs.append(k)
        return attrs

    def edge_attrs(self) -> List[str]:
        """
        Return a list of edge attributes.

        This method returns a list containing the names of all edge-level attributes,
        ignoring the special `edge_index` attribute.

        Returns:
            list: list of edge attributes
        """
        attrs = []
        for k in self.data.keys():
            if k != "edge_index" and k.startswith("edge_"):
                attrs.append(k)
        return attrs

    @property
    def nodes(self) -> list:
        """
        Return indices or IDs of all nodes in the graph.

        This method returns a list object that contains all nodes.
        If an IndexMap is used, nodes are returned as string IDs.
        If no IndexMap is used, nodes are returned as integer indices.

        Returns:
            list: list of all nodes using IDs or indices (if no mapping is used)
        """
        node_list = self.mapping.to_ids(np.arange(self.n)).tolist()
        if self.order > 1:
            return list(map(tuple, node_list))
        return node_list

    @property
    def edges(self) -> list:
        """Return all edges in the graph.

        This method returns a list object that contains all edges, where each
        edge is a tuple of two elements. If an IndexMap is used to map node
        indices to string IDs, edges are returned as tuples of string IDs.
        If no mapping is used, edges are returned as tuples of integer indices.

        Returns:
            list: list object yielding all edges using IDs or indices (if no mapping is used)
        """
        edge_list = self.mapping.to_ids(self.data.edge_index.t()).tolist()
        if self.order > 1:
            return [tuple(map(tuple, x)) for x in edge_list]
        return list(map(tuple, edge_list))

    def get_successors(self, row_idx: int) -> torch.Tensor:
        """Return a tensor containing the indices of all successor nodes for a given node identified by an index.

        Args:
            row_idx:   Index of node for which predecessors shall be returned.

        Returns:
            tensor: tensor containing indices of all successor nodes of the node indexed by `row_idx`
        """

        if row_idx + 1 < self.row_ptr.size(0):
            row_start = self.row_ptr[row_idx]
            row_end = self.row_ptr[row_idx + 1]
            return self.col[row_start:row_end]
        else:
            return torch.tensor([], device=self.data.edge_index.device)

    def get_predecessors(self, col_idx: int) -> torch.Tensor:
        """Return a tensor containing the indices of all predecessor nodes for a given node identified by an index.

        Args:
            col_idx:   Index of node for which predecessors shall be returned.

        Returns:
            tensor: tensor containing indices of all predecessor nodes of the node indexed by `col_idx`
        """
        if col_idx + 1 < self.col_ptr.size(0):
            col_start = self.col_ptr[col_idx]
            col_end = self.col_ptr[col_idx + 1]
            return self.row[col_start:col_end]
        else:
            return torch.tensor([], device=self.data.edge_index.device)

    def successors(self, node: Union[int, str] | tuple) -> list:
        """Return all successors of a given node.

        This method returns a generator object that yields all successors of a
        given node. If an IndexMap is used, successors are returned
        as string IDs. If no mapping is used, successors are returned as indices.

        Args:
            node:   Index or string ID of node for which successors shall be returned.

        Returns:
            list: list with all successors of the node identified
                by `node` using ID or index (if no mapping is used)
        """

        node_list = self.mapping.to_ids(self.get_successors(self.mapping.to_idx(node))).tolist()  # type: ignore

        if self.order > 1:
            return list(map(tuple, node_list))
        return node_list

    def predecessors(self, node: Union[str, int] | tuple) -> list:
        """Return the predecessors of a given node.

        This method returns a generator object that yields all predecessors of a
        given node. If a `node_id` mapping is used, predecessors will be returned
        as string IDs. If no mapping is used, predecessors are returned as indices.

        Args:
            node:   Index or string ID of node for which predecessors shall be returned.

        Returns:
            list: list with all predecessors of the node identified
                by `node` using ID or index (if no mapping is used)
        """
        node_list = self.mapping.to_ids(self.get_predecessors(self.mapping.to_idx(node))).tolist()  # type: ignore

        if self.order > 1:
            return list(map(tuple, node_list))
        return node_list

    def is_edge(self, v: Union[str, int], w: Union[str, int]) -> bool:
        """Return whether edge $(v,w)$ exists in the graph.

        If an index to ID mapping is used, nodes are assumed to be string IDs. If no
        mapping is used, nodes are assumed to be integer indices.

        Args:
            v: source node of edge as integer index or string ID
            w: target node of edge as integer index or string ID

        Returns:
            bool: True if edge exists, False otherwise
        """
        row = self.mapping.to_idx(v)
        row_start = self.row_ptr[row]
        row_end = self.row_ptr[row + 1]

        return self.mapping.to_idx(w) in self.col[row_start:row_end]

    def sparse_adj_matrix(self, edge_attr: Any = None) -> Any:
        """Return sparse adjacency matrix representation of (weighted) graph.

        Args:
            edge_attr: the edge attribute that shall be used as edge weight

        Returns:
            scipy.sparse.coo_matrix: sparse adjacency matrix representation of graph
        """
        if edge_attr is None:
            return torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index.as_tensor(), num_nodes=self.n)
        else:
            return torch_geometric.utils.to_scipy_sparse_matrix(
                self.data.edge_index.as_tensor(), edge_attr=self.data[edge_attr], num_nodes=self.n
            )

    @property
    def in_degrees(self) -> Dict[str, float]:
        """Return in-degrees of nodes in directed network.

        Returns:
            dict: dictionary containing in-degrees of nodes
        """
        return self.degrees(mode="in")

    @property
    def out_degrees(self) -> Dict[str, float]:
        """Return out-degrees of nodes in directed network.

        Returns:
            dict: dictionary containing out-degrees of nodes
        """
        return self.degrees(mode="out")

    def degrees(self, mode: str = "in") -> Dict[str, float]:
        """
        Return degrees of nodes.

        Args:
            mode: `in` or `out` to calculate the in- or out-degree for
                directed networks.

        Returns:
            dict: dictionary containing degrees of nodes
        """
        if mode == "in":
            d = torch_geometric.utils.degree(self.data.edge_index[1], num_nodes=self.n, dtype=torch.int)
        else:
            d = torch_geometric.utils.degree(self.data.edge_index[0], num_nodes=self.n, dtype=torch.int)
        return {self.mapping.to_id(i): d[i].item() for i in range(self.n)}

    def weighted_outdegrees(self) -> torch.Tensor:
        """
        Compute the weighted outdegrees of each node in the graph.

        Args:
            graph (Graph): pathpy graph object.

        Returns:
            tensor: Weighted outdegrees of nodes.
        """
        edge_weight = getattr(self.data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(self.data.num_edges, device=self.data.edge_index.device)
        weighted_outdegree = scatter(
            edge_weight, self.data.edge_index[0], dim=0, dim_size=self.data.num_nodes, reduce="sum"
        )
        return weighted_outdegree

    def transition_probabilities(self) -> torch.Tensor:
        """
        Compute transition probabilities based on weighted outdegrees.

        Returns:
            tensor: Transition probabilities.
        """
        weighted_outdegree = self.weighted_outdegrees()
        source_ids = self.data.edge_index[0]
        edge_weight = getattr(self.data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(self.data.num_edges, device=self.data.edge_index.device)
        return edge_weight / weighted_outdegree[source_ids]

    def laplacian(self, normalization: Any = None, edge_attr: Any = None) -> Any:
        """Return Laplacian matrix for a given graph.

        This wrapper method will use [`torch_geometric.utils.laplacian`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.laplacian)
        to return a Laplcian matrix representation of a given graph.

        Args:
            normalization: normalization parameter passed to pyG `get_laplacian`
                function
            edge_attr: optinal name of numerical edge attribute that shall
                be passed to pyG `get_laplacian` function as edge weight

        Returns:
            scipy.sparse.coo_matrix: Laplacian matrix representation of graph
        """
        if edge_attr is None:
            index, weight = torch_geometric.utils.get_laplacian(
                self.data.edge_index.as_tensor(), normalization=normalization
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)
        else:
            index, weight = torch_geometric.utils.get_laplacian(
                self.data.edge_index.as_tensor(),
                normalization=normalization,
                edge_weight=self.data[edge_attr],
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)

    def __getitem__(self, key: Union[tuple, str]) -> Any:
        """Return node, edge, or graph attribute.

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
            return self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]]
        else:
            raise KeyError(key[0] + " is not a node or edge attribute")

    def __setitem__(self, key: str, val: torch.Tensor) -> None:
        """Store node, edge, or graph attribute.

        Args:
            key: name of attribute to be stored
            val: value of attribute
        """
        if not isinstance(key, tuple):
            if key.startswith("node_"):
                if val.size(0) != self.n:
                    raise ValueError("Attribute must have same length as number of nodes")
                self.data[key] = val
            elif key.startswith("edge_"):
                if val.size(0) != self.m:
                    raise ValueError("Attribute must have same length as number of edges")
                self.data[key] = val
            else:
                self.data[key] = val
        elif key[0].startswith("node_"):  # type: ignore
            if key[0] not in self.data.keys():
                raise KeyError(
                    "Attribute does not yet exist. Setting the value of a specific node attribute"
                    + "requires that the attribute already exists."
                )
            self.data[key[0]][self.mapping.to_idx(key[1])] = val
        elif key[0].startswith("edge_"):  # type: ignore
            if key[0] not in self.data.keys():
                raise KeyError(
                    "Attribute does not yet exist. Setting the value of a specific node attribute"
                    + "requires that the attribute already exists."
                )
            self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]] = val
        else:
            raise KeyError("node and edge specific attributes should be prefixed with 'node_' or 'edge_'")

    @property
    def n(self) -> int:
        """
        Return number of nodes.

        Returns:
            int: number of nodes in the graph
        """
        return self.data.num_nodes  # type: ignore

    @property
    def m(self) -> int:
        """
        Return number of edges.

        Returns the number of edges in the graph. For an undirected graph, the number of directed edges is returned.

        Returns:
            int: number of edges in the graph
        """
        return self.data.num_edges  # type: ignore

    @property
    def order(self) -> int:
        """
        Return order of graph.

        Returns:
            int: order of the (De Bruijn) graph
        """
        return self.data.node_sequence.size(1)  # type: ignore

    def is_directed(self) -> bool:
        """Return whether graph is directed.

        Returns:
            bool: True if graph is directed, False otherwise
        """
        return not self.data.edge_index.is_undirected

    def is_undirected(self) -> bool:
        """Return whether graph is undirected.

        Returns:
            bool: True if graph is undirected, False otherwise
        """
        return self.data.edge_index.is_undirected

    def has_self_loops(self) -> bool:
        """Return whether graph contains self-loops.

        Returns:
            bool: True if graph contains self-loops, False otherwise
        """
        return self.data.has_self_loops()

    def __add__(self, other: Graph, reduce: str = "sum") -> Graph:
        """Combine Graph object with other Graph object.

        The semantics of this operation depends on the optional IndexMap
        of both graphs. If no IndexMap is included, the two underlying data objects
        are concatenated, thus merging edges from both graphs while leaving node indices
        unchanged. If both graphs include IndexMaps that assign node IDs to indices,
        indices will be adjusted, creating a new mapping for the union of node Ids in both graphs.

        Node IDs of graphs to be combined can be disjoint, partly overlapping or non-overlapping.

        Args:
            other: Other graph to be combined with this graph
            reduce: Reduction method for node attributes of nodes that are present in both graphs.
                Can be one of "sum", "mean", "mul", "min", "max". Default is "sum".

        Examples:
            Adding two graphs without node IDs:

            >>> g1 = pp.Graph.from_edge_index(torch.Tensor([[0,1,1],[1,2,3]]))
            >>> g1 = pp.Graph.from_edge_index(torch.Tensor([[0,2,3],[3,2,1]]))
            >>> print(g1 + g2)
            Graph with 3 nodes and 6 edges

            Adding two graphs with identical node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('a', 'c'), ('c', 'b')])
            >>> print(g1 + g2)
            Graph with 3 nodes and 4 edges

            Adding two graphs with non-overlapping node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('c', 'd'), ('d', 'e')])
            >>> print(g1 + g2)
            Graph with 6 nodes and 4 edges

            Adding two graphs with partly overlapping node IDs:

            >>> g1 = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
            >>> g2 = pp.Graph.from_edge_list([('b', 'd'), ('d', 'e')])
            >>> print(g1 + g2)
            Graph with 5 nodes and 4 edges
        """

        if self.order > 1:
            raise NotImplementedError("Add operator can only be applied to order 1 graphs")

        d1 = self.data.clone()
        m1 = self.mapping

        d2 = other.data.clone()
        m2 = other.mapping

        nodes = np.concatenate([m1.to_ids(np.arange(self.n)), m2.to_ids(np.arange(other.n))])
        mapping = IndexMap(np.unique(nodes))
        d1.edge_index = mapping.to_idxs(m1.to_ids(d1.edge_index), device=d1.edge_index.device)
        d2.edge_index = mapping.to_idxs(m2.to_ids(d2.edge_index), device=d2.edge_index.device)

        d = d1.concat(d2)
        d.num_nodes = mapping.num_ids()
        d.edge_index = EdgeIndex(d.edge_index, sparse_size=(d.num_nodes, d.num_nodes))

        # If both graphs contain node attributes, reduce them using the specified method
        for k in d1.keys():
            if k != "node_sequence" and k.startswith("node_"):
                if isinstance(d[k], torch.Tensor):
                    d[k] = torch_geometric.utils.scatter(
                        d[k],
                        mapping.to_idxs(
                            np.concatenate([m1.to_ids(np.arange(self.n)), m2.to_ids(np.arange(other.n))]),
                            device=d[k].device,
                        ),
                        dim_size=d.num_nodes,
                        reduce=reduce,
                    )
                else:
                    raise ValueError("Node attribute " + k + " is not a tensor and cannot be reduced.")
        return Graph(d, mapping=mapping)

    def __str__(self) -> str:
        """Return a string representation of the graph."""

        attr = self.data.to_dict()
        attr_types = {}
        for k in attr:
            t = type(attr[k])
            if t == torch.Tensor:
                attr_types[k] = str(t) + " -> " + str(attr[k].size())
            else:
                attr_types[k] = str(t)

        from pprint import pformat

        if self.is_undirected():
            s = "Undirected graph with {0} nodes and {1} (directed) edges\n".format(self.n, self.m)
        else:
            s = "Directed graph with {0} nodes and {1} edges\n".format(self.n, self.m)

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
