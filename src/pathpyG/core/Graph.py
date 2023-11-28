from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
    Union,
    Any,
    Optional,
    Generator,
)

import torch

import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.transforms.to_undirected import ToUndirected

from pathpyG.utils.config import config


class Graph:
    """
    A graph object storing nodes, edges, and attributes.

    An object than be be used to store directed or undirected graphs with node
    and edge attributes. Data on nodes and edges are stored in an underlying instance of
    `torch_geometric.Data`.
    """

    def __init__(self, edge_index: torch.Tensor,
                 node_id: Optional[List[str]] = None,
                 **kwargs: Optional[torch.Tensor]):
        """Generate graph instance from an edge index.

        Generate a Graph instance from a `torch.Tensor` that contains an `edge_index`
        with optional `node_id` list that maps integer node indices to string node ids.

        Args:
            edge_index: edge_index containing source and target
            index of all edges

            node_id:    Optional list of node identifiers

            **kwargs:   Optional keyword arguments that are passed to constructor
            of torch_geometric.Data. Keyword arguments starting with `node_` will be
            mapped to node attributes, keywords arguments starting with `edge_` will
            be mapped to edge attributes. Other keyword arguments will be mapped to
            graph attributes.
        
        Usage example:

            import pathpyG as pp

            g = pp.Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]))

            g = pp.Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
                                    node_id=['a', 'b', 'c'])

            g = pp.Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
                                    node_id=['a', 'b', 'c'],
                                    node_age=torch.LongTensor([12, 42, 17]),
                                    edge_weight=torch.FloatTensor([1.0, 2.5, 0.7[]))
        """
        if node_id is None:
            node_id = []

        assert len(node_id) == len(set(node_id)), "node_id entries must be unique"

        # Create pyG Data object
        if len(node_id) == 0:
            self.data = Data(edge_index=edge_index, node_id=node_id, **kwargs)
        else:
            self.data = Data(
                edge_index=edge_index, node_id=node_id, num_nodes=len(node_id), **kwargs
            )

        # create mappings between node ids and node indices
        self.node_index_to_id = dict(enumerate(node_id))
        self.node_id_to_index = {v: i for i, v in enumerate(node_id)}

        # create mapping between edge tuples and edge indices
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in edge_index.t()])
        }

        # initialize adjacency matrix
        self._sparse_adj_matrix: Any = (
            torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()
        )

    def add_node_id(self, node_id: List[str]) -> None:
        """Add a mapping of node indices to node IDs.

        Args:
            node_id: list of string IDs, corresponding to node indices
        """
        assert len(node_id) == len(set(node_id)), "node_id entries must be unique"

        self.node_index_to_id = dict(enumerate(node_id))
        self.node_id_to_index = {v: i for i, v in enumerate(node_id)}
        self.data["node_id"] = node_id

    def to_undirected(self) -> None:
        """
        Transform graph into undirected graph.

        This method transforms the current graph instance into an undirected graph by
        adding all directed edges in opposite direction. It applies `ToUndirected`
        transform to the underlying `torch_geometric.Data` object, which automatically
        duplicates edge attributes for newly created directed edges.

        Usage example:

            import pathpyG as pp
            g = pp.Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]))
            g.to_undirected()
        """
        tf = ToUndirected()
        self.data = tf(self.data)

    @staticmethod
    def attr_types(attr: Dict) -> Dict:
        """
        Return name, type, and size of all node, edge, and graph attributes.

        This method returns a dictionary that contains the name (key), as well as
        the type and size of all attributes.
        """
        a = {}
        for k in attr:
            t = type(attr[k])
            if t == torch.Tensor:
                a[k] = str(t) + " -> " + str(attr[k].size())
            else:
                a[k] = str(t)
        return a

    def node_attrs(self) -> List:
        """
        Return a list of node attributes.

        This method returns a list containing the names of all node-level attributes,
        ignoring the special `node_id` attribute.
        """
        attrs = []
        for k in self.data.keys():
            if k != "node_id" and k.startswith("node_"):
                attrs.append(k)
        return attrs

    def edge_attrs(self) -> List:
        """
        Return a list of edge attributes.

        This method returns a list containing the names of all edge-level attributes,
        ignoring the special `edge_index` attribute.
        """
        attrs = []
        for k in self.data.keys():
            if k != "edge_index" and k.startswith("edge_"):
                attrs.append(k)
        return attrs

    @property
    def nodes(self) -> Generator[Union[int, str], None, None]:
        """
        Return indices or IDs of all nodes in the graph.

        This method returns a generator object that yields all nodes.
        If `node_id` is used to map node indices to string IDs, nodes
        are returned as str IDs. If no mapping to IDs is used, nodes
        are returned as integer indices.
        """
        if len(self.node_id_to_index) > 0:
            for v in self.node_id_to_index:
                yield v
        else:
            for v in range(self.N):
                yield v

    @property
    def edges(self) -> Generator[Union[Tuple[int, int], Tuple[str, str]], None, None]:
        """
        Return all edges in the graph.
        
        This method returns a generator object that yields all edges.
        If `node_id` is used to map node indices to string IDs, edges
        are returned as tuples of str IDs. If no mapping to IDs is used, nodes
        are returned as tuples of integer indices.
        """
        if len(self.node_index_to_id) > 0:
            for e in self.data.edge_index.t():
                yield self.node_index_to_id[e[0].item()], self.node_index_to_id[
                    e[1].item()
                ]
        else:
            for e in self.data.edge_index.t():
                yield e[0].item(), e[1].item()

    def successors(self, node: Union[int, str] | tuple) \
            -> Generator[Union[int, str] | tuple, None, None]:
        """
        Return the successors of a given node.

        This method returns a generator object that yields all successors of a
        given node. If a `node_id` mapping is used, successors will be returned
        as string IDs. If no mapping is used, successors are returned as indices.

        Args:
            node:   Index or string ID of node for which successors shall be returned.
        """ 
        coo_matrix = self._sparse_adj_matrix.tocoo()      
        if len(self.node_index_to_id) > 0:
            # Get array of col indices for which entries in row are non-zero
            non_zero_cols = coo_matrix.col[coo_matrix.row == self.node_id_to_index[node]]
            for i in non_zero_cols:  # type: ignore
                yield self.node_index_to_id[i]
        else:
            non_zero_cols = coo_matrix.row[coo_matrix.col == node]
            for i in non_zero_cols:  # type: ignore
                yield i

    def predecessors(self, node: Union[str, int] | tuple) \
            -> Generator[Union[int, str] | tuple, None, None]:
        """Return the predecessors of a given node.

        This method returns a generator object that yields all predecessors of a
        given node. If a `node_id` mapping is used, predecessors will be returned
        as string IDs. If no mapping is used, predecessors are returned as indices.

        Args:
            node:   Index or string ID of node for which predecessors shall be returned.
        """
        coo_matrix = self._sparse_adj_matrix.tocoo()      
        if len(self.node_index_to_id) > 0:
            # Get array of col indices for which entries in row are non-zero
            non_zero_rows = coo_matrix.row[coo_matrix.col == self.node_id_to_index[node]]
            for i in non_zero_rows:  # type: ignore
                yield self.node_index_to_id[i]
        else:
            non_zero_rows = coo_matrix.row[coo_matrix.col == node]
            for i in non_zero_rows:  # type: ignore
                yield i

    def is_edge(self, v: Union[str, int], w: Union[str, int]) -> bool:
        """Return whether edge (v,w) exists in the graph.
        
        If an index to ID mapping is used, nodes are assumed to be string IDs. If no
        mapping is used, nodes are assumed to be integer indices.

        Args:
            v: source node of edge as integer index or string ID
            w: target node of edge as integer index or string ID       
        """
        if len(self.node_index_to_id) > 0:
            return self.node_id_to_index[w] in self._sparse_adj_matrix.getrow(self.node_id_to_index[v]).indices  # type: ignore
        else:
            return w in self._sparse_adj_matrix.getrow(v).indices  # type: ignore

    def get_sparse_adj_matrix(self, edge_attr: Any = None) -> Any:
        """Return sparse adjacency matrix representation of (weighted) graph.

        Args:
            edge_attr: the edge attribute that shall be used as edge weight
        """
        if edge_attr is None:
            return torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index)
        else:
            return torch_geometric.utils.to_scipy_sparse_matrix(
                self.data.edge_index, edge_attr=self.data[edge_attr]
            )

    @property
    def in_degrees(self) -> Dict:
        """Return in-degrees of nodes in directed network."""
        return self.degrees(mode="in")

    @property
    def out_degrees(self) -> Dict:
        """Return out-degrees of nodes in directed network."""
        return self.degrees(mode="out")

    def degrees(self, mode: str = "in") -> Dict:
        """
        Return degrees of nodes.

        Args:
            mode:   `in` or `out` to calculate the in- or out-degree for
                directed networks.
        """
        if mode == "in":
            d = torch_geometric.utils.degree(
                self.data.edge_index[1], num_nodes=self.N, dtype=torch.int
            )
        else:
            d = torch_geometric.utils.degree(
                self.data.edge_index[0], num_nodes=self.N, dtype=torch.int
            )
        if len(self.node_id_to_index) > 0:
            return {
                v: d[self.node_id_to_index[v]].item() for v in self.node_id_to_index
            }
        else:
            return {i: d[i].item() for i in range(self.N)}

    def get_laplacian(self, normalization: Any = None, edge_attr: Any = None) -> Any:
        """Return Laplacian matrix for a given graph.

        This wrapper method will use `torch_geometric.utils.get_laplacian`
        to return a Laplcian matrix representation of a given graph.

        Args:
            normalization:  normalization parameter passed to pyG `get_laplacian`
                            function
            edge_attr:      optinal name of numerical edge attribute that shall
                            be passed to pyG `get_laplacian` function as edge weight
        """
        if edge_attr is None:
            return torch_geometric.utils.get_laplacian(
                self.data.edge_index, normalization=normalization
            )
        else:
            return torch_geometric.utils.get_laplacian(
                self.data.edge_index,
                normalization=normalization,
                edge_weight=self.data[edge_attr],
            )

    def add_node_ohe(self, attr_name: str, dim: int = 0) -> None:
        """Add one-hot encoding of nodes to node attribute.

        Args:
            attr_name: attribute name used to store one-hot encoding
            dim: dimension of one-hot encoding
        """
        if dim == 0:
            dim = self.N
        self.data[attr_name] = torch.eye(dim, dtype=torch.float).to(
            config["torch"]["device"]
        )[: self.N]

    def add_edge_ohe(self, attr_name: str, dim: int = 0) -> None:
        """Add one-hot encoding of edges to edge attribute.

        Args:
            attr_name: attribute name used to store one-hot encoding
            dim: dimension of one-hot encoding
        """
        if dim == 0:
            dim = self.M
        self.data[attr_name] = torch.eye(dim, dtype=torch.float).to(
            config["torch"]["device"]
        )[: self.M]

    def __getitem__(self, key: Union[tuple, str]) -> Any:
        """Return node, edge, or graph attribute.

        Args:
            key: name of attribute to be returned
        """
        if isinstance(key, tuple):
            if key in self.data.keys():
                return self.data[key]
            else:
                print(key, "is not a graph attribute")
                return None
        elif key[0] in self.node_attrs():
            if len(self.node_id_to_index) > 0:
                return self.data[key[0]][self.node_id_to_index[key[1]]]
            else:
                return self.data[key[0]][key[1]]
        elif key[0] in self.edge_attrs():
            if len(self.node_id_to_index) > 0:
                return self.data[key[0]][
                    self.edge_to_index[
                        self.node_id_to_index[key[1]], self.node_id_to_index[key[2]]
                    ]
                ]
            else:
                return self.data[key[0]][self.edge_to_index[key[1], key[2]]]
        elif key in self.data.keys():
            return self.data[key[0]]
        else:
            print(key[0], "is not a node or edge attribute")
            return None

    def __setitem__(self, key: str, val: torch.Tensor) -> None:
        """Store node, edge, or graph attribute.

        Args:
            key: name of attribute to be stored
            val: value of attribute
        """
        if isinstance(key, tuple):
            if key in self.data.keys():
                self.data[key] = val
            else:
                print(key, "is not a graph attribute")
        elif self.key[0].starts_with("node_"):  # type: ignore
            if len(self.node_id_to_index) > 0:
                self.data[key[0]][self.node_id_to_index[key[1]]] = val
            else:
                self.data[key[0]][key[1]] = val
        elif self.key[0].starts_with("edge_"):  # type: ignore
            if len(self.node_id_to_index) > 0:
                self.data[key[0]][
                    self.edge_to_index[
                        self.node_id_to_index[key[1]], self.node_id_to_index[key[2]]
                    ]
                ] = val
            else:
                self.data[key[0]][self.edge_to_index[key[1], key[2]]] = val
        else:
            print(key[0], "is not a node or edge attribute")

    @property
    def N(self) -> int:
        """
        Return number of nodes.

        Returns the number of nodes in the graph.
        """
        return self.data.num_nodes  # type: ignore

    @property
    def M(self) -> int:
        """
        Return number of edges.

        Returns the number of edges in the graph.
        """
        return self.data.num_edges  # type: ignore

    @staticmethod
    def from_pyg_data(d: Any) -> Graph:
        """
        Construct a graph from a `pytorch_geometric.Data` object.

        Args:
            d:  `pytorch_geometric.Data` object containing an edge_index as well as 
                arbitrary node, edge, and graph-level attributes
        """
        x = d.to_dict()

        del x["edge_index"]

        if d.is_node_attr("node_id"):
            del x["node_id"]
            g = Graph(d.edge_index, node_id=d["node_id"], **x)
        else:
            g = Graph(d.edge_index, node_id=[], **x)
        return g

    def to_pyg_data(self) -> Any:
        """Return torch_geometric.Data representing the graph with its attributes."""
        return self.data

    def is_directed(self) -> Any:
        """Return whether graph is directed."""
        return self.data.is_directed()

    def is_undirected(self) -> Any:
        """Return whether graph is undirected."""
        return self.data.is_undirected()

    def has_self_loops(self) -> Any:
        """Return whether graph contains self-loops."""
        return self.data.has_self_loops()

    @staticmethod
    def from_edge_list(edge_list: List[List[str]]) -> Graph:
        """Generate a Graph instance based on an edge list.

        Args:
            edge_list: List of iterables

        Usage example:
        
            import pathpyG as pp
            
            l = [['a', 'b'], ['b', 'c'], ['a', 'c']]
            g = pp.Graph.from_edge_list(l)    
        """
        sources = []
        targets = []

        nodes_index = dict()
        index_nodes = dict()

        n = 0
        for v, w in edge_list:
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

        return Graph(
            edge_index=torch.LongTensor([sources, targets]).to(
                config["torch"]["device"]
            ),
            node_id=[index_nodes[i] for i in range(n)],
        )

    def __str__(self) -> str:
        """Return a string representation of the graph."""

        attr_types = Graph.attr_types(self.data.to_dict())

        s = "Graph with {0} nodes and {1} edges\n".format(self.N, self.M)
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

    def __getattr__(self, name: str) -> Any:
        """Map unknown method to corresponding method of networkx `Graph` object."""
        def wrapper(*args, **kwargs) -> Any:
            # print('unknown method {0} was called, delegating call to networkx object'.format(name))
            g = torch_geometric.utils.to_networkx(self.data)
            return getattr(g, name)(*args, **kwargs)
        return wrapper
