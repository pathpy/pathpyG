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
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.transforms.to_undirected import ToUndirected

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap


class Graph:
    """
    A graph object storing nodes, edges, and attributes.

    An object than be be used to store directed or undirected graphs with node
    and edge attributes. Data on nodes and edges are stored in an underlying instance of
    [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).
    """

    def __init__(self, data: Data, mapping: Optional[IndexMap] = None):
        """Generate graph instance from a pyG `Data` object.

        Generate a Graph instance from a `torch_geometric.Data` object that contains an EdgeIndex
        with an optional mapping of node IDs to indices.

        Args:
            data: A pyG Data object containing an EdgeIndex and meta-data
            mapping: Optional `IndexMap` object that maps node IDs to indices

        Example:
            ```py
            import pathpyG as pp
            from torch_geometric.data import Data
            from torch_geometric import EdgeIndex

            data = Data(edge_index=EdgeIndex([[1,1,2],[0,2,1]]))
            g = pp.Graph(data)

            g = pp.Graph(data, mapping=pp.IndexMap(['a', 'b', 'c']))
            ```
        """
        if mapping is None:
            self.mapping = IndexMap()
        else:
            self.mapping = mapping

        # turn edge index tensor into EdgeIndex object
        if not isinstance(data.edge_index, EdgeIndex):
            data.edge_index = EdgeIndex(data=data.edge_index)

        # sort EdgeIndex and validate
        data.edge_index = data.edge_index.sort_by('row').values
        data.edge_index.validate()

        # set num_nodes property
        data.num_nodes = data.edge_index.max().item()+1
        self.data = data

        # create mapping between edge tuples and edge indices
        self.edge_to_index = {
            (e[0].item(), e[1].item()): i
            for i, e in enumerate([e for e in self.data.edge_index.t()])
        }

    @staticmethod
    def from_edge_index(edge_index: torch.Tensor, mapping: Optional[IndexMap] = None) -> Graph:
        """
        Construct a graph from a torch Tensor containing and edge index.

        Args:
            d:  [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object containing an edge_index as well as 
                arbitrary node, edge, and graph-level attributes
        
        Example:
            ```py
            import pathpyG as pp

            g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]))
            print(g)

            g = pp.Graph.from_edge_index(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
                                    mapping=pp.IndexMap(['a', 'b', 'c']))
            print(g)
            ```
        """       

        return Graph(
            Data(edge_index=edge_index, mapping=mapping)
        )


    @staticmethod
    def from_edge_list(edge_list: List[List[str]], is_undirected=False) -> Graph:
        """Generate a Graph instance based on an edge list.

        Args:
            edge_list: List of iterables

        Example:
            ```
            import pathpyG as pp

            l = [['a', 'b'], ['b', 'c'], ['a', 'c']]
            g = pp.Graph.from_edge_list(l)
            print(g.data)
            print(g.mapping)
            ```
        """
        sources = []
        targets = []

        mapping = IndexMap()

        for v, w in edge_list:
            mapping.add_id(v)
            mapping.add_id(w)
            sources.append(mapping.to_idx(v))
            targets.append(mapping.to_idx(w))

        e = EdgeIndex([sources, targets], is_undirected=is_undirected, device=config['torch']['device'])
        return Graph(Data(edge_index=e), mapping=mapping)

    def to_undirected(self) -> Graph:
        """
        Returns an undirected version of a directed graph.

        This method transforms the current graph instance into an undirected graph by
        adding all directed edges in opposite direction. It applies [`ToUndirected`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.ToUndirected.html#torch_geometric.transforms.ToUndirected)
        transform to the underlying [`torch_geometric.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object, which automatically
        duplicates edge attributes for newly created directed edges.

        Example:
            ```py
            import pathpyG as pp
            g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c'), ('c', 'a')])
            g_u = g.to_undirected()
            ```
        """
        tf = ToUndirected()
        d = tf(self.data)
        # unfortunately, the application of a transform creates a new edge_index of type tensor
        # so we have to recreate the EdgeIndex tensor and sort it again

        e = EdgeIndex(data=d.edge_index, is_undirected=True)
        d.edge_index = e
        return Graph(d, self.mapping)

    def to_weighted_graph(self) -> Graph:
        """Coalesces multi-edges to single-edges with an additional weight attribute"""
        i, w = torch_geometric.utils.coalesce(self.data.edge_index, torch.ones(self.M).to(config["torch"]["device"]))
        return Graph(Data(edge_index=i, edge_weight=w, num_nodes=int(i.max().item()+1)), self.mapping)

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
        If an IndexMap is used, nodes
        are returned as str IDs. If no IndexMap is used, nodes
        are returned as integer indices.
        """
        for i in range(self.N):
            yield self.mapping.to_id(i)

    @property
    def edges(self) -> Generator[Union[Tuple[int, int], Tuple[str, str]], None, None]:
        """
        Return all edges in the graph.
        
        This method returns a generator object that yields all edges.
        If an IndexMap is used to map node indices to string IDs, edges
        are returned as tuples of str IDs. If no mapping is used, nodes
        are returned as tuples of integer indices.
        """
        for e in self.data.edge_index.t():
            yield self.mapping.to_id(e[0].item()), self.mapping.to_id(e[1].item())

    def get_successors(self, row_idx: int):
        """
        """
        ((row_ptr, col), perm) = self.data.edge_index.get_csr()        
        if row_idx + 1 < row_ptr.size(0):
            row_start = row_ptr[row_idx]
            row_end = row_ptr[row_idx + 1]
            return col[row_start:row_end]
        else:
            return torch.tensor([])

    def get_predecessors(self, col_idx: int):
        """
        """
        ((col_ptr, row), perm) = self.data.edge_index.get_csc()        
        if col_idx + 1 < col_ptr.size(0):
            col_start = col_ptr[col_idx]
            col_end = col_ptr[col_idx + 1]
            return row[col_start:col_end]
        else:
            return torch.tensor([])

    def successors(self, node: Union[int, str] | tuple) \
            -> Generator[Union[int, str] | tuple, None, None]:
        """
        Return the successors of a given node.

        This method returns a generator object that yields all successors of a
        given node. If an Index mapping is used, successors will be returned
        as string IDs. If no mapping is used, successors are returned as indices.

        Args:
            node:   Index or string ID of node for which successors shall be returned.
        """ 

        for j in self.get_successors(self.mapping.to_idx(node)):  # type: ignore
            yield self.mapping.to_id(j.item())

    def predecessors(self, node: Union[str, int] | tuple) \
            -> Generator[Union[int, str] | tuple, None, None]:
        """Return the predecessors of a given node.

        This method returns a generator object that yields all predecessors of a
        given node. If a `node_id` mapping is used, predecessors will be returned
        as string IDs. If no mapping is used, predecessors are returned as indices.

        Args:
            node:   Index or string ID of node for which predecessors shall be returned.
        """
        for i in self.get_predecessors(self.mapping.to_idx(node)):  # type: ignore
            yield self.mapping.to_id(i.item())

    def is_edge(self, v: Union[str, int], w: Union[str, int]) -> bool:
        """Return whether edge $(v,w)$ exists in the graph.
        
        If an index to ID mapping is used, nodes are assumed to be string IDs. If no
        mapping is used, nodes are assumed to be integer indices.

        Args:
            v: source node of edge as integer index or string ID
            w: target node of edge as integer index or string ID       
        """
        row = self.mapping.to_idx(v)
        ((row_ptr, col), perm) = self.data.edge_index.get_csr()
        row_start = row_ptr[row]
        row_end   = row_ptr[row + 1]

        return self.mapping.to_idx(w) in col[row_start:row_end]

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
    def in_degrees(self) -> Dict[str, float]:
        """Return in-degrees of nodes in directed network."""
        return self.degrees(mode="in")

    @property
    def out_degrees(self) -> Dict[str, float]:
        """Return out-degrees of nodes in directed network."""
        return self.degrees(mode="out")

    def degrees(self, mode: str = "in") -> Dict[str, float]:
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
        return {self.mapping.to_id(i): d[i].item() for i in range(self.N)}

    def get_laplacian(self, normalization: Any = None, edge_attr: Any = None) -> Any:
        """Return Laplacian matrix for a given graph.

        This wrapper method will use [`torch_geometric.utils.get_laplacian`](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.get_laplacian)
        to return a Laplcian matrix representation of a given graph.

        Args:
            normalization:  normalization parameter passed to pyG `get_laplacian`
                            function
            edge_attr:      optinal name of numerical edge attribute that shall
                            be passed to pyG `get_laplacian` function as edge weight
        """
        if edge_attr is None:
            index, weight =torch_geometric.utils.get_laplacian(
                self.data.edge_index, normalization=normalization
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)
        else:
            index, weight = torch_geometric.utils.get_laplacian(
                self.data.edge_index,
                normalization=normalization,
                edge_weight=self.data[edge_attr],
            )
            return torch_geometric.utils.to_scipy_sparse_matrix(index, weight)

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
        if not isinstance(key, tuple):
            if key in self.data.keys():
                return self.data[key]
            else:
                print(key, "is not a graph attribute")
                return None
        elif key[0] in self.node_attrs():
            return self.data[key[0]][self.mapping.to_idx(key[1])]
        elif key[0] in self.edge_attrs():
            return self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]]
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
        if not isinstance(key, tuple):
            if key in self.data.keys():
                self.data[key] = val
            else:
                print(key, "is not a graph attribute")
        elif self.key[0].starts_with("node_"):  # type: ignore
            self.data[key[0]][self.mapping.to_idx(key[1])] = val
        elif self.key[0].starts_with("edge_"):  # type: ignore
            self.data[key[0]][self.edge_to_index[self.mapping.to_idx(key[1]), self.mapping.to_idx(key[2])]] = val
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


    def is_directed(self) -> Any:
        """Return whether graph is directed."""
        return not self.data.edge_index.is_undirected()

    def is_undirected(self) -> Any:
        """Return whether graph is undirected."""
        return self.data.edge_index.is_undirected()

    def has_self_loops(self) -> Any:
        """Return whether graph contains self-loops."""
        return self.data.has_self_loops()

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
