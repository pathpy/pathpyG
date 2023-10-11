from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, AnyStr, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data, HeteroData
from scipy.sparse import csr_array
from torch_geometric.transforms.to_undirected import ToUndirected
from pathpyG.utils.config import config

class Graph:
    """ Graph based on torch_geometric.Data"""

    def __init__(self, edge_index: torch.Tensor, node_id: List=[], **kwargs):
        """
        Generates a Graph instance based on torch.Tensor containing edge_index with
        optional node_id mapping node indices to node ids

        Example:
        >>> g = Graph(torch.LongTensor([[1, 1, 2], [0, 2, 1]]),
                                node_id=['a', 'b', 'c'])
        """

        assert len(node_id) == len(set(node_id)), 'node_id entries must be unique'

        # Create pyG Data object
        if len(node_id)==0:
            self.data = Data(edge_index=edge_index, node_id=node_id, **kwargs)
        else:
            self.data = Data(edge_index=edge_index, node_id=node_id, num_nodes=len(node_id), **kwargs)

        # create mappings between node ids and node indices
        self.node_index_to_id = dict(enumerate(node_id))
        self.node_id_to_index = {v: i for i, v in enumerate(node_id)}

        # create mapping between edge tuples and edge indices
        self.edge_to_index = {(e[0].item(), e[1].item()):i for i, e in enumerate([e for e in edge_index.t()])}

        # initialize adjacency matrix
        self._sparse_adj_matrix: csr_array = torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()


    def add_node_id(self, node_id):
        assert len(node_id) == len(set(node_id)), 'node_id entries must be unique'

        self.node_index_to_id = dict(enumerate(node_id))
        self.node_id_to_index = {v: i for i, v in enumerate(node_id)}
        self.data['node_id'] = node_id

    def to_undirected(self):
        """Transforms a graph into an undirected graph, by adding all directed edges in opposite direction.
            Applies a ToUndirected transform to the underlying torch_geometric.Data object, which will
            automatically duplicate edge attributes for newly created directed edges.
        """
        tf = ToUndirected()
        self.data = tf(self.data)

    @staticmethod
    def attr_types(attr: Dict) -> Dict:
        a = {}
        for k in attr:
            t = type(attr[k])
            if t == torch.Tensor:
                a[k] = str(t) + ' -> ' + str(attr[k].size())
            else:
                a[k] = str(t)
        return a

    def node_attrs(self) -> List:
        attrs = []
        for k in self.data.keys:
            if k != 'node_id' and k.startswith('node_'):
                attrs.append(k)
        return attrs

    def edge_attrs(self) -> List:
        attrs = []
        for k in self.data.keys:
            if k != 'edge_index' and k.startswith('edge_'):
                attrs.append(k)
        return attrs

    @property
    def nodes(self) -> Generator[Union[int, str], None, None]:
        if len(self.node_id_to_index) > 0:
            for v in self.node_id_to_index:
                yield v
        else:
            for v in range(self.N):
                yield v

    @property
    def edges(self) -> Generator[Union[Tuple[int, int], Tuple[str, str]], None, None]:
        if len(self.node_index_to_id) > 0:
            for e in self.data.edge_index.t():
                yield self.node_index_to_id[e[0].item()], self.node_index_to_id[e[1].item()]
        else:
            for e in self.data.edge_index.t():
                yield e[0].item(), e[1].item()

    def successors(self, node) -> Generator[Union[int, str], None, None]:
        if len(self.node_index_to_id) > 0:
            for i in self._sparse_adj_matrix.getrow(self.node_id_to_index[node]).indices: # type: ignore
                yield self.node_index_to_id[i]
        else:
            for i in self._sparse_adj_matrix.getrow(node).indices: # type: ignore
                yield i

    def predecessors(self, node) -> Generator[Union[int, str], None, None]:
        if len(self.node_index_to_id) > 0:
            for i in self._sparse_adj_matrix.getcol(self.node_id_to_index[node]).indices: # type: ignore
                yield self.node_index_to_id[i]
        else:
            for i in self._sparse_adj_matrix.getcol(node).indices: # type: ignore
                yield i

    def is_edge(self, v, w):
        if len(self.node_index_to_id) > 0:
            return self.node_id_to_index[w] in self._sparse_adj_matrix.getrow(self.node_id_to_index[v]).indices # type: ignore
        else:
            return w in self._sparse_adj_matrix.getrow(v).indices # type: ignore

    def get_sparse_adj_matrix(self, edge_attr=None):
        if edge_attr == None:
            return torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index)
        else:
            return torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index,
                                                                edge_attr=self.data[edge_attr])

    @property
    def in_degrees(self) -> Dict:
        return self.degrees(mode='in')

    @property
    def out_degrees(self) -> Dict:
        return self.degrees(mode='out')

    def degrees(self, mode='in') -> Dict:
        if mode == 'in':
            d = torch_geometric.utils.degree(self.data.edge_index[1], num_nodes = self.N, dtype=torch.int)
        else:
            d = torch_geometric.utils.degree(self.data.edge_index[0], num_nodes = self.N, dtype=torch.int)
        if len(self.node_id_to_index)>0:
            return {v: d[self.node_id_to_index[v]].item() for v in self.node_id_to_index}
        else:
            return {i: d[i].item() for i in range(self.N)}

    def get_laplacian(self, normalization=None, edge_attr=None):
        if edge_attr == None:
            return torch_geometric.utils.get_laplacian(self.data.edge_index,
                                                       normalization=normalization)
        else:
            return torch_geometric.utils.get_laplacian(self.data.edge_index,
                                                       normalization=normalization,
                                                       edge_weight=self.data[edge_attr])

    def add_node_ohe(self, attr_name, dim=0):
        if dim == 0:
            dim = self.N
        self.data[attr_name] = torch.eye(dim, dtype=torch.float).to(config['torch']['device'])[:self.N]

    def add_edge_ohe(self, attr_name, dim=0):
        if dim == 0:
            dim = self.M
        self.data[attr_name] = torch.eye(dim, dtype=torch.float).to(config['torch']['device'])[:self.M]


    def __getitem__(self, key):

        if type(key) != tuple:
            if key in self.data.keys:
                return self.data[key]
            else:
                print(key, 'is not a graph attribute')
        elif key[0] in self.node_attrs():
            if len(self.node_id_to_index) > 0:
                return self.data[key[0]][self.node_id_to_index[key[1]]]
            else:
                return self.data[key[0]][key[1]]
        elif key[0] in self.edge_attrs():
            if len(self.node_id_to_index) > 0:
                return self.data[key[0]][self.edge_to_index[self.node_id_to_index[key[1]], self.node_id_to_index[key[2]]]]
            else:
                return self.data[key[0]][self.edge_to_index[key[1], key[2]]]
        elif key in self.data.keys:
                return self.data[key[0]]
        else:
            print(key[0], 'is not a node or edge attribute')


    def __setitem__(self, key, val):
        if type(key) != tuple:
            if key in self.data.keys:
                self.data[key] = val
            else:
                print(key, 'is not a graph attribute')
        elif self.key[0].starts_with('node_'): # type: ignore
            if len(self.node_id_to_index) > 0:
                self.data[key[0]][self.node_id_to_index[key[1]]] = val
            else:
                self.data[key[0]][key[1]] = val
        elif self.key[0].starts_with('edge_'): # type: ignore
            if len(self.node_id_to_index) > 0:
                self.data[key[0]][self.edge_to_index[self.node_id_to_index[key[1]], self.node_id_to_index[key[2]]]] = val
            else:
                self.data[key[0]][self.edge_to_index[key[1], key[2]]] = val
        else:
            print(key[0], 'is not a node or edge attribute')

    @property
    def N(self) -> int:
        return self.data.num_nodes # type: ignore

    @property
    def M(self) -> int:
        return self.data.num_edges

    @staticmethod
    def from_pyg_data(d: Data):

        x = d.to_dict()

        del x['edge_index']

        if d.is_node_attr('node_id'):
            del x['node_id']
            g = Graph(d.edge_index, node_id=d['node_id'], **x)
        else:
            g = Graph(d.edge_index, node_id=[], **x)
        return g

    def to_pyg_data(self) -> Data | HeteroData:
        """
        Returns an instance of torch_geometric.data.Data containing the
        edge_index as well as node, edge, and graph attributes
        """
        return self.data

    def is_directed(self):
        return self.data.is_directed()

    def is_undirected(self):
        return self.data.is_undirected()

    def has_self_loops(self):
        return self.data.has_self_loops()

    @staticmethod
    def from_edge_list(edge_list) -> Graph:
        """
        Generates a Graph instance based on an edge list.

        Example:
        >>> Graph.from_edge_list([['a', 'b'], ['b', 'c'], ['a', 'c']])
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

        return Graph(edge_index=torch.LongTensor([sources, targets]).to(config['torch']['device']),
                     node_id=[index_nodes[i] for i in range(n)])

    def __str__(self) -> str:
        """
        Returns a string representation of the graph
        """

        attr_types = Graph.attr_types(self.data.to_dict())

        s = "Graph with {0} nodes and {1} edges\n".format(self.N, self.M)
        if len(self.data.node_attrs()) > 0:
            s += "\nNode attributes\n"
            for a in self.data.node_attrs():
                s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.edge_attrs()) > 1:
            s += "\nEdge attributes\n"
            for a in self.data.edge_attrs():
                if a != 'edge_index':
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.keys) > len(self.data.edge_attrs()) + len(self.data.node_attrs()):
            s += "\nGraph attributes\n"
            for a in self.data.keys:
                if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        return s
