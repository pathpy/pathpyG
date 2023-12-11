from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data

from pathpyG import Graph
from pathpyG import PathData
from pathpyG import TemporalGraph

from pathpyG.utils.config import config
from pathpyG.algorithms.temporal import temporal_graph_to_event_dag


# TODO: Add description for arguments
class HigherOrderGraph(Graph):
    """HigherOrderGraph based on torch_geometric.Data."""

    def __init__(self, paths: PathData, order: int = 1, node_id: Any = None, **kwargs: Any):
        """Generate HigherOrderGraph based on a given PathData instance.

        Args:
            paths:
            order:
            node_id:
            **kwargs:

        Example:
            ```py
            import pathpyG as pp

            paths = pp.PathData()
            paths.add_walk(torch.Tensor([[0, 1, 2], [1, 2, 3]]))
            g2 = Graph(paths, k=2, node_id=['a', 'b', 'c', 'd'])
            ```
        """
        if node_id is None:
            node_id = []

        assert len(node_id) == len(set(node_id)), 'node_id entries must be unique'

        # generate edge_index with higher-order nodes represented as tensors
        self.order = order

        index, edge_weights = paths.edge_index_k_weighted(k=order)

        if self.order > 1:
            # get tensor of unique higher-order nodes
            self._nodes = index.reshape(-1, index.size(dim=2)).unique(dim=0)

            # create mapping to first-order node indices
            ho_nodes_to_index = {tuple(j.tolist()): i for i, j in enumerate(self._nodes)}

            # create new tensor with node indices mapped to indices of higher-order nodes
            edge_index = torch.tensor( (
                [ho_nodes_to_index[tuple(x.tolist())] for x in index[0,:]],
                [ho_nodes_to_index[tuple(x.tolist())] for x in index[1,:]])
                ).to(config['torch']['device'])

            # create mappings between higher-order nodes (with ids) and node indices
            if len(node_id)>0:
                self.node_index_to_id = { i: tuple([node_id[v] for v in j.tolist()]) for i, j in enumerate(self._nodes)}
                self.node_id_to_index = { j: i for i, j in self.node_index_to_id.items()}
            else:
                self.node_index_to_id = { i:tuple([v for v in j.tolist()]) for i, j in enumerate(self._nodes)}
                self.node_id_to_index = { j:i for i, j in self.node_index_to_id.items()}

        else:
            self._nodes = index.reshape(-1).unique(dim=0)
            edge_index = index

            # create mappings between node ids and node indices
            self.node_index_to_id = dict(enumerate(node_id))
            self.node_id_to_index = {v: i for i, v in enumerate(node_id)}

        # Create pyG Data object
        self.data = Data(edge_index=edge_index, num_nodes=len(self._nodes), **kwargs)
        self.data['node_id'] = node_id
        self.data['edge_weight'] = edge_weights


        # create mapping between edge tuples and edge indices
        self.edge_to_index = {(e[0].item(), e[1].item()):i for i, e in enumerate([e for e in edge_index.t()])}

        # initialize adjacency matrix
        self._sparse_adj_matrix: Any = (
            torch_geometric.utils.to_scipy_sparse_matrix(self.data.edge_index).tocsr()
        )

    # @Graph.nodes.getter
    # def nodes(self):
    #     if len(self.node_id_to_index) > 0:
    #         for v in self.node_id_to_index:
    #             yield v
    #     else:
    #         for v in self._nodes:
    #             yield v

    # @Graph.nodes.getter
    # def edges(self):
    #     if len(self.node_index_to_id) > 0:
    #         for e in self.data.edge_index.t():
    #             yield self.node_id_to_index[e[0].item()], self.node_id_to_index[e[1].item()]
    #     else:
    #         for e in self.data.edge_index.t():
    #             yield e[0].item(), e[1].item()




    def __str__(self) -> str:
        """Return a string representation of the higher-order graph."""

        attr_types = Graph.attr_types(self.data.to_dict())

        s = "HigherOrderGraph (k={0}) with {1} nodes and {2} edges\n".format(self.order, len(self._nodes), self.M)
        s += "\tTotal edge weight = {0}".format(self['edge_weight'].sum())
        if len(self.data.node_attrs()) > 0:
            s += "\nNode attributes\n"
            for a in self.data.node_attrs():
                s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.edge_attrs()) > 1:
            s += "\nEdge attributes\n"
            for a in self.data.edge_attrs():
                if a != 'edge_index':
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        if len(self.data.keys()) > len(self.data.edge_attrs()) + len(self.data.node_attrs()):
            s += "\nGraph attributes\n"
            for a in self.data.keys():
                if not self.data.is_node_attr(a) and not self.data.is_edge_attr(a):
                    s += "\t{0}\t\t{1}\n".format(a, attr_types[a])
        return s

    @staticmethod
    def from_temporal_graph(g, delta, order=1):
        """Creates a higher-order De Bruijn graph model for paths in a temporal graph."""
        dag = temporal_graph_to_event_dag(g, delta=1)
        paths = PathData.from_temporal_dag(dag)
        return HigherOrderGraph(paths, order=order, node_id=g.data["node_id"])
