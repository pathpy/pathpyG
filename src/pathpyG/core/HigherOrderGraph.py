from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Any, Optional, Generator

import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric import EdgeIndex

from pathpyG import Graph
from pathpyG import PathData
from pathpyG import TemporalGraph

from pathpyG.utils.config import config
from pathpyG.algorithms.temporal import temporal_graph_to_event_dag
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.HigherOrderIndexMap import HigherOrderIndexMap

# TODO: Add description for arguments
class HigherOrderGraph(Graph):
    """HigherOrderGraph based on torch_geometric.Data."""

    def __init__(self, paths: PathData, order: int = 1, **kwargs: Any):
        """Generate HigherOrderGraph based on a given PathData instance.

        Args:
            paths:
            order:
            **kwargs:

        Example:
            ```py
            import pathpyG as pp

            paths = pp.PathData()
            paths.mapping = pp.IndexMap(['a', 'b', 'c', 'd'])
            paths.add_walk(torch.Tensor([[0, 1, 2], [1, 2, 3]]))
            g2 = HigherOrderGraph(paths, k=2)
            ```
        """

        # generate edge_index with higher-order nodes represented as tensors
        self.order = order

        index, edge_weights = paths.edge_index_k_weighted(k=order)

        if self.order > 1:
            # get tensor of unique higher-order nodes
            _nodes = index.reshape(-1, index.size(dim=2)).unique(dim=0)
            self.mapping = HigherOrderIndexMap(_nodes, paths.mapping.node_ids)

            # create mapping from higher-order nodes (i.e. tensors) to node indices, i.e.
            # assign consecutive indices 0, 1, 2 to higher-order nodes [0,1], [1,2], [2,3]
            ho_nodes_to_index = {tuple(j.tolist()): i for i, j in enumerate(_nodes)}

            # create new tensor with node indices mapped to indices of higher-order nodes
            edge_index = torch.tensor((
                [ho_nodes_to_index[tuple(x.tolist())] for x in index[0, :]],
                [ho_nodes_to_index[tuple(x.tolist())] for x in index[1, :]])
                ).to(config['torch']['device'])

            # Create pyG Data object
            edge_index = edge_index.contiguous()
            self.data = Data(edge_index=EdgeIndex(edge_index), num_nodes=len(_nodes), **kwargs)
            self.data['edge_weight'] = edge_weights

        else:
            _nodes = index.reshape(-1).unique(dim=0)
            edge_index = index

            # create mappings between node ids and node indices
            self.mapping = paths.mapping

            # Create pyG Data object
            edge_index = edge_index.contiguous()
            self.data = Data(edge_index=EdgeIndex(edge_index), num_nodes=len(_nodes), **kwargs)
            self.data['edge_weight'] = edge_weights

        # sort EdgeIndex and validate
        self.data.edge_index = self.data.edge_index.sort_by('row').values
        self.data.edge_index.validate()


        # create mapping between edge tuples and edge indices
        self.edge_to_index = {(e[0].item(), e[1].item()):i for i, e in enumerate([e for e in edge_index.t()])}

    def __str__(self) -> str:
        """Return a string representation of the higher-order graph."""

        attr_types = Graph.attr_types(self.data.to_dict())

        s = "HigherOrderGraph (k={0}) with {1} nodes and {2} edges\n".format(self.order, self.N, self.M)
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
    def from_temporal_graph(g, delta=1, order=1):
        """Creates a higher-order De Bruijn graph model for paths in a temporal graph."""
        dag = temporal_graph_to_event_dag(g, delta=delta)
        paths = PathData.from_temporal_dag(dag)
        return HigherOrderGraph(paths, order=order, node_ids=g.mapping.node_ids)
