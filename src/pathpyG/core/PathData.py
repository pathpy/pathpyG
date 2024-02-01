"""Manages data on paths in graphs."""

from __future__ import annotations
from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional
)

from enum import Enum
from collections import defaultdict

import torch
from torch import Tensor, IntTensor, cat, sum
from torch.nested import as_nested_tensor, nested_tensor
from torch_geometric.utils import to_scipy_sparse_matrix, degree, coalesce

from pathpyG import Graph
from pathpyG import config
from pathpyG.core.IndexMap import IndexMap
from pathpyG.algorithms.temporal import extract_causal_trees



class PathData:
    """An object to store observations of paths, walks and DAGs.

    PathData stores observations of paths, walks and directed acyclic graphs.
    It provides methods to generate edge indices of weighted higher-order De Bruijn
    graph models of paths and walks.
    """

    def __init__(self, paths: nested_tensor, path_freq: Tensor, mapping: Optional[IndexMap] = None) -> None:
        """Create an empty PathData object."""
        self.paths = paths
        self.path_freq = path_freq
        if mapping is None:
            self.mapping: IndexMap = IndexMap()
        else:
            self.mapping: IndexMap = mapping
        self.index_translation: Dict = {}

    @property
    def num_paths(self) -> int:
        """Return the number of stored paths."""
        return self.paths.size(dim=0)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the underlying graph."""
        index = self.edge_index
        return len(index.reshape(-1).unique(dim=0))

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the underlying graph."""
        return self.edge_index.size(dim=1)

    # def add_edge(self, p: Tensor, freq: int = 1) -> None:
    #     """Add an observation of an edge traversal.

    #     This method adds an observation of a traversed edge.

    #     Args:
    #         p: edge_index

    #     Example:
    #         Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following snippet
    #         stores two observations of edge `A` --> `C`:
    #             ```py
    #             import pathpyG as pp

    #             paths = pp.PathData()
    #             paths.add_edge(torch.tensor([[0],[2]]), freq=2)
    #             ```
    #     """
    #     self.add_walk(p, freq)

    # def add_dag(self, p: Tensor, freq: int = 1) -> None:
    #     """Add an observation of a directed acyclic graph.
        
    #     This method adds an observation of a directed acyclic graph,
    #     i.e. a topologically sorted sequence of not necessarily
    #     unique nodes in a graph. Like a walk, a DAG is represented
    #     as an ordered edge index tensor. DAGs can be associated with an
    #     integer that captures the observation frequency.

    #     Path data that can be represented as a collection of directed
    #     acyclic graphs naturally arise in the analysis of time-respecting
    #     paths in temporal graphs.

    #     Args:
    #         p: topologically sorted edge_index of DAG
    #         freq: The number of times this DAG has been observed.

    #     Example:
    #         Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following code snippet
    #         stores three observations of the DAG with edges `A` --> `B`, `B` --> `C`, `B` --> `D`
    #             ```py
    #             import pathpyG as pp

    #             paths = pp.PathData()
    #             paths.add_dag(torch.tensor([[0,1], [1, 2], [1, 3]]))
    #             ```
    #     """
    #     i = len(self.paths)
    #     self.paths[i] = p
    #     self.path_types[i] = PathType.DAG
    #     self.path_freq[i] = freq

    # def add_walk(self, p: Tensor, freq: int = 1) -> None:
    #     """
    #     Add an observation of a path or a walk in a graph.

    #     This method adds an observation of a walk, i.e. a sequence of not necessarily
    #     unique nodes traversed in a graph. A walk of length l is represented as ordered
    #     edge index tensor of size (2,l) where l is the number of traversed edges.
    #     Walks can be associated with an integer that captures the observation frequency.

    #     Since walks are a generalization of paths that allows for multiple traversals of
    #     nodes, walks can be naturally used to store paths in a graph.

    #     Walks can be seen as a special case of DAGs where the in- and out-degree of all
    #     nodes is one. However, for a walk a higher-order model can be computed much more
    #     efficiently using a GPU-based 1D convolution operation. It is thus advisable to
    #     represent path data as walks whenever possible.

    #     Args:
    #         p:  An ordered edge index with size (2,l) that represents the sequence
    #             in which a walk or path traverses the nodes of a graph.
    #         freq:   The number of times this walk has been observed.

    #     Example:
    #         Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following snippet
    #         stores three observations of the walk `A` --> `C` --> `D`:
    #             ```py
    #             import pathpyG as pp

    #             paths = pp.PathData()
    #             paths.add_walk(torch.tensor([[0, 2],[2, 3]]), freq=5)
    #             ```
    #     """
    #     i = len(self.paths)
    #     self.paths[i] = p
    #     self.path_types[i] = PathType.WALK
    #     self.path_freq[i] = freq

    def to_scipy_sparse_matrix(self) -> Any:
        """Return sparse adjacency matrix of underlying graph."""
        return to_scipy_sparse_matrix(self.edge_index)

    @property
    def edge_index(self) -> Tensor:
        """Return edge index of a first-order graph model of all paths."""
        return self.edge_index_k_weighted(k=1)[0]

    @property
    def edge_index_weighted(self) -> Tuple[Tensor, Tensor]:
        """Return edge index and edge weights of a first-order graph
        model of all walks."""
        return self.edge_index_k_weighted(k=1)

    @staticmethod
    def nested_unfold(n: nested_tensor, k: int=1):
        a = []
        for t in n:
            if k < t.size(dim=1):
                a.append(t.unfold(1, k, 1))
            else:
                a.append(torch.tensor([], dtype=int, device=config['torch']['device']))    
        return torch.cat(a, dim=1)

    def edge_index_k_weighted(self, k: int = 1) -> Tuple[Tensor, Tensor]:
        """Compute edge index and edge weights of $k$-th order De Bruijn graph model.
        
        Args:
            k: order of the $k$-th order De Bruijn graph model
        """

        if k == 1:
            # here we simply concatenate all paths to obtain an edge_index
            i = torch.cat([t for t in self.paths], dim=1)

            # we assume that all edges in a path occured multiple times according to path_freq
            f = torch.cat([self.path_freq[i].repeat(t.size(dim=1)) for i, t in enumerate(self.paths)])

            # we coalesce the edge_index, which is just a regular torch_geometric edge index
            return coalesce(i, f)
        else:
            i = PathData.nested_unfold(self.paths, k)

            # here we cannot use the coalesce function as the edge_index will be dimension 2, l, k

            # we again assume that all higher-order edges occur multiple times according to path_freq
            f = torch.cat([self.path_freq[i].repeat(t.size(dim=1)-2+1 if t.size(dim=1)-2+1 > 0 else 0) for i, t in enumerate(self.paths)])
            

            # we make the edge index unique and keep a reverse index that maps each element in i 
            # to the corresponding element in edge_index
            edge_index, reverse_index = i.unique(dim=1, return_inverse=True)

            # for each edge, we now count all occurences
            edge_weights = torch.bincount(reverse_index, weights=f[reverse_index])

            # TODO: check whether weights are correct

        return edge_index, edge_weights

    # WALK METHODS

    @staticmethod
    def edge_index_kth_order_walk(edge_index: Tensor, k: int = 1) -> Tensor:
        """Compute edge index of $k$-th order graph for a given walk.

        The returned $k$-th order `edge_index` has size `(2, l-1, k)` and naturally 
        generalizes first-order edge indices, i.e. for a walk `(0,1,2,3,4,5)`
        represented by the following ordered `edge_index` with size `(2, 5)`

        ```py
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5]
        ]
        ```

        we obtain the following second-order `edge_index` with size `(2, 4, 2)`

        ```py
        [
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            [[1, 2], [2, 3], [3, 4], [4, 5]]
        ]
        ```

        while for the third-order `edge_index` we get a tensor with size `(2, 3, 3)`

        ```py
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        ]
        ```

        Note that for reasons of consistency with edge_index tensors in pyG,
        first-order `edge_indices` of walks of length $l$ have size `(2,l)` rather
        than size `(2, l, 1)`.

        Args:
            k: order of the $k$-th order model.
        """
        if k <= edge_index.size(dim=1):
            return edge_index.unfold(1, k, 1)

        return IntTensor([]).to(config['torch']['device'])

    @staticmethod
    def walk_to_node_seq(walk: Tensor) -> Tensor:
        """Turn `edge_index` for a walk into a sequence of traversed node indices.

        Args:
            walk: ordered `edge_index` of a given walk in a graph

        Example:
            ```pycon
            >>> import pathpyG as pp
            >>> s = pp.PathData.walk_to_node_seq(torch.tensor([[0,2],[2,3]]))
            >>> print(s)
            [0,2,3]
            ```
        """
        return cat([walk[:, 0], walk[1, 1:]])

    # DAG METHODS

    # @staticmethod
    # def edge_index_kth_order_dag(edge_index: Tensor, k: int) -> Tensor:
    #     """Calculate $k$-th order edge_index for a single dag.

    #     Args:
    #         k: order of $k$-th order model
    #     """
    #     x = edge_index
    #     for _ in range(1, k):
    #         x = PathData.lift_order_dag(x)
    #     return x

    @staticmethod
    def map_nodes(edge_index: Tensor, index_translation: Dict) -> Tensor:
        """Efficiently map node indices in edge_index tensor based on dictionary.

        Args:
            edge_index: the tensor for which indices shall be mapped
            index_translation: dictionary mapping incides in original tensor to new indices

        Example:
            ```py
            import pathpyG as pp
            edge_index = IntTensor([[0,1,2], [1,2,3]])

            print(edge_index)
            tensor([[0, 1, 3],
                    [1, 2, 3]])

            index_translation = {0: 1, 1: 0, 2: 3, 3: 2}
            mapped = pp.PathData.map_nodes(edge_index, index_translation)

            print(mapped)
            tensor([[1, 0, 3],
                    [0, 3, 2]])
            ```
        """
        # Inspired by `https://stackoverflow.com/questions/13572448`.
        palette, key = zip(*index_translation.items())
        key = torch.tensor(key).to(config['torch']['device'])
        palette = torch.tensor(palette).to(config['torch']['device'])

        index = torch.bucketize(edge_index.ravel(), palette)
        remapped = key[index].reshape(edge_index.shape)
        return remapped

    # @staticmethod
    # def lift_order_dag(edge_index: Tensor) -> Tensor:
    #     """Efficiently lift edge index of $k$-th order model to $(k+1)$-th order model.

    #     Args:
    #         edge_index: edge_index of $k$-th order model that will be
    #             lifted to $(k+1)$-th order
    #     """
    #     a = edge_index[0].unique(dim=0)
    #     b = edge_index[1].unique(dim=0)
    #     # intersection of a and b corresponds to all center nodes, which have 
    #     # at least one incoming and one outgoing edge
    #     combined = torch.cat((a, b))
    #     uniques, counts = combined.unique(dim=0, return_counts=True)
    #     center_nodes = uniques[counts > 1]

    #     src = []
    #     dst = []

    #     # create edges of order k+1
    #     for v in center_nodes:
    #         # get all predecessors of v, i.e. elements in edge_index[0] where edge_index[1] == v
    #         src_index = torch.all(edge_index[1] == v, axis=1).nonzero().flatten()  # type: ignore
    #         srcs = edge_index[0][src_index]
    #         # get all successors of v, i.e. elements in edge_index[1] where edge_index[0] == v
    #         dst_index = torch.all(edge_index[0] == v, axis=1).nonzero().flatten()  # type: ignore
    #         dsts = edge_index[1][dst_index]
    #         for s in srcs:
    #             for d in dsts:
    #                 src.append(torch.cat((torch.gather(s, 0, torch.tensor([0]).to(config['torch']['device'])), v)))
    #                 dst.append(torch.cat((v, torch.gather(d, 0, torch.tensor([d.size()[0]-1]).to(config['torch']['device'])))))

    #     if len(src) > 0:
    #         return torch.stack((torch.stack(src), torch.stack(dst)))
        
    #     return torch.tensor([]).to(config['torch']['device'])

    # @staticmethod
    # def from_temporal_dag(dag: Graph, detect_walks: bool = True) -> PathData:
    #     """Generate PathData object from temporal DAG where nodes are node-time events.

    #     Args:
    #         dag: A directed acyclic graph representation of a temporal network, where
    #             nodes are time-node events.
    #         detect_walks: whether or not directed acyclic graphs that just correspond
    #                     to walks will be automatically added as walks. If set to false
    #                     the resulting `PathData` object will only contain DAGs. If set
    #                     to true, the PathData object may contain both DAGs and walks.
    #     """
    #     ds = PathData()

    #     out_deg = degree(dag.data.edge_index[0])
    #     in_deg = degree(dag.data.edge_index[1])

    #     # check if dag exclusively consists of simple walks and apply fast method
    #     if torch.max(out_deg).item() == 1.0 and torch.max(in_deg).item() == 1.0:

    #         zero_outdegs = (out_deg==0).nonzero().squeeze()
    #         zero_indegs = (in_deg==0).nonzero().squeeze()

    #         # find indices of those elements in src where in-deg = 0, i.e. elements are in zero_indegs
    #         start_segs = torch.where(torch.isin(dag.data.edge_index[0], zero_indegs))[0]
    #         end_segs = torch.cat((start_segs[1:], torch.tensor([len(dag.data.edge_index[0])], device=config['torch']['device'])))
    #         segments = end_segs - start_segs
    #         index_translation = {
    #             i: dag['node_idx', dag.mapping.to_id(i)] for i in range(dag.N)
    #         }

    #         # Map node-time events to node IDs
    #         # Convert the tensor to a flattened 1D tensor
    #         flat_tensor = dag.data.edge_index.flatten()

    #         # Create a mask tensor to mark indices to be replaced
    #         mask = torch.zeros_like(flat_tensor, device=config['torch']['device'])

    #         for key, value in index_translation.items():
    #             # Find indices where the values match the keys in the mapping
    #             indices = (flat_tensor == key).nonzero(as_tuple=True)
                
    #             # Set the corresponding indices in the mask tensor to 1
    #             mask[indices] = 1
                
    #             # Replace values in the flattened tensor according to the mapping
    #             flat_tensor[indices] = value

    #         # Reshape the flattened tensor back to the original shape
    #         dag.data['edge_index'] = flat_tensor.reshape(dag.data.edge_index.shape)

    #         # split edge index into multiple independent sections: 
    #         # sections are limited by indices in src where in-deg = 0 and indices in tgt where out-deg = 0 
    #         for t in torch.split(dag.data.edge_index, segments.tolist(), dim=1):
    #             ds.add_walk(t)

    #     else:
    #         dags = extract_causal_trees(dag)
    #         for d in dags:
    #             # src = [ dag['node_idx', dag.node_index_to_id[s.item()]] for s in dags[d][0]] # type: ignore
    #             # dst = [ dag['node_idx', dag.node_index_to_id[t.item()]] for t in dags[d][1]] # type: ignore
    #             src = [s for s in dags[d][0]]
    #             dst = [t for t in dags[d][1]]
    #             # ds.add_dag(IntTensor([src, dst]).unique_consecutive(dim=1))
    #             edge_index = torch.LongTensor([src, dst]).to(config['torch']['device'])
    #             if detect_walks and degree(edge_index[1]).max() == 1 and \
    #                     degree(edge_index[0]).max() == 1:
    #                 ds.add_walk(edge_index)
    #             else:
    #                 ds.add_dag(edge_index)
    #         ds.index_translation = {
    #             i: dag['node_idx', dag.mapping.to_id(i)] for i in range(dag.N)
    #             }
    #     ds.mapping = IndexMap(dag.data['temporal_graph_index_map'])
    #     return ds

    def __str__(self) -> str:
        """Return string representation of PathData object."""
        num_walks = 0
        #num_dags = 0
        total = 0
        for i, _ in enumerate(self.paths):
            num_walks += 1
            total += self.path_freq[i]
        s = f"PathData with {num_walks} walks and total weight {total}"
        return s

    @staticmethod
    def from_csv(file: str, sep: str = ',') -> PathData:
        """Read path data from CSV file.

        The CSV file is expected to contain one walk or path per line, where
        nodes are separated by the character given in `sep`. The last
        component in the resulting n-gram is assumed to be the integer
        frequency of the observed walk.

        Args:
            file: filename of csv file containing paths or walks
            sep: character used to separate nodes and integer observation count
        """
        p = []
        w = []
        mapping = IndexMap()
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                path = []
                fields = line.split(sep)
                for v in fields[:-1]:
                    mapping.add_id(v)
                    path.append(mapping.to_idx(v))
                # Performance bottleneck: we need a better solution that a large dictionary 
                # with many tensors, each individually copied to the GPU
                # Possible solution: nested tensors
                p.append(torch.tensor([path[:-1], path[1:]], device=config['torch']['device']))
                w.append(int(float(fields[-1])))
        paths = as_nested_tensor(p, device=config['torch']['device'])
        path_freq = torch.tensor(w, device=config['torch']['device'])
        return PathData(paths, path_freq, mapping)
