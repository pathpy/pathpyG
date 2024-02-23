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
from torch import IntTensor, Tensor, cat
from torch_geometric import EdgeIndex

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.PathData import PathData

class WalkData(PathData):

    def add(self, p: Tensor, freq: int = 1) -> None:
        """Add an observation of a path or a walk in a graph based on a tensor representation

        This method adds an observation of a walk, i.e. a sequence of not necessarily
        unique nodes traversed in a graph. A walk of length l is represented as ordered
        edge index tensor of size (2,l) where l is the number of traversed edges.
        Walks can be associated with an integer that captures the observation frequency.

        Since walks are a generalization of paths that allows for multiple traversals of
        nodes, walks can be naturally used to store paths in a graph.

        Walks can be seen as a special case of DAGs where the in- and out-degree of all
        nodes is one. However, for a walk a higher-order model can be computed much more
        efficiently using a GPU-based 1D convolution operation. It is thus advisable to
        represent path data as walks whenever possible.

        Args:
            p:  An ordered edge index with size (2,l) that represents the sequence
                in which a walk or path traverses the nodes of a graph.
            freq:   The number of times this walk has been observed.

        Example:
            Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following snippet
            stores three observations of the walk `A` --> `C` --> `D`:
                ```py
                import pathpyG as pp

                paths = pp.PathData()
                paths.add_walk(torch.tensor([[0, 2],[2, 3]]), freq=5)
                ```
        """
        i = len(self.paths)
        self.paths[i] = p
        self.path_freq[i] = freq

    def add_edge(self, p: Tensor, freq: int = 1) -> None:
        """Add an observation of an edge traversal.

        This method adds an observation of a traversed edge.

        Args:
            p: edge_index

        Example:
            Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following snippet
            stores two observations of edge `A` --> `C`:
                ```py
                import pathpyG as pp

                paths = pp.PathData()
                paths.add_edge(torch.tensor([[0],[2]]), freq=2)
                ```
        """
        self.add(p, freq)

    def add_walk_seq(self, node_seq, freq=1):
        """Add an observation of a walk based on a sequence of node IDs or indices"""
        idx_seq = [self.mapping.to_idx(v) for v in node_seq ]
        w = IntTensor([idx_seq[:-1], idx_seq[1:]]).to(config['torch']['device'])
        self.add(w, freq)
        

    def __str__(self) -> str:
        """Return string representation of WalkData object."""
        num_walks = 0
        total = 0
        for p in self.paths:
            num_walks += 1
            total += self.path_freq[p]
        s = f"PathData with {num_walks} walks and total weight {total}"
        return s

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

    def edge_index_k_weighted(self, k: int = 1) -> Tuple[Tensor, Tensor]:
        """Compute edge index and edge weights of $k$-th order De Bruijn graph model.
        
        Args:
            k: order of the $k$-th order De Bruijn graph model
        """
        freq: Tensor = torch.Tensor([])

        if k == 1:
            i = cat(list(self.paths.values()), dim=1)
            if self.index_translation:
                i = PathData.map_nodes(i, self.index_translation)
            l_f = []
            for idx in self.paths:
                l_f.append(Tensor([self.path_freq[idx]]*self.paths[idx].size()[1]).to(config['torch']['device']))
            freq = cat(l_f, dim=0)
        else:
            l_p = []
            l_f = []
            for idx in self.paths:
                p = WalkData.edge_index_kth_order(self.paths[idx], k)
                if self.index_translation:
                    p = PathData.map_nodes(p, self.index_translation).unique_consecutive(dim=0)
                l_p.append(p)
                l_f.append(Tensor([self.path_freq[idx]]*(self.paths[idx].size()[1]-k+1)).to(config['torch']['device']))                
            i = cat(l_p, dim=1)
            freq = cat(l_f, dim=0)

        # make edge index unique and keep reverse index, 
        # that maps each element in i to the corresponding element in edge_index
        edge_index, reverse_index = i.unique(dim=1, return_inverse=True)

        # for each edge in edge_index, the elements of x
        # contain all indices in i that correspond to that edge
        x = list((reverse_index == idx).nonzero() 
                 for idx in range(edge_index.size()[1]))

        # for each edge, sum the weights of all occurences
        edge_weights = Tensor([
            sum(freq[x[idx]]) for idx in
            range(edge_index.size()[1])]).to(config['torch']['device'])

        return edge_index, edge_weights


    @staticmethod
    def edge_index_kth_order(edge_index: Tensor, k: int = 1) -> Tensor:
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
    def from_csv(file: str, sep: str = ',', freq=True) -> PathData:
        """Read walk data from CSV file.

        The CSV file is expected to contain one walk per line, where
        nodes are separated by the character given in `sep`. The last
        component in the resulting n-gram is assumed to be the integer
        frequency of the observed walk.

        Args:
            file: filename of csv file containing paths or walks
            sep: character used to separate nodes and integer observation counts
        """
        p = WalkData()
        mapping = IndexMap()
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                path = []
                count = 1
                fields = line.split(sep)                
                if freq:
                    for v in fields[:-1]:
                        mapping.add_id(v)
                        path.append(mapping.to_idx(v))
                    count = int(float(fields[-1]))
                else:
                    for v in fields:
                        mapping.add_id(v)
                        path.append(mapping.to_idx(v))
                w = IntTensor([path[:-1], path[1:]]).to(config['torch']['device'])
                p.add(w, count)
        p.mapping = mapping
        return p