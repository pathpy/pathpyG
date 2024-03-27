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
from torch_geometric.utils import degree

from pathpyG.utils.config import config
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.PathData import PathData
from pathpyG.core.Graph import Graph
from pathpyG.algorithms.temporal import extract_causal_trees


class DAGData(PathData):
    """Class that can be used to store multiple observations of
    directed acyclic graphs.

    Example:
        ```py
        import pathpyG as pp
        from torch import IntTensor

        pp.config['torch']['device'] = 'cuda'

        # Generate toy example graph
        g = pp.Graph.from_edge_list([('a', 'c'),
                             ('b', 'c'),
                             ('c', 'd'),
                             ('c', 'e')])

        # Generate data on observed directed acyclic graphs
        paths = pp.DAGData(g.mapping)
        dag = IntTensor([[0,2,2], # a -> c, c -> d, c -> e
                  [2,3,4]])
        paths.add(dag, freq=1)
        dag = IntTensor([[1,2,2], # b -> c, c -> d, c -> e
                  [2,3,4]])
        paths.add(dag, freq=1)
        print(paths)

        print(paths.edge_index_k_weighted(k=2))
        ```
    """

    def add(self, p: Tensor, freq: int = 1) -> None:
        """Add an observation of a directed acyclic graph.

        This method adds an observation of a directed acyclic graph,
        i.e. a topologically sorted sequence of not necessarily
        unique nodes in a graph. Like a walk, a DAG is represented
        as an ordered edge index tensor. DAGs can be associated with an
        integer that captures the observation frequency.

        Path data that can be represented as a collection of directed
        acyclic graphs naturally arise in the analysis of time-respecting
        paths in temporal graphs.

        Args:
            p: topologically sorted edge_index of DAG
            freq: The number of times this DAG has been observed.

        Example:
            Assuming a `node_id` mapping of `['A', 'B', 'C', 'D']` the following code snippet
            stores three observations of the DAG with edges `A` --> `B`, `B` --> `C`, `B` --> `D`
                ```py
                import pathpyG as pp

                paths = pp.PathData()
                paths.add_dag(torch.tensor([[0,1], [1, 2], [1, 3]]))
                ```
        """
        i = len(self.paths)
        self.paths[i] = p
        self.path_freq[i] = freq

    def __str__(self) -> str:
        """Return string representation of DAGData object."""
        num_dags = 0
        total = 0
        for p in self.paths:
            num_dags += 1
            total += self.path_freq[p]
        s = f"DAGData with {num_dags} dags and total weight {total}"
        return s

    @staticmethod
    def lift_order_dag(edge_index: Tensor) -> IntTensor:
        """Efficiently lift edge index of $k$-th order De Bruijn graph model to $(k+1)$-th order.

        Args:
            edge_index: $k$-th order edge_index to be lifted to $(k+1)$-th order
        """
        a = edge_index[0].unique(dim=0)
        b = edge_index[1].unique(dim=0)
        # intersection of a and b corresponds to all center nodes, which have
        # at least one incoming and one outgoing edge
        combined = torch.cat((a, b))
        uniques, counts = combined.unique(dim=0, return_counts=True)
        center_nodes = uniques[counts > 1]

        src = []
        dst = []

        # create all edges of order k+1
        for v in center_nodes:
            # get all predecessors of v, i.e. elements in edge_index[0] where edge_index[1] == v
            src_index = torch.all(edge_index[1] == v, axis=1).nonzero().flatten()  # type: ignore
            srcs = edge_index[0][src_index]
            # get all successors of v, i.e. elements in edge_index[1] where edge_index[0] == v
            dst_index = torch.all(edge_index[0] == v, axis=1).nonzero().flatten()  # type: ignore
            dsts = edge_index[1][dst_index]
            for s in srcs:
                for d in dsts:
                    src.append(torch.cat((torch.gather(s, 0, torch.tensor([0]).to(config["torch"]["device"])), v)))
                    dst.append(
                        torch.cat(
                            (v, torch.gather(d, 0, torch.tensor([d.size()[0] - 1]).to(config["torch"]["device"])))
                        )
                    )

        if len(src) > 0:
            return torch.stack((torch.stack(src), torch.stack(dst)))

        return torch.tensor([]).to(config["torch"]["device"])

    @staticmethod
    def edge_index_kth_order(edge_index: Tensor, k: int = 1) -> Tensor:
        """Calculate $k$-th order edge_index for a single DAG represented by a tensor.

        Args:
            k: order of $k$-th order De Bruijn Graph model
        """
        # we have to reshape tensors of the form [[0,1,2], [1,2,3]] to [[[0],[1],[2]],[[1],[2],[3]]]
        x = edge_index.reshape(edge_index.size() + (1,))
        for _ in range(1, k):
            x = DAGData.lift_order_dag(x)
        return x

    def edge_index_k_weighted(self, k: int = 1) -> Tuple[Tensor, Tensor]:
        """Compute edge index and edge weights of $k$-th order De Bruijn graph model.

        Args:
            k: order of the $k$-th order De Bruijn graph model
        """
        freq: Tensor = torch.Tensor([])

        if k == 1:
            # TODO: Wrong edge statistics for non-sparse DAGs!
            i = cat(list(self.paths.values()), dim=1)
            if self.index_translation:
                i = PathData.map_nodes(i, self.index_translation)
            l_f = []
            for idx in self.paths:
                l_f.append(Tensor([self.path_freq[idx]] * self.paths[idx].size()[1]).to(config["torch"]["device"]))
            freq = cat(l_f, dim=0)
        else:
            l_p = []
            l_f = []
            for idx in self.paths:
                p = DAGData.edge_index_kth_order(self.paths[idx], k)
                if self.index_translation:
                    p = PathData.map_nodes(p, self.index_translation).unique_consecutive(dim=0)
                if len(p) > 0:
                    l_p.append(p)
                    l_f.append(Tensor([self.path_freq[idx]] * p.size()[1]).to(config["torch"]["device"]))
            i = cat(l_p, dim=1)
            freq = cat(l_f, dim=0)

        # make edge index unique and keep reverse index,
        # that maps each element in i to the corresponding element in edge_index
        edge_index, reverse_index = i.unique(dim=1, return_inverse=True)

        # for each edge in edge_index, the elements of x
        # contain all indices in i that correspond to that edge
        x = list((reverse_index == idx).nonzero() for idx in range(edge_index.size()[1]))

        # for each edge, sum the weights of all occurences
        edge_weights = Tensor([sum(freq[x[idx]]) for idx in range(edge_index.size()[1])]).to(config["torch"]["device"])

        return edge_index, edge_weights

    @staticmethod
    def from_temporal_dag(dag: Graph) -> PathData:
        """Generate DAGData object from temporal DAG where nodes are node-time events.

        Args:
            dag: A directed acyclic graph representation of a temporal network, where
                nodes are time-node events.
            detect_walks: whether or not directed acyclic graphs that just correspond
                        to walks will be automatically added as walks. If set to false
                        the resulting `PathData` object will only contain DAGs. If set
                        to true, the PathData object may contain both DAGs and walks.
        """
        ds = DAGData()

        out_deg = degree(dag.data.edge_index[0])
        in_deg = degree(dag.data.edge_index[1])

        # check if dag exclusively consists of simple walks and apply fast method
        if torch.max(out_deg).item() == 1.0 and torch.max(in_deg).item() == 1.0:

            zero_outdegs = (out_deg == 0).nonzero().squeeze()
            zero_indegs = (in_deg == 0).nonzero().squeeze()

            # find indices of those elements in src where in-deg = 0, i.e. elements are in zero_indegs
            start_segs = torch.where(torch.isin(dag.data.edge_index[0], zero_indegs))[0]
            end_segs = torch.cat(
                (start_segs[1:], torch.tensor([len(dag.data.edge_index[0])], device=config["torch"]["device"]))
            )
            segments = end_segs - start_segs
            index_translation = {i: dag["node_idx", dag.mapping.to_id(i)] for i in range(dag.N)}

            # Map node-time events to node IDs
            # Convert the tensor to a flattened 1D tensor
            flat_tensor = dag.data.edge_index.flatten()

            # Create a mask tensor to mark indices to be replaced
            mask = torch.zeros_like(flat_tensor, device=config["torch"]["device"])

            for key, value in index_translation.items():
                # Find indices where the values match the keys in the mapping
                indices = (flat_tensor == key).nonzero(as_tuple=True)

                # Set the corresponding indices in the mask tensor to 1
                mask[indices] = 1

                # Replace values in the flattened tensor according to the mapping
                flat_tensor[indices] = value

            # Reshape the flattened tensor back to the original shape
            dag.data["edge_index"] = flat_tensor.reshape(dag.data.edge_index.shape)

            # split edge index into multiple independent sections:
            # sections are limited by indices in src where in-deg = 0 and indices in tgt where out-deg = 0
            for t in torch.split(dag.data.edge_index, segments.tolist(), dim=1):
                ds.add(t)

        else:
            dags = extract_causal_trees(dag)
            for d in dags:
                # src = [ dag['node_idx', dag.node_index_to_id[s.item()]] for s in dags[d][0]] # type: ignore
                # dst = [ dag['node_idx', dag.node_index_to_id[t.item()]] for t in dags[d][1]] # type: ignore
                src = [s for s in dags[d][0]]
                dst = [t for t in dags[d][1]]
                # ds.add_dag(IntTensor([src, dst]).unique_consecutive(dim=1))
                edge_index = torch.LongTensor([src, dst]).to(config["torch"]["device"])
                ds.add(edge_index)
            ds.index_translation = {i: dag["node_idx", dag.mapping.to_id(i)] for i in range(dag.N)}
        ds.mapping = IndexMap(dag.data["temporal_graph_index_map"])
        return ds
