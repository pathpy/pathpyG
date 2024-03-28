"""Algorithms for the analysis of time-respecting paths in temporal graphs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union, List, Tuple
from collections import defaultdict
from torch_geometric.utils import degree, sort_edge_index

import numpy as np
import torch

from pathpyG import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.MultiOrderModel import MultiOrderModel

from pathpyG import config


def temporal_graph_to_event_dag(g: TemporalGraph, delta: float = np.infty, sparsify: bool = True) -> Graph:
    """Create directed acyclic event graph where nodes are node-time events and edges are time-respecting paths.

    Args:
        g: the temporal graph to be convered to an event dag
        delta: for a maximum time difference delta two consecutive interactions
                (u, v; t) and (v, w; t') are considered to contribute to a causal
                path from u via v to w iff t'-t <= delta
        sparsify: whether not to add edges for time-respecting paths
            between same nodes for multiple inter-event times
    """
    edge_list = []
    node_names = {}
    edge_times = []

    sources = defaultdict(set)

    for v, _w, t in g.temporal_edges:
        sources[t].add(v)

    for v, w, t in g.temporal_edges:

        if delta < np.infty:
            current_delta = int(delta)
        else:
            current_delta = g.end_time - g.start_time

        event_src = f"{v}-{t}"
        node_names[event_src] = v

        # create one time-unfolded link for all delta in [1, delta]
        # this implies that for delta = 2 and an edge (a,b,1) two
        # time-unfolded links (a_1, b_2) and (a_1, b_3) will be created
        cont = False
        for x in range(1, int(current_delta) + 1):

            # only add edge to event DAG if an edge (w,*) continues a time-respecting path at time t+x
            if w in sources[t + x] or not sparsify:
                event_dst = "{0}-{1}".format(w, t + x)
                node_names[event_dst] = w
                edge_list.append([event_src, event_dst])
                edge_times.append(t)
                cont = True
        if not cont and sparsify:  # if there is no continuing time-respecting path, include edge to t+1
            event_dst = "{0}-{1}".format(w, t + 1)
            node_names[event_dst] = w
            edge_list.append([event_src, event_dst])
            edge_times.append(t)

    dag = Graph.from_edge_list(edge_list)
    dag.data["node_name"] = [node_names[dag.mapping.to_id(i)] for i in range(dag.N)]
    dag.data["node_idx"] = [g.mapping.to_idx(v) for v in dag.data["node_name"]]
    dag.data["edge_ts"] = torch.tensor(edge_times)
    dag.data["temporal_graph_index_map"] = g.mapping.node_ids
    return dag


def extract_causal_trees(dag: Graph) -> Dict[Union[int, str], torch.IntTensor]:
    """Extract all causally isolated trees from a directed acyclic event graph.

    For a directed acyclic graph where all events are related to single root
    event, this function will return a single tree. For other DAGs, it will return
    multiple trees such that each root in the tree is not causally influenced by
    any other node-time event.

    Args:
        dag: the event graph to process
    """
    causal_trees = {}
    d = dag.degrees(mode="in")
    for v in d:
        if d[v] == 0:
            # print('Processing root', v)

            src: List[int] = []
            dst: List[int] = []

            visited = set()
            queue = [v]

            while queue:
                x = queue.pop()
                for w in dag.successors(x):
                    if w not in visited:
                        visited.add(w)
                        queue.append(w)
                        src.append(dag.mapping.to_idx(x))
                        dst.append(dag.mapping.to_idx(w))
            # TODO: Remove redundant zero-degree neighbors of all nodes
            causal_trees[v] = torch.IntTensor([src, dst]).to(config["torch"]["device"])
    return causal_trees


def routes_from_node(g: Graph, v: int, node_sequence: tensor, mapping: IndexMap):
    """
    Construct all paths from node v to any leaf node in a DAG

    Parameters
    ----------
    v:
        node from which to start
    node_mapping: dict
        an optional mapping from node to a different set.

    Returns
    -------
    Counter
    """
    # Collect temporary paths, indexed by the target node
    temp_paths = defaultdict(list)
    temp_paths[v] = [[mapping.to_id(x) for x in node_sequence[v].tolist()]]

    # set of unprocessed nodes
    queue = {v}

    while queue:
        # take one unprocessed node
        x = queue.pop()

        # successors of x expand all temporary
        # paths, currently ending in x
        c = 0
        for w in g.successors(x):
            c += 1
            for p in temp_paths[x]:
                temp_paths[w].append(p + [mapping.to_id(node_sequence[w][1].item())])
            queue.add(w)
        if c > 0:
            del temp_paths[x]
    # flatten dictionary
    return temp_paths


def time_respecting_paths(g: TemporalGraph, delta: float) -> defaultdict:
    """
    Calculate all longest time-respecting paths in a temporal graph.
    """
    in_degree = degree(g.data.edge_index[1], num_nodes=g.N)

    # first-order edge index
    edge_index, timestamps = sort_edge_index(g.data.edge_index, g.data.t)
    node_sequence = torch.arange(g.data.num_nodes, device=edge_index.device).unsqueeze(1)

    # second-order edge index
    null_model_edge_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
    # Update node sequences
    node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)
    # Remove non-time-respecting higher-order edges
    time_diff = timestamps[null_model_edge_index[1]] - timestamps[null_model_edge_index[0]]
    non_negative_mask = time_diff > 0
    delta_mask = time_diff <= delta
    time_respecting_mask = non_negative_mask & delta_mask
    edge_index = null_model_edge_index[:, time_respecting_mask]

    # identify root nodes with in-degree zero
    in_degree = degree(edge_index[1], num_nodes=g.M)
    roots = torch.where(in_degree == 0)[0]

    # create traversable graph
    event_dag = Graph.from_edge_index(edge_index)
    print(event_dag)

    # count all longest time-respecting paths in the temporal graph
    paths = defaultdict(lambda: list())
    i = 0
    for r in roots:
        if i % 10 == 0:
            print(f"Processing root {i+1}/{roots.size(0)}")
        root_paths = routes_from_node(event_dag, r.item(), node_sequence, g.mapping)
        for x in root_paths:
            for p in root_paths[x]:
                paths[len(p) - 1].append(p)
        i += 1
    return paths


def temporal_shortest_paths(g: TemporalGraph, delta: int) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Calculates all shortest paths between all pairs of nodes
    based on a set of empirically observed paths. The length of the
    shortest path for each node pair is provided. Additionally,
    the number of times each shortest path is observed is counted.
    """
    shortest_paths = {}
    shortest_path_lengths = torch.full((g.N, g.N), float("inf"), device=g.data.edge_index.device)
    shortest_path_lengths.fill_diagonal_(float(0))
    shortest_path_counts = torch.full((g.N, g.N), 0, device=g.data.edge_index.device)

    # first-order edge index
    edge_index, timestamps = sort_edge_index(g.data.edge_index, g.data.t)
    node_sequence = torch.arange(g.data.num_nodes, device=edge_index.device).unsqueeze(1)

    # second-order edge index with time-respective filtering
    k = 2
    null_model_edge_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
    # Update node sequences
    node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)
    # Remove non-time-respecting higher-order edges
    time_diff = timestamps[null_model_edge_index[1]] - timestamps[null_model_edge_index[0]]
    non_negative_mask = time_diff > 0
    delta_mask = time_diff <= delta
    time_respecting_mask = non_negative_mask & delta_mask
    edge_index = null_model_edge_index[:, time_respecting_mask]

    # Use node sequences to update shortest path lengths
    def update_paths(node_sequence: torch.Tensor, k: int) -> None:
        path_starts = node_sequence[:, 0]
        path_ends = node_sequence[:, -1]
        mask = shortest_path_lengths[path_starts, path_ends] >= k - 1
        shortest_path_lengths[path_starts[mask], path_ends[mask]] = k - 1
        if mask.sum() > 0:
            shortest_paths[k - 1] = torch.unique(node_sequence[mask], dim=0)
        pairs = torch.cat([shortest_paths[k - 1][:, 0].unsqueeze(1), shortest_paths[k - 1][:, -1].unsqueeze(1)], dim=1)
        unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
        shortest_path_counts[unique_pairs[:, 0], unique_pairs[:, 1]] = counts

    update_paths(node_sequence, k)

    k = 3
    while torch.max(shortest_path_lengths) > k and edge_index.size(1) > 0:
        print(f"k = {k}, edge_index size = {edge_index.size(1)}")
        ho_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
        node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)
        edge_index = ho_index
        update_paths(node_sequence, k)
        k += 1

    return shortest_paths, shortest_path_lengths, shortest_path_counts
