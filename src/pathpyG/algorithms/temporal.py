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


def temporal_graph_to_event_dag(g: TemporalGraph, delta: float = np.infty, create_mapping=False) -> Graph | None:
    """Create directed acyclic event graph where nodes are time-stamped edge events and edges are time-respecting paths of length two.

        Args:
            g: the temporal graph to be convered to an event dag
            delta: for a maximum time difference delta two consecutive interactions 
                    (u, v; t) and (v, w; t') are considered to contribute to a causal
                    path from u via v to w iff t'-t <= delta
    """
    # first-order edge index
    edge_index, timestamps = sort_edge_index(g.data.edge_index, g.data.t)
    node_sequence = torch.arange(g.data.num_nodes, device=edge_index.device).unsqueeze(1)

    # second-order edge index
    null_model_edge_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_sequence.size(0))
    # Update node sequences
    node_sequence = torch.cat([node_sequence[edge_index[0]], node_sequence[edge_index[1]][:, -1:]], dim=1)

    # Remove edges that do not correspond to time-respecting paths
    time_diff = timestamps[null_model_edge_index[1]] - timestamps[null_model_edge_index[0]]
    non_negative_mask = time_diff > 0
    delta_mask = time_diff <= delta
    time_respecting_mask = non_negative_mask & delta_mask
    edge_index = null_model_edge_index[:, time_respecting_mask]

    if edge_index.size(1) == 0:
        print('Warning: Temporal event DAG is empty')
        return None

    # construct graph object with mapping and node sequence tensor
    dag = Graph.from_edge_index(edge_index=edge_index)
    dag.data.node_sequence = node_sequence

    if create_mapping:
        dag.mapping = IndexMap([
            tuple(g.mapping.to_ids(node_sequence[i].tolist()) + [timestamps[i].item()])
                for i in range(node_sequence.size(0))
            ])
    return dag


def routes_from_node(dag: Graph, v: int, mapping: IndexMap):
    """
    Construct all paths from node v to any leaf node in a temporal event DAG

    Parameters
    ----------
    dag: Graph
        temporal event DAG
    v: int
        index of node in temporal event dag from which to explore paths
    node_mapping: IndexMap
        mapping that maps first-order node indices to IDs

    Returns
    -------
    Counter
    """
    # Collect all temporary paths, indexed by target node (initially v)
    temp_paths = defaultdict(list)
    temp_paths[v] = [ mapping.to_ids(dag.data.node_sequence[v].tolist()) ]

    # queue contains all unprocessed nodes
    queue = {v}

    while queue:
        # take one unprocessed node
        x = queue.pop()

        # successors of x expand all temporary paths that currently end in x
        c = 0
        for w in dag.successors(x):
            c += 1
            for p in temp_paths[x]:
                temp_paths[w].append(p + [mapping.to_id(dag.data.node_sequence[w][1].item())])
            queue.add(w)
        if c > 0:
            del temp_paths[x]
    # flatten dictionary
    return temp_paths


def time_respecting_paths(g: TemporalGraph, delta: float) -> defaultdict:
    """
    Calculate all longest time-respecting paths in a temporal graph.
    """

    paths = defaultdict(lambda: list())

    # Construct temporal event DAG
    event_dag = temporal_graph_to_event_dag(g, delta)
    if event_dag:
        print(f'Constructed temporal event DAG with {event_dag.N} nodes and {event_dag.M} edges')
    else:
        return paths
    
    # identify root nodes with in-degree zero
    in_degree = degree(event_dag.data.edge_index[1], num_nodes=event_dag.N)
    roots = torch.where(in_degree == 0)[0]

    # compute all longest time-respecting paths in temporal graph    
    i = 0
    for r in roots:
        if i % 10 == 0:
            print(f'Processing root {i+1}/{roots.size(0)}')
        root_paths = routes_from_node(event_dag, r.item(), g.mapping)
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
    node_idx = torch.arange(g.data.num_nodes, device=edge_index.device,dtype=torch.int32)
    unique_node_seq = node_idx.unsqueeze(1)

    def update_node_seq(node_idx, unique_node_seq):
        idx_seq = torch.cat([node_idx[edge_index[0]].unsqueeze(1), unique_node_seq[:, -1:][node_idx[edge_index[1]]]], dim=1)
        unique_idx_seq, node_idx = torch.unique(idx_seq, dim=0,return_inverse =True)
        unique_node_seq= torch.cat([unique_node_seq[unique_idx_seq[:,0]],unique_idx_seq[:,-1:]],dim=1)
        return unique_node_seq, node_idx

    # second-order edge index with time-respective filtering
    k = 2
    null_model_edge_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=node_idx.size(0))
    # Update node sequences
    unique_node_seq, node_idx= update_node_seq(node_idx, unique_node_seq)
    
    # Remove non-time-respecting higher-order edges
    time_diff = timestamps[null_model_edge_index[1]] - timestamps[null_model_edge_index[0]]
    non_negative_mask = time_diff > 0
    delta_mask = time_diff <= delta
    time_respecting_mask = non_negative_mask & delta_mask
    edge_index = null_model_edge_index[:, time_respecting_mask]

    # Use node sequences to update shortest path lengths
    def update_paths(unique_node_seq: torch.Tensor, k: int) -> None:
        path_starts = unique_node_seq[:, 0]
        path_ends = unique_node_seq[:, -1]

        mask = shortest_path_lengths[path_starts, path_ends] >= k - 1
        shortest_path_lengths[path_starts[mask], path_ends[mask]] = k - 1
        if mask.sum() > 0:
            sp = unique_node_seq[mask]
            pairs = torch.cat([sp[:, 0].unsqueeze(1), sp[:, -1].unsqueeze(1)], dim=1)
            unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
            shortest_path_counts[unique_pairs[:, 0], unique_pairs[:, 1]] = counts
            shortest_paths[k - 1] = sp.cpu()

    update_paths(unique_node_seq, k)
    k = 3
    while torch.max(shortest_path_lengths) > k and edge_index.size(1) > 0:
        size = node_idx.size(0)
        unique_node_seq, node_idx= update_node_seq(node_idx, unique_node_seq)
        edge_index = MultiOrderModel.lift_order_edge_index(edge_index, num_nodes=size)
        update_paths(unique_node_seq, k)
        k += 1
        

    return shortest_paths, shortest_path_lengths, shortest_path_counts