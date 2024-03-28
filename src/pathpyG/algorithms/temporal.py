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
                paths[len(p)-1].append(p)
        i += 1
    return paths

def temporal_shortest_paths(g: TemporalGraph, delta: float) -> Tuple[defaultdict, defaultdict]:
    """Calculate shortest time-respecting paths between all pairs of nodes in a temporal graph"""
    # calculate all longest time-respecting paths
    paths = time_respecting_paths(g, delta)

    # Todo: expand sub-paths contained in longest paths

    # calculate shortest time-respecting paths
    s_p = defaultdict(lambda: defaultdict(set))
    s_p_lengths = defaultdict(lambda: defaultdict(lambda: np.inf))

    for p_length in paths:
        for p in paths[p_length]:
            s = p[0]
            d = p[-1]
            # we found a shorter path of length l between s and d
            if p_length < s_p_lengths[s][d]:
                # update shortest path length
                s_p_lengths[s][d] = p_length
                # redefine set
                s_p[s][d] = set()
                s_p[s][d].add(tuple(p))
            elif p_length == s_p_lengths[s][d]:
                s_p[s][d].add(tuple(p))
    return s_p, s_p_lengths
