"""Algorithms for the analysis of time-respecting paths in temporal graphs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Union, List, Tuple

import numpy as np
from tqdm import tqdm
import torch
from scipy.sparse.csgraph import dijkstra

from pathpyG import Graph
from pathpyG.core.temporal_graph import TemporalGraph

from pathpyG import config

device = config['torch']['device']

def lift_order_temporal(g: TemporalGraph, delta: int = 1):

    # first-order edge index
    edge_index, timestamps = g.data.edge_index, g.data.time

    indices = torch.arange(0, edge_index.size(1), device=g.data.edge_index.device)

    unique_t = torch.unique(timestamps, sorted=True)
    second_order = []

    # lift order: find possible continuations for edges in each time stamp
    for i in tqdm(range(unique_t.size(0))):
        t = unique_t[i]
        
        # find indices of all source edges that occur at unique timestamp t
        src_time_mask = (timestamps == t)
        src_edges = edge_index[:,src_time_mask]
        src_edge_idx = indices[src_time_mask]

        # find indices of all edges that can possibly continue edges occurring at time t for the given delta
        dst_time_mask = (timestamps > t) & (timestamps <= t+delta)
        dst_edges = edge_index[:,dst_time_mask]
        dst_edge_idx = indices[dst_time_mask]

        if dst_edge_idx.size(0)>0 and src_edge_idx.size(0)>0:

            # compute second-order edges between src and dst idx for all edges where dst in src_edges matches src in dst_edges        
            x = torch.cartesian_prod(src_edge_idx, dst_edge_idx).t()
            src_edges = torch.index_select(edge_index, dim=1, index=x[0])
            dst_edges = torch.index_select(edge_index, dim=1, index=x[1])
            ho_edge_index = x[:,torch.where(src_edges[1,:] == dst_edges[0,:])[0]]
            second_order.append(ho_edge_index)

    ho_index = torch.cat(second_order, dim=1)
    return ho_index

def temporal_shortest_paths(g: TemporalGraph, delta: int):
    # generate temporal event DAG
    edge_index = lift_order_temporal(g, delta).to(device)

    # Add indices of first-order nodes as src and dst of paths in augmented
    # temporal event DAG
    src_edges_src = (g.data.edge_index[0] + g.M).to(device)
    src_edges_dst = (torch.arange(0, g.data.edge_index.size(1))).to(device)

    dst_edges_src = torch.arange(0, g.data.edge_index.size(1)).to(device)
    dst_edges_dst = (g.data.edge_index[1] + g.M + g.N).to(device)

    # add edges from source to edges and from edges to destinations
    src_edges = torch.stack([src_edges_src, src_edges_dst]).to(device)
    dst_edges = torch.stack([dst_edges_src, dst_edges_dst]).to(device)
    edge_index = torch.cat([edge_index, src_edges, dst_edges], dim=1)

    # create sparse scipy matrix
    event_graph = Graph.from_edge_index(edge_index, num_nodes=g.M + 2 * g.N) 
    m = event_graph.get_sparse_adj_matrix()

    #print(f"Created temporal event DAG with {event_graph.N} nodes and {event_graph.M} edges")

    # run disjktra for all source nodes
    dist, pred = dijkstra(m, directed=True, indices=np.arange(g.M, g.M+g.N),  return_predecessors=True, unweighted=True)

    # limit to first-order destinations and correct distances
    dist_fo = dist[:, g.M+g.N:] - 1    
    pred_fo = pred[:, g.N+g.M:]
    np.fill_diagonal(dist_fo, 0)

    return dist_fo, pred_fo
