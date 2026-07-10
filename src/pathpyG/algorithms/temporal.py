"""Algorithms for the analysis of time-respecting paths in temporal graphs."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from pathpyG import Graph
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils import to_numpy


def temporal_shortest_paths(g: TemporalGraph | None, delta: int, eg: EventGraph | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute shortest time-respecting paths in a temporal graph.

    Args:
        g: Temporal graph to compute shortest paths on. If None, `eg` must be provided.
        delta: Maximum time difference between events in a path.
        eg: Event graph to compute shortest paths on. If None, `g` must be provided.

    Returns:
        Tuple of two numpy arrays:
        - dist: Shortest time-respecting path distances between all first-order nodes.
        - pred: Predecessor matrix for shortest time-respecting paths between all first-order nodes.
    """
    assert g is None or eg is None, "Only one of g or eg can be provided"
    if g is None:
        assert eg is not None, "If g is None, eg must be provided"
        edge_index = eg.data.edge_index
        g = eg.to_temporal_graph()
    else:
        # generate temporal event DAG
        edge_index = EventGraph.build_edge_index(g, delta)

    # Add indices of first-order nodes as src and dst of paths in augmented
    # temporal event DAG
    src_edges_src = g.data.edge_index[0] + g.m
    src_edges_dst = torch.arange(0, g.data.edge_index.size(1), device=g.data.edge_index.device)

    dst_edges_src = torch.arange(0, g.data.edge_index.size(1), device=g.data.edge_index.device)
    dst_edges_dst = g.data.edge_index[1] + g.m + g.n

    # add edges from source to edges and from edges to destinations
    src_edges = torch.stack([src_edges_src, src_edges_dst])
    dst_edges = torch.stack([dst_edges_src, dst_edges_dst])
    edge_index = torch.cat([edge_index, src_edges, dst_edges], dim=1)

    # create sparse scipy matrix
    event_graph = Graph.from_edge_index(edge_index, num_nodes=g.m + 2 * g.n)
    m = event_graph.sparse_adj_matrix()

    # print(f"Created temporal event DAG with {event_graph.n} nodes and {event_graph.m} edges")

    # run disjktra for all source nodes
    dist, pred = dijkstra(
        m, directed=True, indices=np.arange(g.m, g.m + g.n), return_predecessors=True, unweighted=True
    )

    # limit to first-order destinations and correct distances
    dist_fo = dist[:, g.m + g.n :] - 1
    np.fill_diagonal(dist_fo, 0)

    # limit to first-order destinations and correct predecessors
    pred_fo = pred[:, g.n + g.m :]
    pred_fo[pred_fo == -9999] = -1
    idx_map = np.concatenate([to_numpy(g.data.edge_index[0].cpu()), [-1]])
    pred_fo = idx_map[pred_fo]
    np.fill_diagonal(pred_fo, np.arange(g.n))

    return dist_fo, pred_fo
