"""Algorithms for the analysis of time-respecting paths in temporal graphs."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from pathpyG import Graph
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils import to_numpy


def lift_order_temporal(g: TemporalGraph, delta: float | int = 1):
    """Lift a temporal graph to a second-order temporal event graph.

    Args:
        g: Temporal graph to lift.
        delta: Maximum time difference between events to consider them connected.

    Returns:
        ho_index: Edge index of the second-order temporal event graph.
    """
    # first-order edge index
    edge_index, timestamps = g.data.edge_index, g.data.time

    delta = torch.tensor(delta, device=edge_index.device)  # type: ignore[assignment]
    indices = torch.arange(0, edge_index.size(1), device=edge_index.device)

    unique_t = torch.unique(timestamps, sorted=True)
    second_order = []

    # lift order: find possible continuations for edges in each time stamp
    for t in tqdm(unique_t):
        # find indices of all source edges that occur at unique timestamp t
        src_time_mask = timestamps == t
        src_edge_idx = indices[src_time_mask]

        # find indices of all edges that can possibly continue edges occurring at time t for the given delta
        dst_time_mask = (timestamps > t) & (timestamps <= t + delta)
        dst_edge_idx = indices[dst_time_mask]

        if dst_edge_idx.size(0) > 0 and src_edge_idx.size(0) > 0:
            # compute second-order edges between src and dst idx
            # for all edges where dst in src_edges (edge_index[1, x[:, 0]]) matches src in dst_edges (edge_index[0, x[:, 1]])
            x = torch.cartesian_prod(src_edge_idx, dst_edge_idx)
            ho_edge_index = x[edge_index[1, x[:, 0]] == edge_index[0, x[:, 1]]]
            second_order.append(ho_edge_index)

    if not second_order:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    ho_index = torch.cat(second_order, dim=0).t().contiguous()
    return ho_index


def extract_causal_paths(g: TemporalGraph, delta: float | int = 1, max_paths: Optional[int] = None) -> PathData:
    """Enumerate maximal time-respecting paths in a temporal graph into a [PathData][pathpyG.PathData] object.

    A maximal time-respecting path is a sequence of temporal edges `(e_1, ..., e_L)` such that
    each `e_{i+1}` continues `e_i` within `delta` (as defined by [lift_order_temporal][pathpyG.algorithms.lift_order_temporal]),
    `e_1` has no time-respecting predecessor edge, and `e_L` has no time-respecting successor edge.
    If an edge has more than one valid predecessor or successor, it is part of more than one
    maximal path, so the returned [PathData][pathpyG.PathData] can contain overlapping paths
    that share individual edges.

    Warning:
        The number of maximal time-respecting paths can grow exponentially in the branching
        factor of the temporal graph (an edge with `b` valid continuations multiplies the
        number of paths passing through it by `b`). This function enumerates paths explicitly
        and is intended as a **reference implementation for small graphs** - e.g. to validate
        higher-order model likelihoods computed directly from the event graph - and not as a
        scalable path extraction routine. For large or densely connected temporal graphs, build
        a [MultiOrderModel][pathpyG.MultiOrderModel] directly via
        [MultiOrderModel.from_temporal_graph][pathpyG.MultiOrderModel.from_temporal_graph] instead,
        which never materializes individual paths. Use `max_paths` to fail fast rather than
        exhausting memory on inputs that are too large.

    Args:
        g: The temporal graph to extract paths from.
        delta: The maximum time difference between two consecutive edges in a path.
        max_paths: If set, raise a `RuntimeError` once more than this many maximal paths have
            been found.

    Returns:
        PathData: A [PathData][pathpyG.PathData] object containing all maximal time-respecting
            paths, each stored with weight 1.0.

    Examples:
        >>> import pathpyG as pp
        >>> g = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 5), ("c", "d", 9), ("c", "e", 9)])
        >>> paths = pp.algorithms.extract_causal_paths(g, delta=4)
        >>> sorted(paths.get_walk(i) for i in range(paths.num_paths))
        [('a', 'b', 'c', 'd'), ('a', 'b', 'c', 'e')]
    """
    num_events = g.data.edge_index.size(1)
    paths = PathData(mapping=g.mapping, device=g.data.edge_index.device)
    if num_events == 0:
        return paths

    ho_index = lift_order_temporal(g, delta)
    src = g.data.edge_index[0].tolist()
    dst = g.data.edge_index[1].tolist()

    successors: list[list[int]] = [[] for _ in range(num_events)]
    has_predecessor = [False] * num_events
    for s, d in ho_index.t().tolist():
        successors[s].append(d)
        has_predecessor[d] = True

    roots = [i for i in range(num_events) if not has_predecessor[i]]

    node_seqs: list[list[int]] = []
    # DFS over the event graph from each root to each reachable leaf. The event graph is
    # acyclic because continuations strictly increase in time, so this always terminates.
    stack: list[tuple[int, list[int]]] = [(root, [src[root], dst[root]]) for root in roots]
    while stack:
        event, seq = stack.pop()
        succs = successors[event]
        if not succs:
            node_seqs.append(seq)
            if max_paths is not None and len(node_seqs) > max_paths:
                raise RuntimeError(
                    f"Number of maximal time-respecting paths exceeds max_paths={max_paths}. "
                    "extract_causal_paths does not scale to graphs with a high branching factor; "
                    "consider building a MultiOrderModel directly from the temporal graph instead."
                )
        else:
            for succ in succs:
                stack.append((succ, seq + [dst[succ]]))

    node_id_seqs = [g.mapping.to_ids(seq) for seq in node_seqs]
    paths.append_walks(node_id_seqs, weights=[1.0] * len(node_seqs))
    return paths


def temporal_shortest_paths(g: TemporalGraph, delta: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute shortest time-respecting paths in a temporal graph.

    Args:
        g: Temporal graph to compute shortest paths on.
        delta: Maximum time difference between events in a path.

    Returns:
        Tuple of two numpy arrays:
        - dist: Shortest time-respecting path distances between all first-order nodes.
        - pred: Predecessor matrix for shortest time-respecting paths between all first-order nodes.
    """
    # generate temporal event DAG
    edge_index = lift_order_temporal(g, delta)

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
