"""Algorithms for the analysis of time-respecting paths in temporal graphs."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import cumsum

from pathpyG.core.event_graph import EventGraph
from pathpyG.core.graph import Graph
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.utils import to_numpy

logger = logging.getLogger("root")


def temporal_shortest_paths(
    temporal_graph: TemporalGraph | None, delta: int, event_graph: EventGraph | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute shortest time-respecting paths in a temporal graph.

    Args:
        temporal_graph: Temporal graph to compute shortest paths on. If None, `eg` must be provided.
        delta: Maximum time difference between events in a path.
        event_graph: Event graph to compute shortest paths on. If None, `g` must be provided.

    Returns:
        Tuple of two numpy arrays:
        - dist: Shortest time-respecting path distances between all first-order nodes.
        - pred: Predecessor matrix for shortest time-respecting paths between all first-order nodes.
    """
    assert temporal_graph is None or event_graph is None, "Only one of temporal_graph or event_graph can be provided"
    if temporal_graph is None:
        assert event_graph is not None, "If temporal_graph is None, event_graph must be provided"
        edge_index = event_graph.data.edge_index
        temporal_graph = event_graph.to_temporal_graph()
    else:
        # generate temporal event DAG
        edge_index = EventGraph.build_edge_index(temporal_graph, delta)

    # Add indices of first-order nodes as src and dst of paths in augmented
    # temporal event DAG
    src_edges_src = temporal_graph.data.edge_index[0] + temporal_graph.m
    src_edges_dst = torch.arange(
        0, temporal_graph.data.edge_index.size(1), device=temporal_graph.data.edge_index.device
    )

    dst_edges_src = torch.arange(
        0, temporal_graph.data.edge_index.size(1), device=temporal_graph.data.edge_index.device
    )
    dst_edges_dst = temporal_graph.data.edge_index[1] + temporal_graph.m + temporal_graph.n

    # add edges from source to edges and from edges to destinations
    src_edges = torch.stack([src_edges_src, src_edges_dst])
    dst_edges = torch.stack([dst_edges_src, dst_edges_dst])
    edge_index = torch.cat([edge_index, src_edges, dst_edges], dim=1)

    # create sparse scipy matrix
    augmented_dag = Graph.from_edge_index(edge_index, num_nodes=temporal_graph.m + 2 * temporal_graph.n)
    m = augmented_dag.sparse_adj_matrix()

    # print(f"Created temporal event DAG with {augmented_dag.n} nodes and {augmented_dag.m} edges")

    # run disjktra for all source nodes
    dist, pred = dijkstra(
        m,
        directed=True,
        indices=np.arange(temporal_graph.m, temporal_graph.m + temporal_graph.n),
        return_predecessors=True,
        unweighted=True,
    )

    # limit to first-order destinations and correct distances
    dist_fo = dist[:, temporal_graph.m + temporal_graph.n :] - 1
    np.fill_diagonal(dist_fo, 0)

    # limit to first-order destinations and correct predecessors
    pred_fo = pred[:, temporal_graph.n + temporal_graph.m :]
    pred_fo[pred_fo == -9999] = -1
    idx_map = np.concatenate([to_numpy(temporal_graph.data.edge_index[0].cpu()), [-1]])
    pred_fo = idx_map[pred_fo]
    np.fill_diagonal(pred_fo, np.arange(temporal_graph.n))

    return dist_fo, pred_fo


def walk_counts(
    event_graph: torch.Tensor,
    num_events: int,
    max_length: int,
    reverse: bool = False,
) -> torch.Tensor:
    r"""Count the time-respecting walks of each length that continue after (or precede) each event.

    This is a depth-bounded dynamic program over the event graph. Writing $N_m(e)$ for the number of
    time-respecting walks of $m$ further events that start at event $e$, and $\mathcal{S}(e)$ for the
    events that may follow $e$,

    $$
    N_0(e) = 1, \qquad N_m(e) = \sum_{s \in \mathcal{S}(e)} N_{m-1}(s),
    $$

    so each level is a single scatter-add along the event graph and the cost is
    $\mathcal{O}(\text{max\_length} \cdot |E_{\text{event}}|)$. No walk is ever materialised, which is
    what makes multi-order statistics for temporal graphs tractable.

    With `reverse=True` the recursion runs along reversed edges instead, giving the number of walks of
    $m$ events that *end* at each event.

    Note: Numerics
        The counts grow like the branching factor to the power of `max_length`, so they are computed
        in `float64`. Very long walks on dense event graphs can still overflow; see
        [`MultiOrderModel.from_temporal_graph`][pathpyG.core.multi_order_model.MultiOrderModel.from_temporal_graph], which
        checks for this.

    Args:
        event_graph: Edge index of the event graph, i.e. of the second-order lift of the temporal
            graph, of shape `(2, num_event_graph_edges)`.
        num_events: Number of events, i.e. of time-stamped edges of the temporal graph.
        max_length: Largest number of further events to count.
        reverse: Count preceding instead of following walks.

    Returns:
        Tensor of shape `(max_length + 1, num_events)` whose row `m` holds $N_m$.

    Examples:
        >>> import pathpyG as pp
        >>> from pathpyG.algorithms.temporal import walk_counts
        >>> t = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 3)])
        >>> event_graph = pp.EventGraph.build_edge_index(t, delta=1)
        >>> print(walk_counts(event_graph, num_events=3, max_length=2).tolist())
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    """
    device = event_graph.device
    counts = torch.ones((max_length + 1, num_events), dtype=torch.float64, device=device)
    if max_length == 0:
        return counts

    if event_graph.numel() == 0:
        # Without any event graph edge no event is followed (or preceded) by anything.
        counts[1:] = 0.0
        return counts

    # `accumulate_into` is the event whose count we are building up, `read_from` the neighbour
    # whose shorter-walk count it sums.
    accumulate_into, read_from = (event_graph[1], event_graph[0]) if reverse else (event_graph[0], event_graph[1])
    for length in range(1, max_length + 1):
        level = torch.zeros(num_events, dtype=torch.float64, device=device)
        level.scatter_add_(0, accumulate_into, counts[length - 1][read_from])
        counts[length] = level
    return counts


def _event_graph_successors(
    event_graph: torch.Tensor, num_events: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a CSR view of the event graph: row pointers, successors and out-degrees."""
    device = event_graph.device
    if event_graph.numel() == 0:
        out_degree = torch.zeros(num_events, dtype=torch.long, device=device)
        return cumsum(out_degree), torch.empty(0, dtype=torch.long, device=device), out_degree
    order = torch.argsort(event_graph[0])
    sources = event_graph[0][order]
    successors = event_graph[1][order]
    out_degree = torch.bincount(sources, minlength=num_events)
    return cumsum(out_degree), successors, out_degree


def _extend_walks(
    walks: torch.Tensor,
    last_event: torch.Tensor,
    row_pointer: torch.Tensor,
    successors: torch.Tensor,
    out_degree: torch.Tensor,
    event_destination: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extend every walk by one event, in a single vectorised step.

    Args:
        walks: Node sequences of the current walks, of shape `(num_walks, num_walk_nodes)`.
        last_event: Index of the last event of each walk.
        row_pointer: CSR row pointers of the event graph.
        successors: CSR successor list of the event graph.
        out_degree: Out-degree of each event in the event graph.
        event_destination: Destination first-order node of each event.

    Returns:
        The extended walks, the index of their new last event, and the number of extensions each
        input walk produced (zero for walks that cannot be extended).
    """
    num_extensions = out_degree[last_event]
    total = int(num_extensions.sum().item())
    device = walks.device
    if total == 0:
        return (
            walks.new_empty((0, walks.size(1) + 1)),
            last_event.new_empty(0),
            num_extensions,
        )
    parent = torch.repeat_interleave(torch.arange(walks.size(0), device=device), num_extensions)
    # Offset of each extension within its parent's block of successors.
    offset = torch.arange(total, device=device) - torch.repeat_interleave(cumsum(num_extensions)[:-1], num_extensions)
    chosen = successors[row_pointer[last_event][parent] + offset]
    extended = torch.cat([walks[parent], event_destination[chosen].unsqueeze(1)], dim=1)
    return extended, chosen, num_extensions


def _initial_walks(
    graph: TemporalGraph, delta: float | int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the one-event walks of a temporal graph together with a CSR view of its event graph."""
    data = graph.data if graph.data.is_sorted_by_time() else graph.data.sort_by_time()
    edge_index = data.edge_index
    flat_edge_index = edge_index.as_tensor() if hasattr(edge_index, "as_tensor") else edge_index
    num_events = flat_edge_index.size(1)
    event_graph = EventGraph.build_edge_index(graph, delta)
    row_pointer, successors, out_degree = _event_graph_successors(event_graph, num_events)
    walks = flat_edge_index.t().contiguous()
    last_event = torch.arange(num_events, device=flat_edge_index.device)
    return walks, last_event, row_pointer, successors, out_degree, flat_edge_index[1]


def extract_time_respecting_walks(graph: TemporalGraph, delta: float | int = 1, length: int = 1) -> PathData:
    """Extract every time-respecting walk of exactly `length` events.

    A time-respecting walk is a sequence of time-stamped edges in which each edge starts where the
    previous one ended and follows it by at most `delta`. The extraction is a sequence of gathers
    along the event graph, one per event of the walk, so no Python-level traversal is involved.

    Warning:
        The number of such walks grows with the branching factor of the event graph, and this
        function materialises all of them. Use
        [`walk_counts`][pathpyG.algorithms.walk_counts] if only their number is needed, and
        [`MultiOrderModel.from_temporal_graph`][pathpyG.core.multi_order_model.MultiOrderModel.from_temporal_graph] if
        only multi-order statistics are needed.

    Args:
        graph: The temporal graph.
        delta: Maximum time difference between two consecutive events of a walk.
        length: Number of events each extracted walk has. A walk of `length` events visits
            `length + 1` nodes.

    Returns:
        PathData: The extracted walks, each with weight one.

    Examples:
        >>> import pathpyG as pp
        >>> t = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 3)])
        >>> walks = pp.algorithms.extract_time_respecting_walks(t, delta=1, length=2)
        >>> print(sorted(walks.get_walk(i) for i in range(walks.num_paths)))
        [('a', 'b', 'c'), ('b', 'c', 'd')]
    """
    if length < 1:
        logger.error("length must be at least 1, got %s", length)
        raise ValueError(f"length must be at least 1, got {length}")

    walks, last_event, row_pointer, successors, out_degree, event_destination = _initial_walks(graph, delta)
    for _ in range(length - 1):
        walks, last_event, _ = _extend_walks(walks, last_event, row_pointer, successors, out_degree, event_destination)
        if walks.size(0) == 0:
            break

    return PathData.from_walks(walks, mapping=graph.mapping)


def extract_causal_paths(graph: TemporalGraph, delta: float | int = 1, max_length: int | None = None) -> PathData:
    """Extract the maximal time-respecting paths of a temporal graph.

    A time-respecting walk is *maximal* when it can be extended in neither direction: its first
    event has no admissible predecessor and its last event has no admissible successor. The
    extraction expands all partial walks one event at a time, emitting those that cannot grow
    further, so there is one vectorised step per length rather than one per walk.

    Warning:
        The number of maximal paths can grow exponentially with the branching factor of the event
        graph. Pass `max_length` to bound the work; walks that are still extendable at that length
        are emitted as they are, and a warning is logged.

    Args:
        graph: The temporal graph.
        delta: Maximum time difference between two consecutive events of a path.
        max_length: Largest number of events to expand to. Unbounded if None.

    Returns:
        PathData: The maximal time-respecting paths, each with weight one.

    Examples:
        >>> import pathpyG as pp
        >>> t = pp.TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 9)])
        >>> paths = pp.algorithms.extract_causal_paths(t, delta=1)
        >>> print(sorted(paths.get_walk(i) for i in range(paths.num_paths)))
        [('a', 'b', 'c'), ('c', 'd')]
    """
    walks, last_event, row_pointer, successors, out_degree, event_destination = _initial_walks(graph, delta)

    # A maximal path cannot be extended backwards, so it starts at an event without predecessor.
    num_events = last_event.size(0)
    has_predecessor = torch.zeros(num_events, dtype=torch.bool, device=walks.device)
    if successors.numel() > 0:
        has_predecessor[successors] = True
    keep = ~has_predecessor
    walks, last_event = walks[keep], last_event[keep]

    completed: list[torch.Tensor] = []
    num_expansions = 0
    while walks.size(0) > 0:
        if max_length is not None and num_expansions >= max_length - 1:
            logger.warning(
                "stopped expanding time-respecting paths at max_length=%s; %s paths may be extendable further",
                max_length,
                walks.size(0),
            )
            completed.append(walks)
            break
        extended, new_last_event, num_extensions = _extend_walks(
            walks, last_event, row_pointer, successors, out_degree, event_destination
        )
        maximal = num_extensions == 0
        if bool(maximal.any()):
            completed.append(walks[maximal])
        walks, last_event = extended, new_last_event
        num_expansions += 1

    if not completed:
        return PathData(mapping=graph.mapping)

    flat = torch.cat([block.reshape(-1) for block in completed])
    num_nodes_per_walk = torch.cat(
        [torch.full((block.size(0),), block.size(1), dtype=torch.long, device=block.device) for block in completed]
    )
    return PathData.from_walks(flat, num_nodes_per_walk=num_nodes_per_walk, mapping=graph.mapping)
