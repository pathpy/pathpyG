from __future__ import annotations

import numpy as np
import pytest

from pathpyG.algorithms.temporal import (
    extract_causal_paths,
    extract_time_respecting_walks,
    temporal_shortest_paths,
    walk_counts,
)
from pathpyG.core.event_graph import EventGraph
from pathpyG.core.temporal_graph import TemporalGraph


def test_temporal_shortest_paths(long_temporal_graph):
    dist, pred = temporal_shortest_paths(long_temporal_graph, delta=10)
    assert dist.shape == (long_temporal_graph.n, long_temporal_graph.n)
    assert pred.shape == (long_temporal_graph.n, long_temporal_graph.n)

    true_dist = np.array(
        [
            [0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 2.0, float("inf")],
            [3.0, 0.0, 1.0, 2.0, 2.0, 1.0, 4.0, 5.0, 1.0],
            [2.0, float("inf"), 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
            ],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
            ],
            [1.0, float("inf"), float("inf"), float("inf"), float("inf"), 0.0, 2.0, 1.0, float("inf")],
            [
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
                1.0,
                float("inf"),
            ],
            [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 1.0, float("inf"), 0.0, 1.0],
            [
                float("inf"),
                1.0,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                0.0,
            ],
        ]
    )
    assert np.allclose(dist, true_dist, equal_nan=True)

    true_pred = np.array(
        [
            [0, 0, 0, 2, 2, 2, 0, 2, -1],
            [5, 1, 1, 2, 2, 1, 0, 6, 1],
            [5, -1, 2, 2, 2, 2, 0, 2, 2],
            [-1, -1, -1, 3, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 4, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, 5, 0, 5, -1],
            [-1, -1, -1, -1, -1, -1, 6, 6, -1],
            [-1, -1, -1, -1, -1, 7, -1, 7, 7],
            [-1, 8, -1, -1, -1, -1, -1, -1, 8],
        ]
    )
    assert np.allclose(pred, true_pred)


# --------------------------------------------------------------- time-respecting walks


def enumerate_walks(graph: TemporalGraph, delta, length):
    """Enumerate every time-respecting walk of `length` events, by plain recursion."""
    events = [
        (int(source), int(destination), int(time))
        for source, destination, time in zip(graph.data.edge_index[0], graph.data.edge_index[1], graph.data.time)
    ]

    def extend(prefix):
        if len(prefix) == length:
            return [prefix]
        last = events[prefix[-1]]
        found = []
        for index, event in enumerate(events):
            if event[0] == last[1] and last[2] < event[2] <= last[2] + delta:
                found.extend(extend(prefix + [index]))
        return found

    walks = []
    for index in range(len(events)):
        walks.extend(extend([index]))
    return [tuple([events[i][0] for i in walk] + [events[walk[-1]][1]]) for walk in walks]


def enumerate_maximal_paths(graph: TemporalGraph, delta):
    """Enumerate the time-respecting walks that can be extended in neither direction."""
    events = [
        (int(source), int(destination), int(time))
        for source, destination, time in zip(graph.data.edge_index[0], graph.data.edge_index[1], graph.data.time)
    ]

    def successors(index):
        last = events[index]
        return [i for i, event in enumerate(events) if event[0] == last[1] and last[2] < event[2] <= last[2] + delta]

    has_predecessor = {index for i in range(len(events)) for index in successors(i)}

    def extend(prefix):
        following = successors(prefix[-1])
        if not following:
            return [prefix]
        found = []
        for index in following:
            found.extend(extend(prefix + [index]))
        return found

    paths = []
    for index in range(len(events)):
        if index not in has_predecessor:
            paths.extend(extend([index]))
    return [tuple([events[i][0] for i in path] + [events[path[-1]][1]]) for path in paths]


def walk_ids(path_data):
    return sorted(path_data.get_walk(i) for i in range(path_data.num_paths))


@pytest.mark.parametrize("delta", [1, 4, 10])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_walk_counts_match_enumeration(long_temporal_graph, delta, length):
    """The dynamic program must count exactly the walks an explicit enumeration finds."""
    event_graph = EventGraph.build_edge_index(long_temporal_graph, delta)
    num_events = long_temporal_graph.m
    following = walk_counts(event_graph, num_events, length)
    assert float(following[length].sum()) == len(enumerate_walks(long_temporal_graph, delta, length + 1))


@pytest.mark.parametrize("delta", [1, 4, 10])
@pytest.mark.parametrize("length", [1, 2, 3])
def test_walk_counts_reverse_matches_forward_total(long_temporal_graph, delta, length):
    """Counting backwards finds the same walks, only anchored at the other end."""
    event_graph = EventGraph.build_edge_index(long_temporal_graph, delta)
    num_events = long_temporal_graph.m
    following = walk_counts(event_graph, num_events, length)
    preceding = walk_counts(event_graph, num_events, length, reverse=True)
    assert float(following[length].sum()) == pytest.approx(float(preceding[length].sum()))


def test_walk_counts_on_an_empty_event_graph():
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("c", "d", 99)])
    counts = walk_counts(EventGraph.build_edge_index(graph, 1), num_events=2, max_length=3)
    assert counts[0].tolist() == [1.0, 1.0]
    assert counts[1].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("delta", [1, 4, 10])
@pytest.mark.parametrize("length", [1, 2, 3])
def test_extract_time_respecting_walks_matches_enumeration(long_temporal_graph, delta, length):
    walks = extract_time_respecting_walks(long_temporal_graph, delta=delta, length=length)
    expected = enumerate_walks(long_temporal_graph, delta, length)
    mapping = long_temporal_graph.mapping
    expected_ids = sorted(tuple(mapping.to_ids(list(walk)).tolist()) for walk in expected)
    assert walk_ids(walks) == expected_ids


def test_extract_time_respecting_walks_rejects_invalid_length(simple_temporal_graph):
    with pytest.raises(ValueError, match="length must be at least 1"):
        extract_time_respecting_walks(simple_temporal_graph, delta=1, length=0)


def test_extract_time_respecting_walks_branching():
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("b", "d", 2), ("c", "e", 3)])
    assert walk_ids(extract_time_respecting_walks(graph, delta=1, length=2)) == [
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("b", "c", "e"),
    ]


def test_extract_time_respecting_walks_respects_delta():
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 5)])
    assert walk_ids(extract_time_respecting_walks(graph, delta=4, length=2)) == [("a", "b", "c")]
    assert walk_ids(extract_time_respecting_walks(graph, delta=3, length=2)) == []


@pytest.mark.parametrize("delta", [1, 4, 10])
def test_extract_causal_paths_matches_enumeration(long_temporal_graph, delta):
    paths = extract_causal_paths(long_temporal_graph, delta=delta)
    expected = enumerate_maximal_paths(long_temporal_graph, delta)
    mapping = long_temporal_graph.mapping
    expected_ids = sorted(tuple(mapping.to_ids(list(path)).tolist()) for path in expected)
    assert walk_ids(paths) == expected_ids


def test_extract_causal_paths_splits_at_a_delta_gap():
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 9)])
    assert walk_ids(extract_causal_paths(graph, delta=1)) == [("a", "b", "c"), ("c", "d")]


def test_extract_causal_paths_honours_max_length(caplog):
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("b", "c", 2), ("c", "d", 3), ("d", "e", 4)])
    with caplog.at_level("WARNING"):
        paths = extract_causal_paths(graph, delta=1, max_length=2)
    assert walk_ids(paths) == [("a", "b", "c")]
    assert "max_length" in caplog.text


def test_extract_causal_paths_on_isolated_events():
    graph = TemporalGraph.from_edge_list([("a", "b", 1), ("c", "d", 99)])
    assert walk_ids(extract_causal_paths(graph, delta=1)) == [("a", "b"), ("c", "d")]
