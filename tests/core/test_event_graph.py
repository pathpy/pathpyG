from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import dijkstra
from torch_geometric.data import Data

from pathpyG.core.event_graph import EventGraph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.temporal_graph import TemporalGraph

DELTA = 2


# Example temporal graph used in the tests:
#
#       a
#       | t=1
#       v
#       b -------- t=5 --------> d
#       | t=2
#       v
#       c
#       | t=3
#       v
#       e
#
# Event graph corresponding to the above (with DELTA = 2):
#
#   0 --gap 1--> 1 --gap 1--> 2
#   3 (isolated event)
#
#   where
#
#   event 0: (a->b)@1
#   event 1: (b->c)@2
#   event 2: (c->e)@3
#   event 3: (b->d)@5

@pytest.fixture
def temporal_graph() -> TemporalGraph:
    return TemporalGraph.from_edge_list(
        [
            ("a", "b", 1),
            ("b", "c", 2),
            ("b", "d", 5),
            ("c", "e", 3),
        ]
    )


@pytest.fixture
def event_graph(temporal_graph) -> EventGraph:
    return EventGraph.from_temporal_graph(temporal_graph, delta=DELTA)


def test_basic(event_graph):
    """Basic counts (delta, events, nodes) match the source temporal graph."""
    assert event_graph.delta == DELTA
    assert len(event_graph) == 4
    assert event_graph.num_events == 4
    assert event_graph.n == 4
    assert event_graph.num_fo_nodes == 5


def test_str(event_graph):
    """The string representation lists the delta and all events."""
    assert (
        str(event_graph)
        == "EventGraph (delta=2)\na->b@1\nb->c@2\nc->e@3\nb->d@5"
    )


def test_node_time(event_graph):
    """Each event node carries the timestamp of its underlying edge."""
    assert event_graph.data.node_time.tolist() == [1, 2, 3, 5]


def test_node_sequence(event_graph):
    """Each event node stores the (source, target) first-order node pair."""
    assert event_graph.data.node_sequence.tolist() == [[0, 1], [1, 2], [2, 4], [1, 3]]


def test_fo_mapping(event_graph, temporal_graph):
    """The first-order node mapping round-trips and matches the temporal graph."""
    fo = event_graph.fo_mapping
    assert fo.num_ids() == 5
    for node in "abcde":
        assert fo.to_id(fo.to_idx(node)) == node
        assert fo.to_idx(node) == temporal_graph.mapping.to_idx(node)


def test_continuation_edge_index(event_graph):
    """The continuation edge index matches the expected result."""
    got = event_graph.data.edge_index.as_tensor()
    got_set = {tuple(c) for c in got.t().tolist()}
    assert got_set == {(0, 1), (1, 2)}


def test_event_time(event_graph, ):
    """event_time(i) returns the timestamp of the i-th event."""
    assert [event_graph.event_time(i) for i in range(event_graph.num_events)] == [1, 2, 3, 5]


def test_getitem(event_graph):
    """Indexing an EventGraph yields the (u, v, t) tuple for each event."""
    assert event_graph[0] == ("a", "b", 1)
    assert event_graph[1] == ("b", "c", 2)
    assert event_graph[2] == ("c", "e", 3)
    assert event_graph[3] == ("b", "d", 5)


def test_isolated_events(event_graph):
    """Events with no predecessors or successors are correctly identified."""
    isolated = [
        i
        for i in range(event_graph.num_events)
        if event_graph.get_successors(i).numel() == 0 and event_graph.get_predecessors(i).numel() == 0
    ]
    assert isolated == [3]


def test_edge_deltas(event_graph):
    """Every edge_delta lies within (0, delta]."""
    for i in range(event_graph.num_events):
        for nxt in event_graph.get_successors(i):
            nxt = int(nxt.item())
            delta = event_graph.data.edge_delta[event_graph.edge_to_index[(i, nxt)]].item()
            assert 0 < delta <= event_graph.delta


def test_edge_delta(event_graph):
    """Per-edge time deltas match the expected value."""
    got = {
        tuple(c): d
        for c, d in zip(
            event_graph.data.edge_index.as_tensor().t().tolist(),
            event_graph.data.edge_delta.tolist(),
        )
    }
    assert got == {(0, 1): 1, (1, 2): 1}


def test_shortest_paths_distances(event_graph):
    """shortest_paths() distances match the expected result."""
    dist, _pred = event_graph.shortest_paths()
    expected = np.array(
        [
            [0, 1, 2, np.inf, 3],
            [np.inf, 0, 1, 1, 2],
            [np.inf, np.inf, 0, np.inf, 1],
            [np.inf, np.inf, np.inf, 0, np.inf],
            [np.inf, np.inf, np.inf, np.inf, 0],
        ]
    )
    np.testing.assert_array_equal(dist, expected)


def test_shortest_paths_predecessors(event_graph):
    """shortest_paths() predecessors match the expected result."""
    _dist, pred = event_graph.shortest_paths()
    expected = np.array(
        [
            [0, 0, 1, -1, 2],
            [-1, 1, 1, 1, 2],
            [-1, -1, 2, -1, 2],
            [-1, -1, -1, 3, -1],
            [-1, -1, -1, -1, 4],
        ]
    )
    np.testing.assert_array_equal(pred, expected)


def test_shortest_paths_a_to_d_is_unreachable(event_graph):
    """Transition a->d needs a->b@1 then b->d@5, gap of 4 > delta."""
    dist, _pred = event_graph.shortest_paths()
    assert dist[0, 3] == np.inf


def test_fastest_path_distances(event_graph):
    """Fastest-path distances over edge deltas match the expected result."""
    fastest = dijkstra(event_graph.sparse_adj_matrix(edge_attr="edge_delta"), directed=True)
    expected = np.array(
        [
            [0, 1, 2, np.inf],
            [np.inf, 0, 1, np.inf],
            [np.inf, np.inf, 0, np.inf],
            [np.inf, np.inf, np.inf, 0],
        ]
    )
    np.testing.assert_array_equal(fastest, expected)


def test_to_temporal_graph_round_trip(event_graph, temporal_graph):
    """An EventGraph converts back to an equivalent TemporalGraph."""
    rebuilt = event_graph.to_temporal_graph()
    assert isinstance(rebuilt, TemporalGraph)
    assert torch.equal(
        rebuilt.data.edge_index.as_tensor(),
        temporal_graph.data.edge_index.as_tensor(),
    )
    assert torch.equal(rebuilt.data.time, temporal_graph.data.time)
    assert rebuilt.n == temporal_graph.n
    for node in ("a", "b", "c", "d", "e"):
        assert rebuilt.mapping.to_idx(node) == temporal_graph.mapping.to_idx(node)


def test_multi_order_model_construction(event_graph, temporal_graph):
    """A MultiOrderModel built from an EventGraph matches one from a TemporalGraph."""
    with pytest.raises(AttributeError):
        # no such attribute yet
        mom_eg = MultiOrderModel.from_event_graph(event_graph, max_order=2)
        mom_tg = MultiOrderModel.from_temporal_graph(temporal_graph, delta=DELTA, max_order=2)

        for k in (1, 2):
            assert torch.equal(
                mom_eg.layers[k].data.edge_index.as_tensor(),
                mom_tg.layers[k].data.edge_index.as_tensor(),
            )
            assert torch.equal(
                mom_eg.layers[k].data.edge_weight,
                mom_tg.layers[k].data.edge_weight,
            )


def test_to_device(event_graph):
    """Moving an EventGraph moves its underlying TemporalGraph too."""
    moved = event_graph.to(torch.device("cpu"))
    assert isinstance(moved, EventGraph)
    assert moved is event_graph
    assert moved.to_temporal_graph().data.edge_index.device.type == "cpu"


"""
The following tests illustrate that an EventGraph can be constructed from a raw
`torch_geometric.data.Data` object.
"""
@pytest.fixture
def event_data() -> Data:
    return Data(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        num_nodes=4,
        node_sequence=torch.tensor([[0, 1], [1, 2], [2, 4], [1, 3]]),
        node_time=torch.tensor([1, 2, 3, 5]),
    )


def test_construct_from_data(event_data):
    """An EventGraph can be built from a raw Data object with correct edge deltas."""
    eg = EventGraph(event_data, delta=DELTA)
    got = {
        tuple(c): d
        for c, d in zip(
            eg.data.edge_index.as_tensor().t().tolist(), eg.data.edge_delta.tolist()
        )
    }
    assert got == {(0, 1): 1, (1, 2): 1}


def test_construct_from_data_to_temporal_graph(event_data):
    """An EventGraph built from raw Data still converts to a TemporalGraph."""
    fo = IndexMap(["a", "b", "c", "d", "e"])
    eg = EventGraph(event_data, delta=DELTA, fo_mapping=fo, num_fo_nodes=5)
    tg = eg.to_temporal_graph()
    assert isinstance(tg, TemporalGraph)
    assert tg.n == 5
    assert torch.equal(
        tg.data.edge_index.as_tensor(),
        torch.tensor([[0, 1, 2, 1], [1, 2, 4, 3]]),
    )
    assert tg.data.time.tolist() == [1, 2, 3, 5]