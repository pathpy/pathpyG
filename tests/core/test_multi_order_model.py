# pylint: disable=missing-function-docstring,missing-module-docstring

import torch
from torch_geometric import EdgeIndex

from pathpyG.core.MultiOrderModel import MultiOrderModel
from pathpyG.core.Graph import Graph
from pathpyG.algorithms.temporal import lift_order_temporal

def test_multi_order_model_init():
    model = MultiOrderModel()
    assert model.layers == {}


def test_multi_order_model_str():
    model = MultiOrderModel()
    assert str(model) == "MultiOrderModel with max. order 0"

    model.layers[1] = "foo"
    assert str(model) == "MultiOrderModel with max. order 1"

    model.layers[5] = "bar"
    assert str(model) == "MultiOrderModel with max. order 5"


def test_multi_order_model_lift_order_edge_index():
    # Inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/test/transforms/test_line_graph.py
    # Directed.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [1, 2, 0, 3, 0],
        ]
    )
    ho_index = MultiOrderModel.lift_order_edge_index(edge_index=edge_index, num_nodes=4)
    assert ho_index.tolist() == [[0, 1, 1, 2, 3, 4], [1, 2, 3, 0, 4, 0]]


def test_lift_order_temporal(simple_temporal_graph):
    edge_index = lift_order_temporal(simple_temporal_graph, delta=5)
    event_graph = Graph.from_edge_index(edge_index)
    assert event_graph.N == simple_temporal_graph.M
    # for delta=5 we have three time-respecting paths (a,b,1) -> (b,c,5), (b,c,5) -> (c,d,9) and (b,c,5) -> (c,e,9)
    assert event_graph.M == 3
    assert torch.equal(event_graph.data.edge_index, EdgeIndex([[0, 1, 1], [1, 2, 3]]))

def test_multi_order_model_from_paths(simple_walks_2):
    m = MultiOrderModel.from_PathData(simple_walks_2, max_order=2)
    g1 = m.layers[1]
    g2 = m.layers[2]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [2, 2, 3, 4]]))
    assert torch.equal(g1.data.edge_weight, torch.tensor([2.0, 2.0, 2.0, 2.0]))
