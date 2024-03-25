# pylint: disable=missing-function-docstring,missing-module-docstring

import torch
from torch import IntTensor
from pathpyG import MultiOrderModel
from torch_geometric import EdgeIndex


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


def test_lift_order_dag():
    e1 = torch.tensor([[0, 1, 1, 3], [1, 2, 3, 4]])
    x = MultiOrderModel.lift_order_edge_index(e1, num_nodes=5)
    assert torch.equal(x, IntTensor([[0, 0, 2], [1, 2, 3]]))

    e2 = torch.tensor([[0, 0, 2], [1, 2, 3]])
    x = MultiOrderModel.lift_order_edge_index(e2, num_nodes=4)
    assert torch.equal(x, IntTensor([[1], [2]]))

    e3 = torch.tensor([[1], [2]])
    x = MultiOrderModel.lift_order_edge_index(e3, num_nodes=3)
    assert x.size(1) == 0


def test_edge_index_kth_order_dag(simple_dags):
    m = MultiOrderModel.from_DAGs(simple_dags, max_order=2)
    assert torch.equal(m.layers[1].data.edge_index.data, torch.tensor([[0, 0, 1, 1, 2, 2],
           [1, 2, 2, 4, 3, 4]], device=m.layers[1].data.edge_index.device))
    assert torch.equal(m.layers[2].data.edge_index.data, torch.tensor([[0, 1, 1, 2, 2],
           [3, 4, 5, 4, 5]], device=m.layers[2].data.edge_index.device))


# TODO: 
def test_edge_index_temporal(simple_temporal_graph):
    # dag = temporal_graph_to_event_dag(simple_temporal_graph, delta=5, sparsify=True)
    # paths = DAGData.from_temporal_dag(dag)

    # e1, w1 = DAGData.edge_index_k_weighted(paths, k=1)

    # assert torch.equal(
    #     e1, IntTensor([[0, 1, 2, 2], [1, 2, 3, 4]]).to(config["torch"]["device"])
    # )  # a -> b | b -> c | c -> d | c -> e

    # assert torch.equal(w1, tensor([1.0, 1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    # e2, w2 = DAGData.edge_index_k_weighted(paths, k=2)
    # assert torch.equal(
    #     e2,
    #     IntTensor([[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [2, 4]]]).to(
    #         config["torch"]["device"]
    #     ),
    # )  # a-b -> b-c | b-c -> c-d | b-c -> c-e

    # assert torch.equal(w2, tensor([1.0, 1.0, 1.0]).to(config["torch"]["device"]))

    # e3, w3 = DAGData.edge_index_k_weighted(paths, k=3)
    # assert torch.equal(
    #     e3,
    #     IntTensor([[[0, 1, 2], [0, 1, 2]], [[1, 2, 3], [1, 2, 4]]]).to(
    #         config["torch"]["device"]
    #     ),
    # )

    # assert torch.equal(
    #     w3, tensor([1.0, 1.0]).to(config["torch"]["device"])
    # )  # a-b-c -> b-c-d | a-b-c -> b-c-e
    pass
