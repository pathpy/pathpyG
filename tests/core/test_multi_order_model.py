# pylint: disable=missing-function-docstring,missing-module-docstring

import torch
import numpy as np
from torch import IntTensor
from pathpyG import MultiOrderModel, DAGData, IndexMap
from scipy.stats import chi2
from torch_geometric import EdgeIndex
from torch_geometric.loader import DataLoader


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
    assert torch.equal(
        m.layers[1].data.edge_index.data,
        torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 2, 4, 3, 4]], device=m.layers[1].data.edge_index.device),
    )
    assert torch.equal(
        m.layers[2].data.edge_index.data,
        torch.tensor([[0, 1, 1, 2, 2], [3, 4, 5, 4, 5]], device=m.layers[2].data.edge_index.device),
    )


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


def test_dof():
    line_data = DAGData(IndexMap(list("abcd")))
    line_data.append_walk(("a", "b", "c", "d"))
    max_order = 4
    m = MultiOrderModel.from_DAGs(line_data, max_order=max_order)
    for order in range(max_order + 1):
        assert m.get_mon_dof(assumption="paths", max_order=order) == 3

    #########

    toy_paths_ho = DAGData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    max_order = 2
    m = MultiOrderModel.from_DAGs(toy_paths_ho, max_order=max_order, mode="propagation")
    assert m.get_mon_dof(assumption="paths", max_order=0) == 4
    assert m.get_mon_dof(assumption="paths", max_order=1) == 5
    assert m.get_mon_dof(assumption="paths", max_order=2) == 7


def test_likelihood_ratio_test():
    significance_threshold = 0.1

    llh_zeroth = np.log(1 / 6) * 4 + np.log(2 / 6) * 2
    llh_first = np.log(1 / 6) * 2 + 0 + 2 * np.log(1 / 2)
    llh_second = np.log(1 / 6) * 2 + 0 + 0
    dof_zeroth = 4
    dof_first = 5
    dof_second = 7
    x_01 = -2 * (llh_zeroth - llh_first)
    x_12 = -2 * (llh_first - llh_second)
    dof_diff_01 = dof_first - dof_zeroth
    dof_diff_12 = dof_second - dof_first
    p_01 = 1 - chi2.cdf(x_01, dof_diff_01)
    p_12 = 1 - chi2.cdf(x_12, dof_diff_12)

    toy_paths_ho = DAGData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    dag_graph = next(
        iter(DataLoader(toy_paths_ho.dags, batch_size=len(toy_paths_ho.dags)))
    )  # .to(pp.config["torch"]["device"])
    max_order = 2
    m = MultiOrderModel.from_DAGs(toy_paths_ho, max_order=max_order)

    bool_code_01, p_01_code = m.likelihood_ratio_test(
        dag_graph, max_order_null=0, max_order=1, assumption="paths", significance_threshold=significance_threshold
    )

    bool_code_12, p_12_code = m.likelihood_ratio_test(
        dag_graph, max_order_null=1, max_order=2, assumption="paths", significance_threshold=significance_threshold
    )

    assert bool_code_01 == (p_01 < significance_threshold)
    assert np.isclose(p_01_code, p_01)
    assert bool_code_12 == (p_12 < significance_threshold)
    assert np.isclose(p_12_code, p_12)

def test_log_likelihood():
    toy_paths_ho = DAGData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    max_order = 2
    m = MultiOrderModel.from_DAGs(toy_paths_ho, max_order=max_order, mode="propagation")
    dag_graph = next(iter(DataLoader(toy_paths_ho.dags, batch_size=len(toy_paths_ho.dags))))#.to(pp.config["torch"]["device"])
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=0), np.log(1 / 6) * 4 + np.log(2 / 6) * 2)
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=1), np.log(1 / 6) * 2 + 0 + 2 * np.log(1 / 2))
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=2), np.log(1 / 6) * 2 + 0 + 0)
    ################

    toy_paths = DAGData(IndexMap(list("abcde")))
    toy_paths.append_walk(("a", "c", "d"))
    toy_paths.append_walk(("b", "c", "e"))
    toy_paths.append_walk(("a", "c", "e"))
    toy_paths.append_walk(("b", "c", "d"))
    max_order = 2
    m = MultiOrderModel.from_DAGs(toy_paths, max_order=max_order, mode="propagation")
    dag_graph = next(
        iter(DataLoader(toy_paths.dags, batch_size=len(toy_paths.dags)))
    )  # .to(pp.config["torch"]["device"])
    assert np.isclose(
        m.get_mon_log_likelihood(dag_graph, max_order=0),  # fails already at computing log_lh here
        np.log(2 / 12) * 8 + np.log(4 / 12) * 4,
    )
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=1), np.log(2 / 12) * 4 + 0 + 4 * np.log(1 / 2))
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=2), np.log(1 / 6) * 4 + 0 + 4 * np.log(1 / 2))

    ################

    toy_paths = DAGData(IndexMap(list("abcde")))
    toy_paths.append_walk(("a",))
    toy_paths.append_walk(("a", "b"))
    toy_paths.append_walk(("a", "b", "c"))
    max_order = 2
    m = MultiOrderModel.from_DAGs(toy_paths, max_order=max_order, mode="propagation")
    dag_graph = next(
        iter(DataLoader(toy_paths.dags, batch_size=len(toy_paths.dags)))
    )  # .to(pp.config["torch"]["device"])
    assert np.isclose(
        m.get_mon_log_likelihood(dag_graph, max_order=0),  # fails already at computing log_lh here
        np.log(3 / 6) * 3 + np.log(2 / 6) * 2 + np.log(1 / 6) * 1,
    )
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=1), np.log(3 / 6) * 3 + 0 + 0)
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=2), np.log(3 / 6) * 3 + 0 + 0)


def test_estimate_order():
    significance_threshold = 0.01
    max_order = 2

    ###

    toy_paths_ho = DAGData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"), weight=3)
    toy_paths_ho.append_walk(("b", "c", "e"), weight=3)
    m = MultiOrderModel.from_DAGs(toy_paths_ho, max_order=2)
    assert m.estimate_order(toy_paths_ho, max_order=max_order, significance_threshold=significance_threshold) == 1

    toy_paths_ho = DAGData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"), weight=4)
    toy_paths_ho.append_walk(("b", "c", "e"), weight=4)
    m = MultiOrderModel.from_DAGs(toy_paths_ho, max_order=max_order)
    assert m.estimate_order(toy_paths_ho, max_order=2, significance_threshold=significance_threshold) == 2
