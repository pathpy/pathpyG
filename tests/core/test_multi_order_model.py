# pylint: disable=missing-function-docstring,missing-module-docstring

import torch
from torch_geometric import EdgeIndex

import numpy as np
from scipy.stats import chi2

from pathpyG.core.index_map import IndexMap
from pathpyG.core.path_data import PathData
from pathpyG.core.multi_order_model import MultiOrderModel


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


def test_iterate_lift_order(simple_graph_multi_edges):
    ho_index, node_sequence, edge_weight, gk = MultiOrderModel.iterate_lift_order(
        edge_index=simple_graph_multi_edges.data.edge_index,
        node_sequence=torch.arange(simple_graph_multi_edges.n).unsqueeze(1),
        mapping=simple_graph_multi_edges.mapping,
        save=True,
    )
    assert ho_index.tolist() == [[0, 2], [3, 3]]
    assert node_sequence.tolist() == [[0, 1], [0, 2], [0, 1], [1, 2]]
    assert edge_weight is None
    assert gk.data.edge_index.as_tensor().tolist() == [[0], [2]]
    assert gk.data.node_sequence.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert gk.data.edge_weight.tolist() == [2.0]
    assert gk.order == 2


def test_dof():
    line_data = PathData(IndexMap(list("abcd")))
    line_data.append_walk(("a", "b", "c", "d"))
    max_order = 4
    m = MultiOrderModel.from_path_data(line_data, max_order=max_order)
    for order in range(max_order + 1):
        assert m.get_mon_dof(assumption="paths", max_order=order) == 3

    #########

    toy_paths_ho = PathData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    max_order = 2
    m = MultiOrderModel.from_path_data(toy_paths_ho, max_order=max_order, mode="propagation")
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

    toy_paths_ho = PathData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    dag_graph = toy_paths_ho.data
    max_order = 2
    m = MultiOrderModel.from_path_data(toy_paths_ho, max_order=max_order)

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
    toy_paths_ho = PathData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"))
    toy_paths_ho.append_walk(("b", "c", "e"))
    max_order = 2
    m = MultiOrderModel.from_path_data(toy_paths_ho, max_order=max_order, mode="propagation")
    dag_graph = toy_paths_ho.data
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=0), np.log(1 / 6) * 4 + np.log(2 / 6) * 2)
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=1), np.log(1 / 6) * 2 + 0 + 2 * np.log(1 / 2))
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=2), np.log(1 / 6) * 2 + 0 + 0)
    ################

    toy_paths = PathData(IndexMap(list("abcde")))
    toy_paths.append_walk(("a", "c", "d"))
    toy_paths.append_walk(("b", "c", "e"))
    toy_paths.append_walk(("a", "c", "e"))
    toy_paths.append_walk(("b", "c", "d"))
    max_order = 2
    m = MultiOrderModel.from_path_data(toy_paths, max_order=max_order, mode="propagation")
    dag_graph = toy_paths.data
    assert np.isclose(
        m.get_mon_log_likelihood(dag_graph, max_order=0),  # fails already at computing log_lh here
        np.log(2 / 12) * 8 + np.log(4 / 12) * 4,
    )
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=1), np.log(2 / 12) * 4 + 0 + 4 * np.log(1 / 2))
    assert np.isclose(m.get_mon_log_likelihood(dag_graph, max_order=2), np.log(1 / 6) * 4 + 0 + 4 * np.log(1 / 2))

    ################

    toy_paths = PathData(IndexMap(list("abcde")))
    toy_paths.append_walk(("a",))
    toy_paths.append_walk(("a", "b"))
    toy_paths.append_walk(("a", "b", "c"))
    max_order = 2
    m = MultiOrderModel.from_path_data(toy_paths, max_order=max_order, mode="propagation")
    dag_graph = toy_paths.data
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

    toy_paths_ho = PathData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"), weight=3)
    toy_paths_ho.append_walk(("b", "c", "e"), weight=3)
    m = MultiOrderModel.from_path_data(toy_paths_ho, max_order=2)
    assert m.estimate_order(toy_paths_ho, max_order=max_order, significance_threshold=significance_threshold) == 1

    toy_paths_ho = PathData(IndexMap(list("abcde")))
    toy_paths_ho.append_walk(("a", "c", "d"), weight=4)
    toy_paths_ho.append_walk(("b", "c", "e"), weight=4)
    m = MultiOrderModel.from_path_data(toy_paths_ho, max_order=max_order)
    assert m.estimate_order(toy_paths_ho, max_order=2, significance_threshold=significance_threshold) == 2


def test_multi_order_model_from_paths(simple_walks_2):
    m = MultiOrderModel.from_path_data(simple_walks_2, max_order=2)
    g1 = m.layers[1]
    g2 = m.layers[2]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [2, 2, 3, 4]]))
    assert torch.equal(g1.data.edge_weight, torch.tensor([2.0, 2.0, 2.0, 2.0]))

    assert torch.equal(g2.data.edge_index, EdgeIndex([[0, 1], [2, 3]]))
    assert torch.equal(g2.data.edge_weight, torch.tensor([2.0, 2.0]))


def test_multi_order_from_temporal_graph(simple_temporal_graph):
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4)
    g1 = m.layers[1]
    g2 = m.layers[2]
    g3 = m.layers[3]
    assert torch.equal(g1.data.edge_index, EdgeIndex([[0, 1, 2, 2], [1, 2, 3, 4]]))
    assert torch.equal(g2.data.edge_index, EdgeIndex([[0, 1, 1], [1, 2, 3]]))
    assert torch.equal(g3.data.edge_index, EdgeIndex([[0, 0], [1, 2]]))


def test_to_DBGNN_data(simple_temporal_graph):
    m = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=3, delta=4)
    data = m.to_dbgnn_data(max_order=3)
    assert torch.equal(data.edge_index, EdgeIndex([[0, 1, 2, 2], [1, 2, 3, 4]]))
    assert torch.equal(data.edge_index_higher_order, EdgeIndex([[0, 0], [1, 2]]))


def test_paths_indexing():
    """Indexing test.
    
    This test was create to test that start indexes (ixs_start_paths_ho) of paths in 
    get_intermediate_order_log_likelihood work correcly. Paths 'shrink' when encoded 
    throgh higher-order nodes, and ixs_start_paths_ho has to correctly account for it.
    """
    paths_list = [
        ("d", "b", "c"),
        ("a", "b", "c"),
        ("a", "b", "e"),
        ("d", "b", "e"),
        ("a",)
        ]
    frequencies = [
        1,
        20,
        1,
        20,
        1
        ]
    mapping = IndexMap()
    mapping.add_ids(np.unique(np.hstack(paths_list)))
    pathdata = PathData(mapping)
    pathdata.append_walks(node_seqs=paths_list, weights=frequencies)
    max_order = 3
    mon = MultiOrderModel.from_path_data(pathdata, max_order=max_order)
    detected_order = mon.estimate_order(
        pathdata,
        max_order=max_order
        )
    assert detected_order == 2
