# pylint: disable=missing-function-docstring,missing-module-docstring

import numpy as np
import torch
from scipy.stats import chi2
from torch_geometric import EdgeIndex

from pathpyG.algorithms.temporal import extract_causal_paths
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph


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


def test_estimate_order_from_temporal_graph_first_order():
    # A temporal graph whose time-respecting paths carry no second-order correlations:
    # after b, both c and d follow regardless of whether a or x came first.
    tedges = []
    for i, (first, second) in enumerate([("a", "c"), ("a", "d"), ("x", "c"), ("x", "d")] * 3):
        t = 3 * i
        tedges += [(first, "b", t), ("b", second, t + 1)]
    g = TemporalGraph.from_edge_list(tedges)

    paths = extract_causal_paths(g, delta=1)
    m = MultiOrderModel.from_path_data(paths, max_order=2)
    assert m.estimate_order(paths, max_order=2) == 1


def test_estimate_order_from_temporal_graph_second_order():
    # Same topology, but now the continuation is fully determined by the predecessor:
    # a is always followed by c and x always by d, so order 2 is warranted.
    tedges = []
    for i, (first, second) in enumerate([("a", "c"), ("x", "d")] * 12):
        t = 3 * i
        tedges += [(first, "b", t), ("b", second, t + 1)]
    g = TemporalGraph.from_edge_list(tedges)

    paths = extract_causal_paths(g, delta=1)
    m = MultiOrderModel.from_path_data(paths, max_order=2)
    assert m.estimate_order(paths, max_order=2) == 2


def layer_edge_weights(m, k) -> dict:
    """Map each De Bruijn edge of layer k to its weight, keyed by first-order node sequences."""
    g = m.layers[k]
    node_seq = g.data.node_sequence
    edge_index = g.data.edge_index.as_tensor() if hasattr(g.data.edge_index, "as_tensor") else g.data.edge_index
    return {
        (tuple(node_seq[s].tolist()), tuple(node_seq[d].tolist())): w
        for s, d, w in zip(edge_index[0].tolist(), edge_index[1].tolist(), g.data.edge_weight.tolist())
    }


def test_temporal_and_extracted_path_layers_share_topology(simple_temporal_graph):
    # Both constructions must discover exactly the same set of higher-order De Bruijn edges,
    # since every time-respecting walk lies on at least one maximal time-respecting path.
    delta = 4
    max_order = 3
    m_temporal = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=max_order, delta=delta)
    m_paths = MultiOrderModel.from_path_data(
        extract_causal_paths(simple_temporal_graph, delta=delta), max_order=max_order
    )

    for k in range(1, max_order + 1):
        assert layer_edge_weights(m_temporal, k).keys() == layer_edge_weights(m_paths, k).keys(), f"layer {k} topology"


def test_temporal_and_extracted_path_layers_use_different_weights(simple_temporal_graph):
    """Document that the two constructions count *different* things.

    `from_temporal_graph` weights a layer-k edge by the number of time-respecting walks of k
    events that realize it. Building from extracted maximal paths instead weights it by the
    number of maximal paths containing such a walk, so a walk that branches downstream is
    counted once per branch. The two agree only where no branching or merging occurs.
    """
    delta = 4
    m_temporal = MultiOrderModel.from_temporal_graph(simple_temporal_graph, max_order=2, delta=delta)
    m_paths = MultiOrderModel.from_path_data(extract_causal_paths(simple_temporal_graph, delta=delta), max_order=2)
    a, b, c, d, e = (simple_temporal_graph.mapping.to_idx(x) for x in "abcde")

    # One event each, regardless of how many paths traverse it.
    assert layer_edge_weights(m_temporal, 1) == {((a,), (b,)): 1, ((b,), (c,)): 1, ((c,), (d,)): 1, ((c,), (e,)): 1}
    # a->b and b->c lie on both maximal paths (a,b,c,d) and (a,b,c,e), so they count twice.
    assert layer_edge_weights(m_paths, 1) == {((a,), (b,)): 2, ((b,), (c,)): 2, ((c,), (d,)): 1, ((c,), (e,)): 1}

    assert layer_edge_weights(m_temporal, 2) == {((a, b), (b, c)): 1, ((b, c), (c, d)): 1, ((b, c), (c, e)): 1}
    assert layer_edge_weights(m_paths, 2) == {((a, b), (b, c)): 2, ((b, c), (c, d)): 1, ((b, c), (c, e)): 1}


def test_zeroth_order_emission_uses_path_weights():
    """P0 must be estimated from weight-weighted node occurrences, not raw instance counts.

    Previously `get_zeroth_order_log_likelihood` counted node instances with `torch.unique`,
    ignoring `dag_weight`, while the max_order=0 branch of `get_mon_log_likelihood` used a
    weighted bincount -- so the same P0 was estimated two different ways in one class.
    """
    paths = PathData(IndexMap(list("abcd")))
    paths.append_walk(("a", "b", "c"), weight=1.0)
    paths.append_walk(("d",), weight=9.0)
    m = MultiOrderModel.from_path_data(paths, max_order=1)

    # Weighted occurrences: a,b,c once each, d nine times -> total 12.
    expected = 1.0 * np.log(1 / 12) + 9.0 * np.log(9 / 12)
    assert np.isclose(m.get_zeroth_order_log_likelihood(), expected)
    # Unweighted counts would have given every node 1/4.
    assert not np.isclose(m.get_zeroth_order_log_likelihood(), 10.0 * np.log(1 / 4))

    # Both estimators of P0 must now agree.
    expected_zeroth_model = (1 * np.log(1 / 12)) * 3 + 9 * np.log(9 / 12)
    assert np.isclose(m.get_mon_log_likelihood(max_order=0), expected_zeroth_model)


def test_intermediate_order_uses_weighted_transition_probabilities():
    """Intermediate orders must use the fitted (weight-weighted) transition probabilities.

    Previously this called `transition_probabilities()` with no `edge_attr`, giving each
    out-edge equal probability, while the max-order term used `edge_attr="edge_weight"`.
    """
    paths = PathData(IndexMap(list("abcdz")))
    paths.append_walk(("a", "b", "c", "z"), weight=1.0)
    paths.append_walk(("a", "b", "d", "z"), weight=9.0)
    m = MultiOrderModel.from_path_data(paths, max_order=3)

    # From (a,b) the walk continues to (b,c) with weight 1 and to (b,d) with weight 9.
    expected = 1.0 * np.log(1 / 10) + 9.0 * np.log(9 / 10)
    assert np.isclose(m.get_intermediate_order_log_likelihood(order=2), expected)
    # Treating both continuations as equally likely would have given 10*log(1/2).
    assert not np.isclose(m.get_intermediate_order_log_likelihood(order=2), 10.0 * np.log(1 / 2))


def test_ho_node_and_lower_order_edge_orderings_coincide():
    """An order-(k+1) node is an order-k edge, and the two index spaces must line up elementwise.

    `get_intermediate_order_log_likelihood` multiplies `layers[k+1].data.path_start_weight`
    directly against `layers[k].transition_probabilities()`, which is only valid because
    `aggregate_edge_index` sorts higher-order node sequences and coalesced edges into the same
    lexicographic order.
    """
    paths_list = [("d", "b", "c", "a"), ("a", "b", "c", "d"), ("a", "b", "e", "c"), ("d", "b", "e", "a")]
    mapping = IndexMap()
    mapping.add_ids(np.unique(np.hstack(paths_list)))
    pathdata = PathData(mapping)
    pathdata.append_walks(node_seqs=paths_list, weights=[1, 20, 1, 20])
    max_order = 3
    m = MultiOrderModel.from_path_data(pathdata, max_order=max_order)

    for k in range(1, max_order):
        node_seq_lower = m.layers[k].data.node_sequence
        edge_index = m.layers[k].data.edge_index.as_tensor()
        # The order-(k+1) node sequence implied by each order-k edge, in edge order.
        implied = torch.cat([node_seq_lower[edge_index[0]], node_seq_lower[edge_index[1]][:, -1:]], dim=1)
        assert torch.equal(implied, m.layers[k + 1].data.node_sequence), f"order {k} ordering mismatch"


def test_path_statistics_attached_to_layers(simple_walks_2):
    m = MultiOrderModel.from_path_data(simple_walks_2, max_order=2)
    a, b, c = (simple_walks_2.mapping.to_idx(x) for x in ("A", "B", "C"))

    # Two walks A->C->D and B->C->E with weight 2 each.
    start = m.layers[1].data.path_start_weight
    assert start[a].item() == 2.0
    assert start[b].item() == 2.0
    assert start[c].item() == 0.0
    assert start.sum().item() == 4.0

    # Six node instances in total, weighted by 2: C occurs in both walks.
    instances = m.layers[1].data.node_instance_weight
    assert instances[c].item() == 4.0
    assert instances.sum().item() == 12.0

    # Each walk contributes exactly one first second-order node, so the order-2 starts also
    # sum to the total path weight.
    assert m.layers[2].data.path_start_weight.sum().item() == 4.0


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
