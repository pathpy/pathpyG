"""Tests for the statistics behind multi-order model selection.

Every torchified statistic is checked against an independent brute-force implementation that
enumerates paths in plain Python, and against closed-form values worked out by hand.
"""

# pylint: disable=missing-function-docstring

import math
from collections import defaultdict

import numpy as np
import pytest
import torch
from scipy.special import gammaln
from scipy.stats import chi2

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData

# --------------------------------------------------------------------------- oracles


def enumerate_occurrences(walks, weights, order):
    """Weighted number of occurrences of each path of `order` nodes, over all positions."""
    occurrences = defaultdict(float)
    for walk, weight in zip(walks, weights):
        for position in range(len(walk) - order + 1):
            occurrences[tuple(walk[position : position + order])] += weight
    return dict(occurrences)


def enumerate_starts(walks, weights, order):
    """Weighted number of observations that *start* with each path of `order` nodes."""
    starts = defaultdict(float)
    for walk, weight in zip(walks, weights):
        if len(walk) >= order:
            starts[tuple(walk[:order])] += weight
    return dict(starts)


def reference_log_likelihood(walks, weights, node_ids, max_order):
    """Multi-order log-likelihood, computed by enumerating paths."""
    transitions = defaultdict(lambda: defaultdict(float))
    visits = defaultdict(float)
    for walk, weight in zip(walks, weights):
        for node in walk:
            visits[node] += weight
        for order in range(1, max_order + 1):
            for position in range(order, len(walk)):
                transitions[tuple(walk[position - order : position])][walk[position]] += weight

    total_visits = sum(visits.values())

    def probability(history, successor):
        counts = transitions[history]
        return counts[successor] / sum(counts.values())

    log_likelihood = 0.0
    for walk, weight in zip(walks, weights):
        if max_order == 0:
            log_likelihood += weight * sum(math.log(visits[node] / total_visits) for node in walk)
            continue
        log_likelihood += weight * math.log(visits[walk[0]] / total_visits)
        for position in range(1, len(walk)):
            order = min(position, max_order)
            log_likelihood += weight * math.log(probability(tuple(walk[position - order : position]), walk[position]))
    return log_likelihood


def reference_log_evidence(walks, weights, node_ids, max_order, dirichlet_prior=1.0, successors=None):
    """Dirichlet-multinomial model evidence, computed by enumerating histories."""
    if successors is None:
        successors = defaultdict(set)
        for walk in walks:
            for source, destination in zip(walk[:-1], walk[1:]):
                successors[source].add(destination)

    counts = defaultdict(lambda: defaultdict(float))
    for walk, weight in zip(walks, weights):
        if max_order == 0:
            for node in walk:
                counts[()][node] += weight
            continue
        counts[()][walk[0]] += weight
        for position in range(1, len(walk)):
            order = min(position, max_order)
            counts[tuple(walk[position - order : position])][walk[position]] += weight

    total = 0.0
    for history, successor_counts in counts.items():
        support = len(node_ids) if history == () else len(successors[history[-1]])
        if support == 0:
            continue
        total += sum(gammaln(dirichlet_prior + count) - gammaln(dirichlet_prior) for count in successor_counts.values())
        total += gammaln(dirichlet_prior * support) - gammaln(
            dirichlet_prior * support + sum(successor_counts.values())
        )
    return total


def reference_degrees_of_freedom(edges, num_nodes, max_order, normalization_constraints="per_history"):
    """Free parameters of a graph-constrained multi-order model, by enumerating paths."""
    successors = defaultdict(list)
    for source, destination in sorted(set(edges)):
        successors[source].append(destination)

    degrees_of_freedom = num_nodes - 1
    histories = [(node,) for node in range(num_nodes)]
    for _ in range(1, max_order + 1):
        extensions = sum(len(successors[history[-1]]) for history in histories)
        if normalization_constraints == "per_history":
            constraints = sum(1 for history in histories if successors[history[-1]])
        else:
            constraints = len({history[0] for history in histories if successors[history[-1]]})
        degrees_of_freedom += extensions - constraints
        histories = [history + (node,) for history in histories for node in successors[history[-1]]]
    return degrees_of_freedom


def build(walks, weights, node_ids, max_order):
    paths = PathData(IndexMap(list(node_ids)))
    paths.append_walks(list(walks), list(weights))
    return MultiOrderModel.from_path_data(paths, max_order=max_order)


TOY_WALKS = [("a", "c", "d"), ("b", "c", "e")]
BRANCHING_WALKS = [("a", "b", "c"), ("a", "b", "c"), ("d", "b", "f"), ("a", "b", "f")]
UNEVEN_WALKS = [("a", "b", "c", "d"), ("a", "b", "c", "e"), ("d", "b", "c", "d")]
SHORT_WALKS = [("a",), ("a", "b"), ("a", "b", "c")]
# Four nodes, the first two steps uniform and the third determined by the first: the first-order
# topology carries no information at all and everything is in the second order.
SECOND_ORDER_ONLY_WALKS = [(first, second, first) for first in range(4) for second in range(4)]

CASES = [
    (TOY_WALKS, [4.0, 4.0], list("abcde"), 2),
    (BRANCHING_WALKS, [2.0, 3.0, 1.0, 4.0], list("abcdef"), 2),
    (UNEVEN_WALKS, [3.0, 1.0, 5.0], list("abcde"), 3),
    (SHORT_WALKS, [1.0, 1.0, 1.0], list("abcde"), 2),
]


# --------------------------------------------------------------- node weight identity


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
def test_node_weight_matches_enumeration(walks, weights, node_ids, max_order):
    model = build(walks, weights, node_ids, max_order)
    for order in range(1, max_order + 1):
        layer = model.layers[order]
        expected = enumerate_occurrences(walks, weights, order)
        observed = {
            layer.node_id(i): float(layer.data.node_weight[i]) for i in range(layer.n) if layer.data.node_weight[i] != 0
        }
        if order == 1:
            expected = {key[0]: value for key, value in expected.items()}
        assert observed == pytest.approx(expected)


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
def test_start_weight_identity(walks, weights, node_ids, max_order):
    """node_weight = start_weight + weighted in-degree, at every order and for every layer."""
    model = build(walks, weights, node_ids, max_order)
    for order in range(1, max_order + 1):
        layer = model.layers[order]
        start_weight = model._start_weight(order)
        incoming = torch.zeros(layer.n, dtype=start_weight.dtype)
        incoming.scatter_add_(0, layer.data.edge_index.as_tensor()[1], layer.data.edge_weight)
        assert torch.allclose(layer.data.node_weight, start_weight + incoming)

        expected = enumerate_starts(walks, weights, order)
        if order == 1:
            expected = {key[0]: value for key, value in expected.items()}
        observed = {layer.node_id(i): float(start_weight[i]) for i in range(layer.n) if start_weight[i] != 0}
        assert observed == pytest.approx(expected)


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
def test_higher_order_nodes_are_lower_order_edges(walks, weights, node_ids, max_order):
    """An order-(k+1) node and an order-k edge are the same sequence of k+1 first-order nodes.

    They are the same *set*, but not necessarily in the same order: `Graph` sorts an edge index by
    source node only, so the edges leaving one node can end up in any order. The likelihood
    therefore matches them by sequence, and must not assume positions line up.
    """
    model = build(walks, weights, node_ids, max_order)
    for order in range(1, max_order):
        edge_sequences = model._edge_sequences(order)
        node_sequences = model.layers[order + 1].data.node_sequence
        assert sorted(map(tuple, edge_sequences.tolist())) == sorted(map(tuple, node_sequences.tolist()))


def test_start_weights_are_matched_by_sequence_not_by_position():
    """Regression test: the two index spaces can genuinely disagree on larger graphs.

    On the small examples above the orders happen to coincide, which would let a buggy
    implementation that simply reuses the positions pass. This graph is large enough that they do
    not, so it exercises the matching.
    """
    from pathpyG.algorithms.temporal_generative_models import temporal_clusters

    graph = temporal_clusters(num_nodes=12, num_edges=60, num_clusters=3, num_walks=800, seed=0)
    model = MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=2)

    edge_sequences = model._edge_sequences(1)
    node_sequences = model.layers[2].data.node_sequence
    assert not torch.equal(edge_sequences, node_sequences)
    assert sorted(map(tuple, edge_sequences.tolist())) == sorted(map(tuple, node_sequences.tolist()))

    # Each layer-1 edge must carry the start weight of the two-gram it actually is.
    start_weight = model._start_weight(2)
    by_sequence = {tuple(sequence): float(weight) for sequence, weight in zip(node_sequences.tolist(), start_weight)}
    on_edges = model._start_weight_on_edges(1)
    for position, sequence in enumerate(edge_sequences.tolist()):
        assert float(on_edges[position]) == pytest.approx(by_sequence[tuple(sequence)])


# ------------------------------------------------------------------- log-likelihood


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
def test_log_likelihood_matches_enumeration(walks, weights, node_ids, max_order):
    model = build(walks, weights, node_ids, max_order)
    for order in range(max_order + 1):
        assert model.log_likelihood(max_order=order) == pytest.approx(
            reference_log_likelihood(walks, weights, node_ids, order)
        )


def test_log_likelihood_closed_form():
    """Hand-computed values for the toy example used throughout the tutorials."""
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    assert model.log_likelihood(max_order=0) == pytest.approx(np.log(1 / 6) * 4 + np.log(2 / 6) * 2)
    assert model.log_likelihood(max_order=1) == pytest.approx(np.log(1 / 6) * 2 + 2 * np.log(1 / 2))
    assert model.log_likelihood(max_order=2) == pytest.approx(np.log(1 / 6) * 2)


def test_log_likelihood_short_paths():
    """Paths shorter than the order contribute only the terms they are long enough for."""
    model = build(SHORT_WALKS, [1.0, 1.0, 1.0], list("abcde"), 2)
    assert model.log_likelihood(max_order=0) == pytest.approx(np.log(3 / 6) * 3 + np.log(2 / 6) * 2 + np.log(1 / 6) * 1)
    assert model.log_likelihood(max_order=1) == pytest.approx(np.log(3 / 6) * 3)
    assert model.log_likelihood(max_order=2) == pytest.approx(np.log(3 / 6) * 3)


def test_intermediate_orders_use_observed_transition_frequencies():
    """Regression test for issue #340.

    The intermediate-order term used to call `transition_probabilities()` without an edge
    attribute, scoring the first transition of each path with uniform instead of
    relative-frequency probabilities. That answered -7.1670 instead of -6.6438 here and biased
    `estimate_order` towards lower orders on data with uneven branching.
    """
    model = build([("a", "b", "c"), ("a", "d", "e")], [3.0, 1.0], list("abcde"), 2)
    expected = 4 * np.log(1 / 3) + 3 * np.log(3 / 4) + 1 * np.log(1 / 4)
    assert model.log_likelihood(max_order=2) == pytest.approx(expected)
    assert model.log_likelihood(max_order=2) == pytest.approx(-6.6438, abs=1e-4)


# --------------------------------------------------------------- degrees of freedom


def test_degrees_of_freedom_known_values():
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    assert [model.degrees_of_freedom(max_order=order) for order in (0, 1, 2)] == [4, 5, 7]


def test_degrees_of_freedom_on_a_line():
    """On a line graph no order adds a free parameter beyond the zeroth order."""
    model = build([("a", "b", "c", "d")], [1.0], list("abcd"), 4)
    assert [model.degrees_of_freedom(max_order=order) for order in range(5)] == [3, 3, 3, 3, 3]


def test_degrees_of_freedom_conventions_differ():
    """The two conventions agree at order one and differ once a node starts several histories.

    On `a->b, a->c, b->d, c->d` the two second-order histories `ab` and `ac` both have exactly one
    successor, so they contribute no free parameter. Counting constraints per first-order node
    instead sees only the single start node `a` and subtracts one too few.
    """
    graph = Graph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    paths = PathData(graph.mapping)
    paths.append_walks([("a", "b", "d"), ("a", "c", "d")], [1.0, 1.0])
    model = MultiOrderModel.from_path_data(paths, max_order=2)

    per_history = [model.degrees_of_freedom(max_order=order) for order in (0, 1, 2)]
    per_node = [
        model.degrees_of_freedom(max_order=order, normalization_constraints="per_first_order_node")
        for order in (0, 1, 2)
    ]
    assert per_history == [3, 4, 4]
    assert per_node == [3, 4, 5]


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
@pytest.mark.parametrize("normalization_constraints", ["per_history", "per_first_order_node"])
def test_degrees_of_freedom_match_enumeration(walks, weights, node_ids, max_order, normalization_constraints):
    model = build(walks, weights, node_ids, max_order)
    edges = [(u, v) for walk in walks for u, v in zip(walk[:-1], walk[1:])]
    mapping = IndexMap(list(node_ids))
    indexed_edges = [(int(mapping.to_idx(u)), int(mapping.to_idx(v))) for u, v in edges]
    for order in range(max_order + 1):
        assert model.degrees_of_freedom(
            max_order=order, normalization_constraints=normalization_constraints
        ) == reference_degrees_of_freedom(
            indexed_edges, len(node_ids), order, normalization_constraints=normalization_constraints
        )


def test_degrees_of_freedom_rejects_unknown_convention():
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="unknown normalization_constraints"):
        model.degrees_of_freedom(normalization_constraints="per_banana")


def test_degrees_of_freedom_rejects_order_above_model():
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="cannot be larger"):
        model.degrees_of_freedom(max_order=3)


# ------------------------------------------------------------- likelihood ratio test


def test_likelihood_ratio_test_matches_hand_computation():
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    log_likelihoods = {order: model.log_likelihood(max_order=order) for order in (0, 1, 2)}
    degrees_of_freedom = {order: model.degrees_of_freedom(max_order=order) for order in (0, 1, 2)}

    for null_order, order in ((0, 1), (1, 2), (0, 2)):
        statistic = -2 * (log_likelihoods[null_order] - log_likelihoods[order])
        expected = float(chi2.sf(statistic, degrees_of_freedom[order] - degrees_of_freedom[null_order]))
        rejected, p_value = model.likelihood_ratio_test(
            null_max_order=null_order, max_order=order, significance_threshold=0.1
        )
        assert p_value == pytest.approx(expected)
        assert rejected == (expected < 0.1)


def test_likelihood_ratio_test_rejects_invalid_orders():
    model = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="must be smaller"):
        model.likelihood_ratio_test(null_max_order=2, max_order=2)
    with pytest.raises(ValueError, match="cannot be larger"):
        model.likelihood_ratio_test(null_max_order=1, max_order=3)


# ------------------------------------------------------------------ Bayesian evidence


@pytest.mark.parametrize("walks,weights,node_ids,max_order", CASES)
@pytest.mark.parametrize("dirichlet_prior", [0.5, 1.0, 2.5])
def test_log_evidence_matches_enumeration(walks, weights, node_ids, max_order, dirichlet_prior):
    model = build(walks, weights, node_ids, max_order)
    for order in range(max_order + 1):
        assert model.log_evidence(max_order=order, dirichlet_prior=dirichlet_prior) == pytest.approx(
            reference_log_evidence(walks, weights, node_ids, order, dirichlet_prior=dirichlet_prior)
        )


def test_log_evidence_ignores_unobserved_histories():
    """A history after which nothing was observed contributes exactly zero, whatever its support."""
    model = build(TOY_WALKS, [4.0, 4.0], list("abcde"), 2)
    # `d` and `e` end every path, so no transition was ever observed after them. Giving them
    # successors enlarges their parameter space but cannot change the evidence.
    with_dead_end_successors = Graph.from_edge_list(
        [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("d", "a"), ("e", "b")]
    )
    assert model.log_evidence(max_order=2, constraint_graph=with_dead_end_successors) == pytest.approx(
        model.log_evidence(max_order=2)
    )

    # Giving an *observed* history another possible successor does change it: `c` was seen to
    # branch two ways, and a third possibility spreads the prior mass further.
    with_extra_branch = Graph.from_edge_list([("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("c", "a")])
    assert model.log_evidence(max_order=2, constraint_graph=with_extra_branch) < model.log_evidence(max_order=2)


def test_bayes_factor_is_a_difference_of_evidences():
    model = build(TOY_WALKS, [4.0, 4.0], list("abcde"), 2)
    assert model.bayes_factor(null_max_order=1, max_order=2) == pytest.approx(
        model.log_evidence(max_order=2) - model.log_evidence(max_order=1)
    )


def test_log_evidence_rejects_invalid_prior():
    model = build(TOY_WALKS, [4.0, 4.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="dirichlet_prior must be positive"):
        model.log_evidence(max_order=1, dirichlet_prior=0.0)


# ----------------------------------------------------------------- order estimation


@pytest.mark.parametrize("weight,expected", [(3.0, 1), (4.0, 2)])
def test_estimate_order_likelihood_ratio(weight, expected):
    model = build(TOY_WALKS, [weight, weight], list("abcde"), 2)
    assert model.estimate_order(max_order=2, significance_threshold=0.01) == expected


def test_estimate_order_bayesian_needs_less_data():
    """At eight observations the Bayes factor already prefers order two; the test does not."""
    model = build(TOY_WALKS, [3.0, 3.0], list("abcde"), 2)
    assert model.estimate_order(max_order=2, method="likelihood_ratio") == 1
    assert model.estimate_order(max_order=2, method="bayesian") == 2


def test_estimate_order_can_return_zero():
    """Too little data to support any structure at all is reported as order zero."""
    model = build(SECOND_ORDER_ONLY_WALKS, [1.0] * len(SECOND_ORDER_ONLY_WALKS), [0, 1, 2, 3], 2)
    assert model.estimate_order(max_order=2, method="likelihood_ratio") == 0
    assert model.estimate_order(max_order=2, method="bayesian") == 0


def test_estimate_order_selection_rules_differ():
    """`stop_at_first_insignificant` gives up where the other two rules keep looking.

    The data is a pure second-order pattern on a uniform first-order topology, so order one
    explains nothing that order zero does not -- the two log-likelihoods are identical -- and the
    improvement appears only at order two.
    """
    model = build(SECOND_ORDER_ONLY_WALKS, [5.0] * len(SECOND_ORDER_ONLY_WALKS), [0, 1, 2, 3], 2)
    assert model.log_likelihood(max_order=1) == pytest.approx(model.log_likelihood(max_order=0))

    assert model.estimate_order(max_order=2, order_selection="largest_significant") == 2
    assert model.estimate_order(max_order=2, order_selection="dominates_all_lower_orders") == 2
    assert model.estimate_order(max_order=2, order_selection="stop_at_first_insignificant") == 0


def test_estimate_order_rejects_arguments_of_the_other_method():
    model = build(TOY_WALKS, [4.0, 4.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="do not apply to method"):
        model.estimate_order(method="likelihood_ratio", dirichlet_prior=2.0)
    with pytest.raises(ValueError, match="do not apply to method"):
        model.estimate_order(method="bayesian", significance_threshold=0.05)


def test_estimate_order_rejects_unknown_names():
    model = build(TOY_WALKS, [4.0, 4.0], list("abcde"), 2)
    with pytest.raises(ValueError, match="unknown method"):
        model.estimate_order(method="vibes")
    with pytest.raises(ValueError, match="unknown order_selection"):
        model.estimate_order(order_selection="whatever")
    with pytest.raises(ValueError, match="cannot be larger"):
        model.estimate_order(max_order=9)


def test_estimate_order_requires_cached_layers():
    """Without the intermediate layers the multi-order likelihood cannot be assembled."""
    paths = PathData(IndexMap(list("abcde")))
    paths.append_walks(TOY_WALKS, [4.0, 4.0])
    model = MultiOrderModel.from_path_data(paths, max_order=3, cached=False)
    assert set(model.layers) == {1, 3}
    with pytest.raises(ValueError, match="cached=True"):
        model.estimate_order(max_order=3)


def test_estimate_order_regression_on_mixed_length_paths():
    """Start indices have to account for paths that are too short to have a higher-order node.

    Regression test for the index arithmetic in the intermediate-order terms.
    """
    walks = [("d", "b", "c"), ("a", "b", "c"), ("a", "b", "e"), ("d", "b", "e"), ("a",)]
    model = build(walks, [1.0, 20.0, 1.0, 20.0, 1.0], list("abcde"), 3)
    assert model.estimate_order(max_order=3) == 2


# ---------------------------------------------------------------- out-of-sample use


def test_log_likelihood_in_sample_equals_explicit_self():
    model = build(BRANCHING_WALKS, [2.0, 3.0, 1.0, 4.0], list("abcdef"), 2)
    for order in (0, 1, 2):
        assert model.log_likelihood(max_order=order, data=model) == pytest.approx(model.log_likelihood(max_order=order))


def test_log_likelihood_scores_another_models_observations():
    """Probabilities come from this model, counts from the other one."""
    train = build(BRANCHING_WALKS, [2.0, 3.0, 1.0, 4.0], list("abcdef"), 2)
    test = build([("a", "b", "c")], [1.0], list("abcdef"), 2)

    # a is visited 9 times out of 30 node visits; a->b is the only transition out of a; and of the
    # 9 occurrences of (a, b), 5 continue to c.
    expected = math.log(9 / 30) + math.log(9 / 9) + math.log(5 / 9)
    assert train.log_likelihood(max_order=2, data=test) == pytest.approx(expected)


def test_log_likelihood_is_minus_infinity_on_unseen_transitions():
    train = build([("a", "b", "c")], [1.0], list("abcdef"), 2)
    test = build([("a", "b", "f")], [1.0], list("abcdef"), 2)
    assert train.log_likelihood(max_order=2, data=test) == -math.inf


def test_log_likelihood_warns_on_mismatched_observation_sets(caplog, long_temporal_graph):
    from_paths = build(TOY_WALKS, [1.0, 1.0], list("abcde"), 2)
    temporal = MultiOrderModel.from_temporal_graph(long_temporal_graph, delta=10, max_order=2)
    with caplog.at_level("WARNING"):
        temporal.log_likelihood(max_order=1, data=from_paths)
    assert "different observation sets" in caplog.text


def test_log_likelihood_rejects_unalignable_node_sets():
    train = build([("a", "b", "c")], [1.0], list("abc"), 2)
    other = build([("x", "y", "z")], [1.0], list("xyz"), 2)
    with pytest.raises(ValueError, match="cannot be aligned"):
        train.log_likelihood(max_order=1, data=other)


# ------------------------------------------------------------------ constraint graph


def test_constraint_graph_increases_degrees_of_freedom():
    """Untraversed edges of the true topology are free parameters the data simply never used."""
    model = build([("a", "b", "c")], [1.0], list("abcd"), 2)
    # The data only ever takes a->b->c, but the underlying graph also allows b->d and c->a.
    known_graph = Graph.from_edge_list([("a", "b"), ("b", "c"), ("b", "d"), ("c", "a")])

    assert model.degrees_of_freedom(max_order=1) == 3
    # Order 1: b now has two successors instead of one, so one extra free parameter; and c has one.
    assert model.degrees_of_freedom(max_order=1, constraint_graph=known_graph) == 4
    assert model.degrees_of_freedom(max_order=2, constraint_graph=known_graph) > model.degrees_of_freedom(max_order=2)


def test_constraint_graph_must_contain_every_observed_transition():
    model = build([("a", "b", "c")], [1.0], list("abc"), 2)
    too_small = Graph.from_edge_list([("a", "b")])
    with pytest.raises(ValueError, match="does not contain"):
        model.degrees_of_freedom(max_order=1, constraint_graph=too_small)


def test_constraint_graph_changes_the_evidence():
    walks = [("a", "b", "c"), ("a", "b", "c"), ("d", "b", "c")]
    model = build(walks, [1.0, 1.0, 1.0], list("abcd"), 2)
    known_graph = Graph.from_edge_list([("a", "b"), ("d", "b"), ("b", "c"), ("b", "d"), ("c", "a")])
    node_ids = list("abcd")
    successors = {"a": {"b"}, "d": {"b"}, "b": {"c", "d"}, "c": {"a"}}
    for order in (0, 1, 2):
        assert model.log_evidence(max_order=order, constraint_graph=known_graph) == pytest.approx(
            reference_log_evidence(walks, [1.0, 1.0, 1.0], node_ids, order, successors=successors)
        )
