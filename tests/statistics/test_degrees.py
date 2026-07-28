import numpy as _np
import pytest
import torch

from pathpyG.statistics.degrees import (
    degree_assortativity,
    degree_central_moment,
    degree_distribution,
    degree_generating_function,
    degree_raw_moment,
    degree_sequence,
    mean_degree,
    mean_neighbor_degree,
)


def test_degree_sequence_undirected(simple_graph):
    seq = degree_sequence(simple_graph)
    assert (seq == torch.tensor([1.0, 3.0, 2.0, 2.0, 2.0])).all()


def test_degree_sequence_directed(toy_example_graph_directed):
    seq_in = degree_sequence(toy_example_graph_directed, mode="in")
    assert (seq_in == torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0])).all()
    seq_out = degree_sequence(toy_example_graph_directed, mode="out")
    assert (seq_out == torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0])).all()
    seq_total = degree_sequence(toy_example_graph_directed, mode="total")
    assert (seq_total == torch.tensor([2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0])).all()


def test_degree_distribution(simple_graph):
    dist = degree_distribution(simple_graph)
    assert dist[0] == 0.0
    assert dist[1] == 1 / 5
    assert dist[2] == 3 / 5
    assert dist[3] == 1 / 5


def test_degree_distribution_directed(toy_example_graph_directed):
    dist_in = degree_distribution(toy_example_graph_directed, mode="in")
    assert dist_in[0] == 0.0
    assert dist_in[1] == 5 / 7
    assert dist_in[2] == 2 / 7

    dist_out = degree_distribution(toy_example_graph_directed, mode="out")
    assert dist_out[0] == 0.0
    assert dist_out[1] == 5 / 7
    assert dist_out[2] == 2 / 7

    dist_total = degree_distribution(toy_example_graph_directed, mode="total")
    assert dist_total[0] == 0.0
    assert dist_total[2] == 4 / 7
    assert dist_total[3] == 2 / 7
    assert dist_total[4] == 1 / 7


def test_degree_raw_moment(simple_graph):
    k_1 = degree_raw_moment(simple_graph, k=1)
    assert k_1 == 2.0
    k_2 = degree_raw_moment(simple_graph, k=2)
    assert _np.isclose(k_2, 4.4)
    k_3 = degree_raw_moment(simple_graph, k=3)
    assert _np.isclose(k_3, 10.4)


def test_mean_degree(toy_example_graph_directed):
    seq_total = torch.tensor([2.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0])
    mean_deg = mean_degree(toy_example_graph_directed)
    assert mean_deg == seq_total.mean().item()
    mean_deg = mean_degree(toy_example_graph_directed.to_undirected())
    assert mean_deg == seq_total.mean().item()


def test_mean_neighbor_degree(simple_graph):
    mnd = mean_neighbor_degree(simple_graph)
    assert _np.isclose(mnd, 2.2)
    mnd_excl = mean_neighbor_degree(simple_graph, exclude_backlink=True)
    assert _np.isclose(mnd_excl, 1.2)


def test_degree_central_moment(simple_graph):
    k_1 = degree_central_moment(simple_graph, k=1)
    assert k_1 == 0.0
    k_2 = degree_central_moment(simple_graph, k=2)
    assert _np.isclose(k_2, 0.4)
    k_3 = degree_central_moment(simple_graph, k=3)
    assert _np.isclose(k_3, 0.0)


def test_degree_assortativity(toy_example_graph):
    assert _np.isclose(degree_assortativity(toy_example_graph), -0.26, atol=1e-2)


def test_degree_generating_function(simple_graph):
    y = degree_generating_function(simple_graph, x=0.5)
    assert isinstance(y, float)
    assert _np.isclose(y, 0.275)
    y = degree_generating_function(simple_graph, x=_np.array([0, 0.5, 1.0]))
    assert (torch.isclose(y, torch.tensor([0, 0.275, 1.0]))).all()
    y = degree_generating_function(simple_graph, x=[0, 0.5, 1.0])
    assert (torch.isclose(y, torch.tensor([0, 0.275, 1.0]))).all()
    y = degree_generating_function(simple_graph, x=torch.tensor([0, 0.5, 1.0]))
    assert (torch.isclose(y, torch.tensor([0, 0.275, 1.0]))).all()
    with pytest.raises(TypeError):
        degree_generating_function(simple_graph, x="invalid_type")