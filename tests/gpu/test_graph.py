# pylint: disable=missing-function-docstring,missing-module-docstring
# pylint: disable=invalid-name

from __future__ import annotations

import pytest
import scipy.sparse as s
import torch
from torch_geometric.edge_index import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.testing import get_random_edge_index

from pathpyG import Graph, IndexMap


@pytest.mark.gpu
def test_init(gpu):
    edge_index = get_random_edge_index(100, 100, 1000).to(gpu)
    data = Data(edge_index=edge_index, num_nodes=100)
    g = Graph(data)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_init_with_edge_index(gpu):
    edge_index = EdgeIndex(get_random_edge_index(100, 100, 1000)).to(gpu)
    data = Data(edge_index=edge_index, num_nodes=100)
    g = Graph(data)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_init_with_mapping(gpu):
    edge_index = get_random_edge_index(100, 100, 1000).to(gpu)
    data = Data(edge_index=edge_index, num_nodes=100)
    idxs = [str(i) for i in range(100)]
    mapping = IndexMap(idxs)
    g = Graph(data, mapping)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_from_edge_index(gpu):
    edge_index = get_random_edge_index(100, 100, 1000).to(gpu)
    g = Graph.from_edge_index(edge_index)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_from_edge_list(gpu):
    edge_list = [
        ("a", "b"),
        ("c", "a"),
        ("b", "c"),
    ]
    g = Graph.from_edge_list(edge_list, device=gpu)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_from_edge_list_undirected(gpu):
    edge_list = [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
        ("c", "a"),
        ("a", "c"),
    ]
    g = Graph.from_edge_list(edge_list, is_undirected=True, device=gpu)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu


@pytest.mark.gpu
def test_to_undirected(simple_graph, gpu):
    simple_graph.to(gpu)

    g_u = simple_graph.to_undirected()
    assert g_u.data.is_undirected()

    assert g_u.data.edge_index.device == gpu
    assert g_u.row.device == gpu
    assert g_u.row_ptr.device == gpu
    assert g_u.col.device == gpu
    assert g_u.col_ptr.device == gpu


@pytest.mark.gpu
def test_to_device(simple_graph, gpu, cpu):
    g = simple_graph.to(gpu)
    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu

    g.to(cpu)
    assert g.data.edge_index.device == cpu
    assert g.row.device == cpu
    assert g.row_ptr.device == cpu
    assert g.col.device == cpu
    assert g.col_ptr.device == cpu


@pytest.mark.gpu
def test_add_operator_partial_overlap(gpu):
    # partial overlap
    g1 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "c", "d"])).to(gpu)
    g2 = Graph.from_edge_index(torch.IntTensor([[0, 1, 1], [1, 2, 3]]), mapping=IndexMap(["a", "b", "g", "h"])).to(gpu)
    g = g1 + g2
    assert g.N == 6
    assert g.M == g1.M + g2.M

    # we need to sort because the order may vary when merged on GPU
    assert torch.equal(
        g.data.edge_index.sort_by("col")[0], torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 2, 3, 4, 5]], device=gpu)
    )

    assert g.data.edge_index.device == gpu
    assert g.row.device == gpu
    assert g.row_ptr.device == gpu
    assert g.col.device == gpu
    assert g.col_ptr.device == gpu
