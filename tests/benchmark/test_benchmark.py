from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.PathData import PathData
from pathpyG import config
from pathpyG.core.HigherOrderGraph import HigherOrderGraph

def construct_higher_order(max_order):
    paths = PathData.from_csv('docs/data/tube_paths_train.ngram')
    for k in range(1, max_order):
        gk = HigherOrderGraph(paths, order=k, path_freq='path_freq')

def read_path_data():
    paths = PathData.from_csv('docs/data/tube_paths_train.ngram')

@pytest.mark.benchmark
def test_higher_order_gpu(benchmark):

    config['torch']['device'] = 'cuda'
    benchmark.pedantic(construct_higher_order, kwargs={'max_order': 8}, iterations=1, rounds=2)

@pytest.mark.benchmark
def test_higher_order_cpu(benchmark):

    config['torch']['device'] = 'cpu'
    benchmark.pedantic(construct_higher_order, kwargs={'max_order': 8}, iterations=1, rounds=2)

@pytest.mark.benchmark
def test_read_path_data_cpu(benchmark):

    config['torch']['device'] = 'cpu'
    benchmark.pedantic(read_path_data, iterations=5, rounds=2)

@pytest.mark.benchmark
def test_read_path_data_gpu(benchmark):

    config['torch']['device'] = 'cuda'
    benchmark.pedantic(read_path_data, iterations=5, rounds=2)
