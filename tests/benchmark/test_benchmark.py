from __future__ import annotations

import pytest
import torch

from pathpyG.core.Graph import Graph
from pathpyG.core.DAGData import DAGData
from pathpyG import config
from pathpyG.core.MultiOrderModel import MultiOrderModel

def construct_higher_order(max_order):
    dags = DAGData.from_ngram('docs/data/tube_paths_train.ngram')
    m = MultiOrderModel.from_DAGs(dags, max_order=10)

@pytest.mark.benchmark
def test_higher_order_gpu(benchmark):

    config['torch']['device'] = 'cuda'
    benchmark.pedantic(construct_higher_order, kwargs={'max_order': 10}, iterations=1, rounds=2)

@pytest.mark.benchmark
def test_higher_order_cpu(benchmark):

    config['torch']['device'] = 'cpu'
    benchmark.pedantic(construct_higher_order, kwargs={'max_order': 10}, iterations=1, rounds=2)
