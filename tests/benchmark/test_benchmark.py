from __future__ import annotations

import pytest

from pathpyG import config
from pathpyG.algorithms.temporal_generative_models import temporal_clusters
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.io.pandas import read_csv_path_data

# to run benchmarks, do the following:
# > pip install pytest-benchmark
# > run pytest -m benchmark


def higher_order_paths(max_order):
    paths = read_csv_path_data("docs/data/tube_paths_train.ngram", weight=True, sep=",")
    MultiOrderModel.from_path_data(paths, max_order=max_order)


def higher_order_temporal_graph(max_order):
    graph = temporal_clusters(seed=0)
    MultiOrderModel.from_temporal_graph(graph, delta=1, max_order=max_order)


@pytest.mark.benchmark
def test_higher_order_paths_gpu(benchmark):

    config["torch"]["device"] = "cuda"
    benchmark.pedantic(higher_order_paths, kwargs={"max_order": 10}, iterations=1, rounds=2)


@pytest.mark.benchmark
def test_higher_order_paths_cpu(benchmark):

    config["torch"]["device"] = "cpu"
    benchmark.pedantic(higher_order_paths, kwargs={"max_order": 10}, iterations=1, rounds=2)


@pytest.mark.benchmark
def test_higher_order_temporal_graph_gpu(benchmark):

    config["torch"]["device"] = "cuda"
    benchmark.pedantic(higher_order_temporal_graph, kwargs={"max_order": 5}, iterations=1, rounds=10)


@pytest.mark.benchmark
def test_higher_order_temporal_graph_cpu(benchmark):

    config["torch"]["device"] = "cpu"
    benchmark.pedantic(higher_order_temporal_graph, kwargs={"max_order": 5}, iterations=1, rounds=10)
