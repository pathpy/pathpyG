from __future__ import annotations

import pytest

from pathpyG import config
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph

# to run benchmarks, do the following:
# > pip install pytest-benchmark
# > run pytest -m benchmark


def higher_order_paths(max_order):
    paths = PathData.from_ngram("docs/data/tube_paths_train.ngram")
    MultiOrderModel.from_PathData(paths, max_order=max_order)


def higher_order_temporal_graph(max_order):
    t = TemporalGraph.from_csv("docs/data/ants_1_1.tedges")
    MultiOrderModel.from_temporal_graph(t, delta=30, max_order=max_order)


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
