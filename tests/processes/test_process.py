"""Tests for the process module."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest
from pathpyG.processes.process import BaseProcess


class TestProcess(BaseProcess):
    def init(self, seed):
        self._seed = seed
        self._time = 0

    def random_seed(self):
        return 42

    def step(self):
        self._time += 1
        return ["node1"]

    @property
    def time(self):
        return self._time

    def state_to_color(self, states):
        return "red"

    def node_state(self, v):
        return "active"


@pytest.fixture
def simple_process(simple_graph):
    return TestProcess(simple_graph)


def test_init(simple_process):
    assert simple_process._seed == 42  # pylint: disable=protected-access


def test_random_seed(simple_process):
    assert simple_process.random_seed() == 42


def test_step(simple_process):
    assert simple_process.step() == ["node1"]
    assert simple_process.time == 1


def test_time(simple_process):
    assert simple_process.time == 0


def test_state_to_color(simple_process):
    assert simple_process.state_to_color("active") == "red"


def test_node_state(simple_process):
    assert simple_process.node_state("node1") == "active"


def test_simulation_run(simple_process):
    steps = 5
    result = list(simple_process.simulation_run(steps))
    assert len(result) == steps
    i = 1
    for time, nodes in result:
        assert time == i
        assert nodes == ["node1"]
        i += 1


def test_run_experiment(simple_process, simple_graph):
    steps = 5
    runs = 2
    df = simple_process.run_experiment(steps, runs)
    assert len(df) == runs * (steps + simple_graph.N)
