"""Base classes for simulation of dynamical processes"""

import abc
from typing import Iterable, Any, Tuple

from torch import Tensor

from pathpyG import Graph


class BaseProcess:
    """Abstract base class for all implementations of discrete-time dynamical processes."""

    def __init__(self, graph: Graph):
        """initialize process."""
        self._graph = graph

    @property
    def graph(self) -> Graph:
        """Returns the graph on which the process is simulated."""
        return self._graph

    @abc.abstractmethod
    def state_to_color(self, states: Any) -> Tuple[int, int, int] | str:
        """Abstract method mapping node states to RGB colors or color names."""

    @abc.abstractmethod
    def run_experiment(self, steps: int, runs: int | Iterable | Tensor | None = 1) -> Tensor:
        """Perform one or more simulation runs of the process with a given number of steps."""
