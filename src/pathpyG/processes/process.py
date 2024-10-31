"""Base classes for simulation of dynamical processes"""

import abc
from typing import Generator, Iterable, Any, Optional, Tuple, Union

from pandas import DataFrame

from pathpyG import Graph
from tqdm import tqdm


class BaseProcess:
    """Abstract base class for all implementations of discrete-time dynamical processes."""

    def __init__(self, graph: Graph):
        """initialize process."""
        self._graph = graph
        self.init(self.random_seed())

    @property
    def graph(self) -> Graph:
        """Returns the graph on which the process is simulated."""
        return self._graph

    @abc.abstractmethod
    def init(self, seed: Any) -> None:
        """Abstract method to initialize the process with a given seed state."""

    @abc.abstractmethod
    def random_seed(self) -> Any:
        """Abstract method to generate a random seed state for the process."""

    @abc.abstractmethod
    def step(self) -> Iterable[str]:
        """Abstract method to simulate a single step of the process. Returns
        an iterable of node uids whose state has been changed in this step."""

    @property
    @abc.abstractmethod
    def time(self) -> int:
        """Abstract property returning the current time."""

    @abc.abstractmethod
    def state_to_color(self, states: Any) -> Union[Tuple[int, int, int], str]:
        """Abstract method mapping node states to RGB colors or color names."""

    @abc.abstractmethod
    def node_state(self, v: str | int) -> Any:
        """Abstract method returning the current state of a given node."""

    def simulation_run(
        self, steps: int, seed: Optional[Any] = None
    ) -> Generator[tuple[int, Iterable[str]], None, None]:
        """Abstract generator method that initializes the process, runs a number of steps and yields a tuple consisting
        of the current time and the set of nodes whose state has changed in each step."""
        if seed is None:
            self.init(self.random_seed())
        else:
            self.init(seed)
        for _ in range(steps):
            ret = self.step()
            if ret is not None:
                yield self.time, ret
            else:
                return None

    def run_experiment(self, steps: int, runs: Optional[Union[int, Iterable[Any]]] = 1) -> DataFrame:
        """Perform one or more simulation runs of the process with a given number of steps."""

        # Generate initializations for different runs
        seeds = []
        if isinstance(runs, int):
            for _ in range(runs):
                seeds.append(self.random_seed())
        elif isinstance(runs, Iterable):
            for s in runs:
                seeds.append(s)
        else:
            raise ValueError("Parameter runs must be an integer or an iterable of seeds.")

        results = []
        run_id: int = 0
        for seed in tqdm(seeds):

            # initialize seed state and record initial state
            self.init(seed)
            for v in self.graph.nodes:
                results.append(
                    {"run_id": run_id, "seed": seed, "time": self.time, "node": v, "state": self.node_state(v)}
                )

            # simulate the given number of steps
            for time, updated_nodes in self.simulation_run(steps, seed):
                # record the new state of each changed node
                for v in updated_nodes:
                    results.append(
                        {"run_id": run_id, "seed": seed, "time": time, "node": v, "state": self.node_state(v)}
                    )
            run_id += 1

        return DataFrame.from_records(results)
