"""Histogram plot classes."""
from __future__ import annotations
import logging

from typing import TYPE_CHECKING, Any
from pathpyG.visualisations.plot import PathPyPlot

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.graph import Graph

# create logger
logger = logging.getLogger("root")


def hist(
    network: Graph, key: str = "indegrees", bins: int = 10, **kwargs: Any
) -> HistogramPlot:
    """Plot a histogram."""
    return HistogramPlot(network, key, bins, **kwargs)


class HistogramPlot(PathPyPlot):
    """Histogram plot class for a network property."""

    _kind = "hist"

    def __init__(
        self, network: Graph, key: str = "indegrees", bins: int = 10, **kwargs: Any
    ) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.network = network
        self.config = kwargs
        self.config["bins"] = bins
        self.config["key"] = key
        self.generate()

    def generate(self) -> None:
        """Generate the plot."""
        logger.debug("Generate histogram.")

        data: dict = {}

        match self.config["key"]:
            case "indegrees":
                logger.debug("Generate data for in-degrees")
                data["values"] = list(self.network.degrees(mode="in").values())
            case "outdegrees":
                logger.debug("Generate data for out-degrees")
                data["values"] = list(self.network.degrees(mode="out").values())
            case _:
                logger.error(
                    f"The <{self.config['key']}> property",
                    "is currently not supported for hist plots.",
                )
                raise KeyError

        data["title"] = self.config["key"]
        self.data["data"] = data
