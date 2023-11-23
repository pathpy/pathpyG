"""Histogram plot classes."""
from __future__ import annotations
import logging

from typing import TYPE_CHECKING, Any
from pathpyG.visualisations.plot import PathPyPlot

# pseudo load class for type checking
if TYPE_CHECKING:
    from pathpyG.core.Graph import Graph

# create logger
logger = logging.getLogger("root")


def hist(
    network: Graph, key: str = "degree", bins: int = 10, **kwargs: Any
) -> HistogramPlot:
    """Plot a histogram."""
    return HistogramPlot(network, key, bins, **kwargs)


class HistogramPlot(PathPyPlot):
    """Histogram plot class for a network properties."""

    _kind = "hist"

    def __init__(
        self, network: Graph, key: str = "degree", bins: int = 10, **kwargs: Any
    ) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.key = "degree"
        self.network = network
        self.config = kwargs
        self.generate()

    def generate(self) -> None:
        """Generate the plot."""
        logger.debug("Generate histogram.")
