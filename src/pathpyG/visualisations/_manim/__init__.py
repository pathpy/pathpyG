from typing import Any
from pathpyG.visualisations._manim.network_plots import NetworkPlot
from pathpyG.visualisations._manim.network_plots import StaticNetworkPlot
from pathpyG.visualisations._manim.network_plots import TemporalNetworkPlot

PLOT_CLASSES: dict = {
    "network": NetworkPlot,
    "static": StaticNetworkPlot,
    "temporal": TemporalNetworkPlot,
}


def plot(data: dict, kind: str = "network", **kwargs: Any) -> Any:
    """Plot function."""
    return PLOT_CLASSES[kind](data, **kwargs)
