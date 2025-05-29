"""
This is Manim Base Plot Class
"""

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
    """
    Function to create and return a Manim-based network plot.

    This function selects a plotting classs based on `kind` argument
    and initializes  it with the given data and optional keyword arguments.

    Args:
        data (dict): The network data to be visualized.
        kind (str, optional): The type of plot to create
        **kwargs (Any): Additional keywork arguments passed to the onstructor.
            These include options for styling and customizing the animation.

    Returns:
        Any: An instance of selected plot class.
    """
    return PLOT_CLASSES[kind](data, **kwargs)
