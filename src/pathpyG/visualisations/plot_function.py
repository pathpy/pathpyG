"""Class to plot pathpy networks."""

# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plot.py -- Module to plot pathpyG networks
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Wed 2023-12-06 17:28 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
import importlib
import logging
import os
from enum import Enum
from typing import Any, Optional

from pathpyG import config
from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.visualisations.plot_backend import PlotBackend
# from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot

# create logger
logger = logging.getLogger("root")

# supported backends
class Backends(str, Enum):
    """Supported backends."""
    d3js = "d3js"
    matplotlib = "matplotlib"
    tikz = "tikz"
    manim = "manim"

    @staticmethod
    def is_backend(backend: str) -> bool:
        """Check if value is a valid backend."""
        return backend in Backends.__members__.values()

# supported file formats
FORMATS: dict = {
    ".html": Backends.d3js,
    ".tex": Backends.tikz,
    ".pdf": Backends.tikz,
    ".svg": Backends.tikz,
    ".png": Backends.matplotlib,
    ".mp4": Backends.manim,
    ".gif": Backends.manim,
}

# Supported Plot Classes
PLOT_CLASSES: dict = {
    "static": NetworkPlot,
    # "temporal": TemporalNetworkPlot,
}

def _get_plot_backend(backend: Optional[str], filename: Optional[str], default: str) -> PlotBackend:
    """Return the plotting backend to use."""
    # check if backend is valid backend type based on enum
    if backend is not None and not Backends.is_backend(backend):
        logger.error(f"The backend <{backend}> was not found.")
        raise KeyError
    # use given backend if valid
    elif isinstance(backend, str) and Backends.is_backend(backend):
        logger.debug(f"Using backend <{backend}>.")
        _backend = backend
    # if no backend was given use the backend suggested for the file format
    else:
        # Get file ending and try to infer backend
        if isinstance(filename, str):
            _backend = FORMATS.get(os.path.splitext(filename)[1], default)
            logger.debug(f"Using backend <{_backend}> inferred from file ending.")
        else:
            # use default backend per default
            _backend = default
            logger.debug(f"Using default backend <{_backend}>.")

    # try to load backend class or return error
    try:
        module = importlib.import_module(f"pathpyG.visualisations._{_backend}.backend")
    except ImportError as e:
        logger.error(f"The <{_backend}> backend could not be imported.")
        raise ImportError from e

    return getattr(module, f"{_backend.capitalize()}Backend")  # type: ignore[return-value]


def plot(graph: Graph, kind: Optional[str] = None, show_labels=None, **kwargs: Any) -> PathPyPlot:
    """Make plot of pathpyG objects.

    Creates and displays a plot for a given `pathpyG` object. This function can
    generate different types of network plots based on the nature of the input
    data and specified plot kind.

    The function dynamically determines the plot type if not explicitly
    provided, based on the input data type. It supports static network plots
    for `Graph` objects, temporal network plots for `TemporalGraph` objects,
    and potentially other types if specified in `kind`.

    Args:
        graph (Graph): A `pathpyG` object representing the network data. This can
            be a `Graph` or `TemporalGraph` object, or other compatible types.
        kind (Optional[str], optional): A string keyword defining the type of
            plot to generate. Options include:
            - 'static' : Generates a static (aggregated) network plot. Ideal
              for `Graph` objects.
            - 'temporal' : Creates a temporal network plot, which includes time
              components. Suitable for `TemporalGraph` objects.
            - 'hist' : Produces a histogram of network properties. (Note:
              Implementation for 'hist' is not present in the given function
              code, it's mentioned for possible extension.)
            The default behavior (when `kind` is None) is to infer the plot type from the graph type.
        show_labels (Optional[bool], optional): Whether to display node labels
            on the plot. If None, the function will decide based on the IndexMap.
        **kwargs (Any): Optional keyword arguments to customize the plot. These
              arguments are passed directly to the plotting class. Common options
              could include layout parameters, color schemes, and plot size.

    Returns:
        PathPyPlot: A `PathPyPlot` object representing the generated plot.
            This could be an instance of a plot class from
            `pathpyG.visualisations.network_plots`, depending on the kind of
             plot generated.

    Raises:
        NotImplementedError: If the `kind` is not recognized or if the function
        cannot infer the plot type from the `graph` type.

    Examples:
        This will create a static network plot of the `graph` and save it to 'graph.png'.

        >>> import pathpyG as pp
        >>> graph = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
        >>> plot(graph, kind="static", filename="graph.png")

    Note:
        - If a 'filename' is provided in `kwargs`, the plot will be saved to
          that file. Otherwise, it will be displayed using `plt.show()`.
        - The function's behavior and the available options in `kwargs` might
          change based on the type of plot being generated.
    """
    if kind is None:
        if isinstance(graph, TemporalGraph):
            kind = "temporal"
        elif isinstance(graph, Graph):
            kind = "static"
        else:
            raise NotImplementedError

    if show_labels is None:
        show_labels = graph.mapping.has_ids

    filename = kwargs.pop("filename", None)
    _backend: str = kwargs.pop("backend", None)

    plt = PLOT_CLASSES[kind](graph, **kwargs)
    plot_backend_class = _get_plot_backend(
        backend=_backend, filename=filename, default=config.get("visualisation").get("default_backend")  # type: ignore[union-attr]
    )
    plot_backend = plot_backend_class(plt, show_labels=show_labels)
    if filename:
        plot_backend.save(filename)
    else:
        if config["environment"]["interactive"]:
            plot_backend.show()
    return plot_backend
