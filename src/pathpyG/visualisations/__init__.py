"""PathpyG visualizations."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : __init__.py -- plotting functions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Thu 2023-12-07 08:58 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
# flake8: noqa
# pylint: disable=unused-import
from typing import Optional, Any

from pathpyG.core.Graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph

from pathpyG.visualisations.plot import PathPyPlot
from pathpyG.visualisations.network_plots import NetworkPlot
from pathpyG.visualisations.network_plots import StaticNetworkPlot
from pathpyG.visualisations.network_plots import TemporalNetworkPlot

from pathpyG.visualisations.layout import layout

PLOT_CLASSES: dict = {
    "network": NetworkPlot,
    "static": StaticNetworkPlot,
    "temporal": TemporalNetworkPlot,
}


def plot(data: Graph, kind: Optional[str] = None, **kwargs: Any) -> PathPyPlot:
    """Make plot of pathpyG objects.

    Creates and displays a plot for a given `pathpyG` object. This function can
    generate different types of network plots based on the nature of the input
    data and specified plot kind.

    The function dynamically determines the plot type if not explicitly
    provided, based on the input data type. It supports static network plots
    for `Graph` objects, temporal network plots for `TemporalGraph` objects,
    and potentially other types if specified in `kind`.

    Args:

        data (Graph): A `pathpyG` object representing the network data. This can
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

            The default behavior (when `kind` is None) is to infer the plot type from the data type.

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
        cannot infer the plot type from the `data` type.

    Example Usage:

        >>> import pathpyG as pp
        >>> graph = Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
        >>> plot(graph, kind='static', filename='graph.png')

        This will create a static network plot of the `graph` and save it to 'graph.png'.

    Note:

        - If a 'filename' is provided in `kwargs`, the plot will be saved to
          that file. Otherwise, it will be displayed using `plt.show()`.

        - The function's behavior and the available options in `kwargs` might
          change based on the type of plot being generated.

    Todo:

        - Cleanup the file and use `plt.show()` only in an interactive environment.
    """
    if kind is None:
        if isinstance(data, TemporalGraph):
            kind = "temporal"
        elif isinstance(data, Graph):
            kind = "static"
        else:
            raise NotImplementedError

    filename = kwargs.pop("filename", None)

    plt = PLOT_CLASSES[kind](data, **kwargs)
    if filename:
        plt.save(filename, **kwargs)
    else:
        plt.show(**kwargs)
    return plt


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
