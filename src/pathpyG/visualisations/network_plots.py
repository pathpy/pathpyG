"""Network plots with d3js."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_plots.py -- Network plots
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 15:43 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import logging

from typing import Any
from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


def network_plot(**kwargs: Any) -> NetworkPlot:
    """Plot a static network.

    This function generates a static plot of the network, thereby different
    output can be chosen, including

    - interactive html with d3js
    - tex file with tikz code
    - pdf from the tex source
    - png based on matplotlib

    The appearance of the plot can be modified by keyword arguments which will
    be explained in more detail below.

    Parameters
    ----------
    network : Network

        A :py:class`Network` object

    kwargs : keyword arguments, optional (default = no attributes)

        Attributes used to modify the appearance of the plot.
        For details see below.

    Keyword arguments used for the plotting:

    filename : str optional (default = None)

        Filename to save. The file ending specifies the output. i.e. is the
        file ending with '.tex' a tex file will be created; if the file ends
        with '.pdf' a pdf is created; if the file ends with '.html', a html
        file is generated generated. If no ending is defined a temporary html
        file is compiled and shown.


    **Nodes:**

    - ``node_size`` : diameter of the node

    - ``node_color`` : The fill color of the node. Possible values are:

            - A single color string referred to by name, RGB or RGBA code, for
              instance 'red' or '#a98d19' or (12,34,102).

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.


    - ``node_cmap`` : Colormap for node colors. If node colors are given as int
      or float values the color will be assigned based on a colormap. Per
      default the color map goes from red to green. Matplotlib colormaps or
      seaborn color palettes can be used to style the node colors.

    - ``node_opacity`` : fill opacity of the node. The default is 1. The range
      of the number lies between 0 and 1. Where 0 represents a fully
      transparent fill and 1 a solid fill.


    **Edges**

    - ``edge_size`` : width of the edge

    - ``edge_color`` : The line color of the edge. Possible values are:

            - A single color string referred to by name, RGB or RGBA code, for
              instance 'red' or '#a98d19' or (12,34,102).

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.


    - ``edge_cmap`` : Colormap for edge colors. If node colors are given as int
      or float values the color will be assigned based on a colormap. Per
      default the color map goes from red to green. Matplotlib colormaps or
      seaborn color palettes can be used to style the edge colors.

    - ``edge_opacity`` : line opacity of the edge. The default is 1. The range
      of the number lies between 0 and 1. Where 0 represents a fully
      transparent fill and 1 a solid fill.

    **General**

    - ``keep_aspect_ratio``

    - ``margin``

    - ``layout``

    References
    ----------
    .. [tn] https://github.com/hackl/tikz-network

    """
    return NetworkPlot(**kwargs)


class NetworkPlot(PathPyPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        # self.network = network
        # self.config = kwargs
        # self.generate()

        pass


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
