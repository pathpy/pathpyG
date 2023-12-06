"""Initialize pathpyG plotting functions."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : __init__.py -- plotting functions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Wed 2023-12-06 17:06 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
# flake8: noqa
# pylint: disable=unused-import
from typing import Optional, Any

from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph

from pathpyG.visualisations.network_plots import NetworkPlot
from pathpyG.visualisations.network_plots import StaticNetworkPlot
from pathpyG.visualisations.network_plots import TemporalNetworkPlot

PLOT_CLASSES: dict = {
    "network": NetworkPlot,
    "static": StaticNetworkPlot,
    "temporal": TemporalNetworkPlot,
}


def plot(data: dict, kind: Optional[str] = None, **kwargs: Any) -> Any:
    """Plot function."""
    if kind is None:
        if isinstance(data, TemporalGraph):
            kind = "temporal"
        elif isinstance(data, Graph):
            kind = "static"
        else:
            raise NotImplementedError

    plt = PLOT_CLASSES[kind](data, **kwargs)
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
