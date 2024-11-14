"""Initialize tikz plotting functions."""

# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : __init__.py -- tikz plotting cunctions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Sun 2023-11-19 15:29 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
# flake8: noqa
# pylint: disable=unused-import
from typing import Any
from pathpyG.visualisations._tikz.network_plots import NetworkPlot
from pathpyG.visualisations._tikz.network_plots import StaticNetworkPlot
from pathpyG.visualisations._tikz.network_plots import TemporalNetworkPlot

PLOT_CLASSES: dict = {
    "network": NetworkPlot,
    "static": StaticNetworkPlot,
    "temporal": TemporalNetworkPlot,
}


def plot(data: dict, kind: str = "network", **kwargs: Any) -> Any:
    """Plot function."""
    return PLOT_CLASSES[kind](data, **kwargs)


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
