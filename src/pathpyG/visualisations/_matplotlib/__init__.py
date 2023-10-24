"""Initialize matplotlib plotting functions"""
#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : __init__.py -- matplotlib plotting cunctions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 11:55 juergen>
#
# Copyright (c) 2016-2021 Pathpy Developers
# =============================================================================
# flake8: noqa
# pylint: disable=unused-import
from typing import Any
from pathpyG.visualisations._matplotlib.network_plots import NetworkPlot

PLOT_CLASSES: dict = {
    "network": NetworkPlot,
}


def plot(data: dict, kind: str, **kwargs: Any) -> Any:
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
