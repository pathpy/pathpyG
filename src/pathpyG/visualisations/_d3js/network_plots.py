"""Network plots with d3js."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_plots.py -- Network plots with d3js
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Sun 2023-11-19 10:41 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import json

# import logging

from typing import Any

from pathpyG.visualisations._d3js.core import D3jsPlot

# create logger
# logger = logging.getLogger("root")


class NetworkPlot(D3jsPlot):
    """Network plot class for a static network."""

    _kind = "network"

    def __init__(self, data: dict, **kwargs: Any) -> None:
        """Initialize network plot class."""
        super().__init__()
        self.data = data
        self.config = kwargs
        self.generate()

    def generate(self) -> None:
        """Clen up data."""
        self.config.pop("node_cmap", None)
        self.config.pop("edge_cmap", None)

    def to_json(self) -> Any:
        """Convert data to json."""
        return json.dumps(self.data)


class TemporalNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "temporal"


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
