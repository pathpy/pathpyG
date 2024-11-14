"""Network plots with d3js."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : network_plots.py -- Network plots with d3js
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Thu 2024-11-14 14:21 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import json
import numpy as np

# import logging

from typing import Any

from pathpyG.visualisations._d3js.core import D3jsPlot

# create logger
# logger = logging.getLogger("root")

class NpEncoder(json.JSONEncoder):
    """Encode np values to python for json export."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
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
        for node in self.data["nodes"]:
            node.pop("x", None)
            node.pop("y", None)

    def to_json(self) -> Any:
        """Convert data to json."""
        print(self.data)
        return json.dumps(self.data, cls=NpEncoder)


class StaticNetworkPlot(NetworkPlot):
    """Network plot class for a temporal network."""

    _kind = "static"


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
