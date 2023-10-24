"""Generic matplotlib plot class."""
# =============================================================================
# File      : core.py -- Plots with matplotlib backend
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 18:17 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from __future__ import annotations

import logging

from typing import Any

from pathpyG.visualisations.plot import PathPyPlot

# create logger
logger = logging.getLogger("root")


class MatplotlibPlot(PathPyPlot):
    """Base class for plotting matplotlib objects."""

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    def save(self, filename: str, **kwargs: Any) -> None:  # type: ignore
        """Save the plot to the hard drive."""
        self.to_fig().savefig(filename)

    def show(self, **kwargs: Any) -> None:  # type: ignore
        """Show the plot on the device."""
        self.to_fig().show()

    def to_fig(self) -> Any:  # type: ignore
        """Convert to matplotlib figure."""
        raise NotImplementedError


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
