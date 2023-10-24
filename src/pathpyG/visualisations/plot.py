"""Class to plot pathpy networks."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plot.py -- Module to plot pathpyG networks
# Author    : Jürgen Hackl <hackl@princeton.edu>
#
# Copyright (c) 2016-2019 Pathpy Developers
# =============================================================================
import logging

# from typing import Any

# create logger
logger = logging.getLogger("root")


# from copy import deepcopy


class PathPyPlot:
    """Abstract class for assemblig plots.

    Attributes
    ----------
    data : dict
        data of the plot object
    config : dict
        configuration for the plot

    """

    def __init__(self) -> None:
        """Initialize plot class."""
        logger.debug("Initalize PathPyPlot class")
        self.data: dict = {}
        self.config: dict = {}

    @property
    def _kind(self) -> str:
        """Specify kind str. Must be overridden in child class."""
        raise NotImplementedError

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError

    # def save(self, filename: str, **kwargs: Any) -> None:
    #     """Save the plot to the hard drive."""
    #     _backend: str = kwargs.pop("backend", self.config.get("backend", None))

    #     # plot_backend = _get_plot_backend(_backend, filename)
    #     # plot_backend.plot(
    #     #     deepcopy(self.data), self._kind, **deepcopy(self.config)
    #     # ).save(filename, **kwargs)

    # def show(self, **kwargs):
    #     """Show the plot on the device."""
    #     _backend: str = kwargs.pop("backend", self.config.get("backend", None))

    # plot_backend = _get_plot_backend(_backend, None)
    # plot_backend.plot(
    #     deepcopy(self.data), self._kind, **deepcopy(self.config)
    # ).show(**kwargs)
