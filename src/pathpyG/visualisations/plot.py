"""Class to plot pathpy networks."""
# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plot.py -- Module to plot pathpyG networks
# Author    : Jürgen Hackl <hackl@princeton.edu>
#
# Copyright (c) 2016-2019 Pathpy Developers
# =============================================================================
import os
import logging
import importlib

from typing import Optional, Any

# from copy import deepcopy

# create logger
logger = logging.getLogger("root")

# supported backends
_backends: set = {"d3js", "tikz", "matplotlib"}

# supported file formats
_formats: dict = {
    ".html": "d3js",
    ".tex": "tikz",
    ".pdf": "tikz",
    ".png": "matplotlib",
}


def _get_plot_backend(
    backend: Optional[str] = None,
    filename: Optional[str] = None,
    default: str = "d3js",
) -> Any:
    """Return the plotting backend to use."""
    # use default backend per default
    _backend: str = default

    if isinstance(filename, str):
        _backend = _formats.get(os.path.splitext(filename)[1], default)

    # if no backend was found use the backend suggested for the file format
    if backend is not None and backend not in _backends and filename is not None:
        logger.error(f"The backend <{backend}> was not found.")
        raise KeyError

    # if no backend was given use the backend suggested for the file format
    elif isinstance(backend, str) and backend in _backends:
        _backend = backend

    try:
        module = importlib.import_module(f"pathpyG.visualisations._{_backend}")
    except ImportError:
        logger.error(f"The <{_backend}> backend could not be imported.")
        raise ImportError from None

    return module


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
