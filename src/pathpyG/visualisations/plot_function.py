"""Network visualization orchestration module.

Provides the main plotting interface for pathpyG networks with automatic backend
selection and plot type detection. Serves as the unified entry point for all 
visualization functionality across different backends and graph types.

Key Features:
    - Multi-backend support (matplotlib, TikZ, d3.js, manim)
    - Automatic plot type detection (static vs temporal)
    - File format-based backend inference
    - Unified plotting interface for all graph types

Supported Backends:
    - **matplotlib**: PNG, JPG plots for static visualization
    - **TikZ**: PDF, SVG, TEX for publication-quality vector graphics
    - **d3.js**: HTML for interactive web visualization
    - **manim**: MP4, GIF for animated temporal networks

Examples:
    Plot a static network with the matplotlib backend and save it as `network.png`:

    >>> import pathpyG as pp
    >>> g = pp.Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    >>> pp.plot(g, filename='network.png')

    <img src="../plot/network.png" alt="Example static network plot" width="550"/>
    
    Plot a temporal network with the default d3.js backend:
    
    >>> import pathpyG as pp
    >>> tg = pp.TemporalGraph.from_edge_list([('a', 'b', 1), ('b', 'c', 2), ('a', 'c', 3)])
    >>> pp.plot(tg)

    <iframe src="../plot/temporal_network.html" width="650" height="520"></iframe>
    ```

!!! tip "Backend Selection"
    Backends are auto-selected from file extensions or can be explicitly 
    specified via the `backend` parameter.
"""

# !/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : plot.py -- Module to plot pathpyG networks
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Wed 2023-12-06 17:28 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
import importlib
import logging
import os
from enum import Enum
from typing import Any, Optional

from pathpyG import config
from pathpyG.core.graph import Graph
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG.visualisations._d3js.backend import D3jsBackend
from pathpyG.visualisations.network_plot import NetworkPlot
from pathpyG.visualisations.plot_backend import PlotBackend
from pathpyG.visualisations.temporal_network_plot import TemporalNetworkPlot

# create logger
logger = logging.getLogger("root")

# supported backends
class Backends(str, Enum):
    """Enumeration of supported visualization backends.
    
    Defines the available backend engines for network visualization,
    each optimized for different output formats and use cases.
    """
    d3js = "d3js"
    matplotlib = "matplotlib"
    tikz = "tikz"
    manim = "manim"

    @staticmethod
    def is_backend(backend: str) -> bool:
        """Check if string is a valid backend identifier.
        
        Args:
            backend: Backend name to validate
            
        Returns:
            True if backend is supported, False otherwise
        """
        return backend in Backends.__members__.values()

# supported file formats
FORMATS: dict = {
    ".html": Backends.d3js,
    ".tex": Backends.tikz,
    ".pdf": Backends.tikz,
    ".svg": Backends.tikz,
    ".png": Backends.matplotlib,
    ".mp4": Backends.manim,
    ".gif": Backends.manim,
}

# Supported Plot Classes
PLOT_CLASSES: dict = {
    "static": NetworkPlot,
    "temporal": TemporalNetworkPlot,
}

def _get_plot_backend(backend: Optional[str], filename: Optional[str], default: str) -> type[PlotBackend]:
    """Determine and import the appropriate plotting backend.
    
    Resolves backend selection based on explicit backend parameter,
    file extension inference, or default fallback. Dynamically imports
    the selected backend module.
    
    Args:
        backend: Explicit backend name or None for auto-detection
        filename: Output filename for extension-based inference
        default: Fallback backend when no preference specified
        
    Returns:
        Backend class ready for instantiation
        
    Raises:
        KeyError: If specified backend is not supported
        ImportError: If backend module cannot be imported
    """
    # check if backend is valid backend type based on enum
    if backend is not None and not Backends.is_backend(backend):
        logger.error(f"The backend <{backend}> was not found.")
        raise KeyError
    # use given backend if valid
    elif isinstance(backend, str) and Backends.is_backend(backend):
        logger.debug(f"Using backend <{backend}>.")
        _backend = backend
    # if no backend was given use the backend suggested for the file format
    else:
        # Get file ending and try to infer backend
        if isinstance(filename, str):
            _backend = FORMATS.get(os.path.splitext(filename)[1], default)
            logger.debug(f"Using backend <{_backend}> inferred from file ending.")
        else:
            # use default backend per default
            _backend = default
            logger.debug(f"Using default backend <{_backend}>.")

    # try to load backend class or return error
    try:
        module = importlib.import_module(f"pathpyG.visualisations._{_backend}.backend")
    except ImportError as e:
        logger.error(f"The <{_backend}> backend could not be imported.")
        raise ImportError from e

    return getattr(module, f"{_backend.capitalize()}Backend")


def plot(graph: Graph, kind: Optional[str] = None, show_labels=None, **kwargs: Any) -> PlotBackend:
    """Make plot of pathpyG objects.

    Creates and displays a plot for a given `pathpyG` object. This function can
    generate different types of network plots based on the nature of the input
    data and specified plot kind.

    The function dynamically determines the plot type if not explicitly
    provided, based on the input data type. It supports static network plots
    for `Graph` objects, temporal network plots for `TemporalGraph` objects,
    and potentially other types if specified in `kind`.

    Args:
        graph: A `pathpyG` object representing the network data. This can
            be a `Graph` or `TemporalGraph` object, or other compatible types.
        kind: A string keyword defining the type of plot to generate. Options include:
            **'static'**, and **'temporal'**.
        show_labels: Whether to display node labels (None uses graph.mapping.has_ids)
        **kwargs: Backend-specific plotting parameters including:
            **filename**: Output file path (triggers backend auto-selection);
            **backend**: Explicit backend choice;
            **layout**: Layout algorithm name;
            **style**: Various styling parameters (colors, sizes, etc.)

    Returns:
        Configured backend instance ready for display or saving

    Raises:
        NotImplementedError: If graph type cannot be auto-detected for plotting
        KeyError: If specified backend is not supported
        ImportError: If required backend cannot be loaded

    Examples:
        This will create a static network plot of the `graph` and save it to 'graph.png'.

        >>> import pathpyG as pp
        >>> graph = pp.Graph.from_edge_list([["a", "b"], ["b", "c"], ["a", "c"]])
        >>> pp.plot(graph, kind="static", filename="graph.png")

        <img src="../plot/graph.png" alt="Example static network plot" width="550"/>
        
    Note:
        - If a 'filename' is provided in `kwargs`, the plot will be saved to
          that file. Otherwise, it will be displayed using `plt.show()`.
        - The function's behavior and the available options in `kwargs` might
          change based on the type of plot being generated.

    !!! abstract "Backend Auto-Selection"
        When filename is provided, backend is inferred from extension:
        
        | Extension | Backend | Best For |
        |-----------|---------|----------|
        | .png, .jpg | matplotlib | Quick visualization |
        | .pdf, .svg, .tex | tikz | Publication quality |
        | .html | d3js | Interactive exploration |
        | .mp4, .gif | manim | Animated sequences |
    """
    if kind is None:
        if isinstance(graph, TemporalGraph):
            kind = "temporal"
        elif isinstance(graph, Graph):
            kind = "static"
        else:
            raise NotImplementedError

    if show_labels is None:
        show_labels = graph.mapping.has_ids

    filename = kwargs.pop("filename", None)
    _backend: str = kwargs.pop("backend", None)

    plot_backend_class = _get_plot_backend(
        backend=_backend, filename=filename, default=config.get("visualisation").get("default_backend")  # type: ignore[union-attr]
    )

    # Check if backend is d3js and set layout to None if not specifically given as argument
    if plot_backend_class == D3jsBackend:
        if "layout" not in kwargs:
            kwargs["layout"] = None

    plt = PLOT_CLASSES[kind](graph, **kwargs)
    plot_backend = plot_backend_class(plt, show_labels=show_labels)
    if filename:
        plot_backend.save(filename)
    else:
        if config["environment"]["interactive"]:
            plot_backend.show()
    return plot_backend
