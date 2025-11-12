"""PathPyG: A Python library for the analysis of paths and temporal networks."""

# flake8: noqa: I001

from importlib.metadata import version as get_version

# We set the version in pyproject.toml because dynamic versioning is currently not supported by uv_build
__version__ = get_version("pathpyG")

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.core.temporal_graph import TemporalGraph
from pathpyG import algorithms, io, nn, statistics
from pathpyG.utils.config import config
from pathpyG.utils.logger import logger
from pathpyG.utils.progress import tqdm
from pathpyG.visualisations import layout, plot

__all__ = [
    "Graph",
    "TemporalGraph",
    "PathData",
    "MultiOrderModel",
    "IndexMap",
    "io",
    "nn",
    "algorithms",
    "statistics",
    "config",
    "logger",
    "tqdm",
    "layout",
    "plot",
]
