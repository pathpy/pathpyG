"""pathpyG is an Open Source package facilitating next-generation network analytics and
graph learning for time series data on graphs.

Building on the industry-proven data structures and concepts of `pytorch`
and `torch_geometric`, pathpyG makes it easier than ever to apply machine learning
to temporal graph data.

pathpyG is jointly developed at University of Wuerzburg, Princeton University,
and University of Zurich. The research behind pathpyG has been funded by the
Swiss National Science Foundation via 
[grant 176938](https://data.snf.ch/grants/grant/176938).
"""
import torch

__version__ = "0.1.0"

from pathpyG.utils.config import config
from pathpyG.utils.progress import tqdm

from pathpyG.core.Graph import Graph
from pathpyG.core.IndexMap import IndexMap
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.path_data import PathData
from pathpyG.core.MultiOrderModel import MultiOrderModel
from pathpyG import io
from pathpyG import nn
from pathpyG import algorithms
from pathpyG import processes
from pathpyG.visualisations import plot, layout


if config["device"] == "cuda":
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
