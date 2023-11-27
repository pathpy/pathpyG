import torch

__version__ = "0.0.1"

from pathpyG.utils.config import config
from pathpyG.utils.progress import tqdm

from pathpyG.visualisations import plot, layout
from pathpyG.core.Graph import Graph
from pathpyG.core.TemporalGraph import TemporalGraph
from pathpyG.core.HigherOrderGraph import HigherOrderGraph
from pathpyG.core.PathData import PathData
from pathpyG import io
from pathpyG import nn
from pathpyG import algorithms

if config['device'] == 'cuda':
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'