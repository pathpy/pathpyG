import torch

__version__ = "0.0.1"

from pathpyG.utils.config import config
from pathpyG.utils.progress import tqdm

if config['device'] == 'cuda':
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'