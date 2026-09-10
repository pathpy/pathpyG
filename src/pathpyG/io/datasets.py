"""Functions to locate the example datasets used by the pathpyG tutorials."""

import logging
from pathlib import Path

logger = logging.getLogger("root")

# Base URL of the `docs/data` folder in the pathpyG GitHub repository.
EXAMPLE_DATA_URL = "https://raw.githubusercontent.com/pathpy/pathpyG/main/docs/data/"

# The `docs/data` folder is not part of the installed package. This path is resolved
# correctly when pathpyG is used from a source checkout.
_EXAMPLE_DATA_DIR = Path(__file__).resolve().parents[3] / "docs" / "data"


def example_data(name: str) -> str:
    """Return a local path or a URL for one of the example datasets in `docs/data`.

    Args:
        name: File name of the dataset, e.g. `temporal_clusters.tedges`.
    Returns:
        Either the path of a local copy of the dataset or a URL from which it can be read.
    """
    local = _EXAMPLE_DATA_DIR / name
    if local.is_file():
        logger.info(f"Reading example dataset from local path {local}")
        return str(local)

    url = EXAMPLE_DATA_URL + name
    logger.info(f"No local copy of {name} found, reading example dataset from {url}")
    return url
