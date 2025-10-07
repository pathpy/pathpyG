import logging

from pathpyG import config

logger = logging.getLogger("root")


class PathPyPlot:
    """Abstract class for assembling plots.

    Attributes:
        data (pd.DataFrame): Data of the plot object.
        config (dict): Configuration for the plot.
    """

    def __init__(self) -> None:
        """Initialize plot class."""
        self.data: dict = {}
        self.config: dict = config.get("visualisation", {}).copy()
        logger.debug(f"Intialising PathpyPlot with config: {self.config}")

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError
