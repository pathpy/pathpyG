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
        if isinstance(self.config["node"]["color"], list):
            self.config["node"]["color"] = tuple(self.config["node"]["color"])
        if isinstance(self.config["edge"]["color"], list):
            self.config["edge"]["color"] = tuple(self.config["edge"]["color"])
        logger.debug(f"Intialising PathpyPlot with config: {self.config}")

    def generate(self) -> None:
        """Generate the plot."""
        raise NotImplementedError
