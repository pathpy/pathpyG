"""Abstract base class for plot data preparation.

Provides common foundation for assembling plot data and configuration
before backend-specific rendering. Handles configuration loading and
data structure initialization.
"""
import logging

from pathpyG import config

logger = logging.getLogger("root")


class PathPyPlot:
    """Abstract base class for plot data assembly.

    Prepares network data and configuration for backend rendering.
    Subclasses implement specific plot types (static, temporal, histogram, etc.).
    
    Attributes:
        data: Dictionary containing processed plot data
        config: Visualization configuration from pathpyG settings
    """

    def __init__(self) -> None:
        """Initialize plot with empty data and default configuration.
        
        Loads visualization config and normalizes color settings from
        lists to tuples for consistency across backends.
        """
        self.data: dict = {}
        self.config: dict = config.get("visualisation", {}).copy()
        if isinstance(self.config["node"]["color"], list):
            self.config["node"]["color"] = tuple(self.config["node"]["color"])
        if isinstance(self.config["edge"]["color"], list):
            self.config["edge"]["color"] = tuple(self.config["edge"]["color"])
        logger.debug(f"Intialising PathpyPlot with config: {self.config}")

    def generate(self) -> None:
        """Generate plot data structures.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
