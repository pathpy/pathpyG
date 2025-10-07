"""Base class for all plot backends."""

from pathpyG.visualisations.pathpy_plot import PathPyPlot


class PlotBackend:
    """Base class for all plot backends."""
    def __init__(self, plot: PathPyPlot):
        """Initialize the backend with a plot."""
        self.data = plot.data
        self.config = plot.config

    def save(self, filename: str) -> None:
        """Save the plot to the hard drive."""
        raise NotImplementedError("Subclasses should implement this method.")

    def show(self) -> None:
        """Show the plot on the device."""
        raise NotImplementedError("Subclasses should implement this method.")
