"""Abstract base class for visualization backends.

Defines the common interface that all visualization backends (matplotlib, TikZ, 
d3.js, manim) must implement. Handles plot data extraction and provides 
standardized save/show methods.

Example:
    ```python
    class CustomBackend(PlotBackend):
        def save(self, filename: str) -> None:
            # Implementation for saving
            pass
            
        def show(self) -> None:
            # Implementation for display
            pass
    ```
"""

from pathpyG.visualisations.pathpy_plot import PathPyPlot


class PlotBackend:
    """Abstract base class for all visualization backends.
    
    Provides common interface for matplotlib, TikZ, d3.js, and manim backends.
    Extracts plot data and configuration for backend-specific rendering.
    """
    def __init__(self, plot: PathPyPlot, show_labels: bool) -> None:
        """Initialize backend with plot data and configuration.
        
        Args:
            plot: PathPyPlot instance containing network data
            show_labels: Whether to display node labels
        """
        self.data = plot.data
        self.config = plot.config
        self.show_labels = show_labels

    def save(self, filename: str) -> None:
        """Save plot to file.
        
        Args:
            filename: Output file path
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def show(self) -> None:
        """Display plot on screen.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses should implement this method.")
