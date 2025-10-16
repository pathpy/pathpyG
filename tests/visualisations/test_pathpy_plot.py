"""Unit tests for PathPyPlot base class."""

import logging

import pytest

from pathpyG import config
from pathpyG.visualisations.pathpy_plot import PathPyPlot


class TestPathPyPlotInitialization:
    """Test PathPyPlot initialization and configuration."""

    def test_pathpy_plot_creates_empty_data(self) -> None:
        """Test that PathPyPlot initializes with empty data dict."""
        plot = PathPyPlot()
        assert isinstance(plot.data, dict)
        assert len(plot.data) == 0

    def test_pathpy_plot_loads_config(self) -> None:
        """Test that PathPyPlot loads visualization config."""
        plot = PathPyPlot()
        assert isinstance(plot.config, dict)
        assert "node" in plot.config
        assert "edge" in plot.config

    def test_pathpy_plot_config_is_copy(self) -> None:
        """Test that config is a copy and modifications don't affect global config."""
        plot1 = PathPyPlot()
        plot2 = PathPyPlot()
        
        # Modify plot1 config
        plot1.config["custom_key"] = "custom_value"
        
        # plot2 should not have the custom key
        assert "custom_key" not in plot2.config
        
        # Global config should not be affected
        vis_config = config.get("visualisation", {})
        assert "custom_key" not in vis_config

    def test_pathpy_plot_normalizes_node_color_list_to_tuple(self) -> None:
        """Test that list colors are converted to tuples."""
        # Get original config
        original_config = config.get("visualisation", {}).copy()
        
        # Create plot
        plot = PathPyPlot()
        
        # If node color is a tuple or was converted from list
        if "node" in plot.config and "color" in plot.config["node"]:
            node_color = plot.config["node"]["color"]
            # Should be tuple if it was originally a list or tuple
            if isinstance(original_config.get("node", {}).get("color"), (list, tuple)):
                assert isinstance(node_color, tuple)

    def test_pathpy_plot_normalizes_edge_color_list_to_tuple(self) -> None:
        """Test that list edge colors are converted to tuples."""
        plot = PathPyPlot()
        
        if "edge" in plot.config and "color" in plot.config["edge"]:
            edge_color = plot.config["edge"]["color"]
            # Should be tuple if config specified list or tuple
            original_config = config.get("visualisation", {})
            if isinstance(original_config.get("edge", {}).get("color"), (list, tuple)):
                assert isinstance(edge_color, tuple)

    def test_pathpy_plot_logs_initialization(self, caplog) -> None:
        """Test that initialization logs debug message."""
        with caplog.at_level(logging.DEBUG, logger="root"):
            _ = PathPyPlot()
        
        # Check that initialization was logged
        assert any(
            "Intialising PathpyPlot with config:" in record.message
            for record in caplog.records
        )


class TestPathPyPlotGenerate:
    """Test PathPyPlot generate method."""

    def test_generate_not_implemented(self) -> None:
        """Test that generate() raises NotImplementedError."""
        plot = PathPyPlot()
        
        with pytest.raises(NotImplementedError):
            plot.generate()


class TestPathPyPlotSubclassing:
    """Test PathPyPlot subclassing behavior."""

    def test_subclass_can_override_generate(self) -> None:
        """Test that subclasses can implement generate()."""
        class CustomPlot(PathPyPlot):
            def generate(self) -> None:
                self.data["test_key"] = "test_value"
        
        plot = CustomPlot()
        plot.generate()
        
        assert plot.data["test_key"] == "test_value"

    def test_subclass_inherits_data_and_config(self) -> None:
        """Test that subclasses inherit data and config."""
        class CustomPlot(PathPyPlot):
            def generate(self) -> None:
                pass
        
        plot = CustomPlot()
        
        assert hasattr(plot, "data")
        assert hasattr(plot, "config")
        assert isinstance(plot.data, dict)
        assert isinstance(plot.config, dict)

    def test_subclass_can_modify_config_in_init(self) -> None:
        """Test that subclasses can modify config during initialization."""
        class CustomPlot(PathPyPlot):
            def __init__(self, custom_option: str):
                super().__init__()
                self.config["custom_option"] = custom_option
            
            def generate(self) -> None:
                pass
        
        plot = CustomPlot("my_value")
        
        assert plot.config["custom_option"] == "my_value"

    def test_subclass_can_populate_data_in_generate(self) -> None:
        """Test that subclasses can populate data in generate()."""
        class DataPlot(PathPyPlot):
            def generate(self) -> None:
                self.data["values"] = [1, 2, 3, 4, 5]
                self.data["labels"] = ["a", "b", "c", "d", "e"]
        
        plot = DataPlot()
        assert len(plot.data) == 0  # Before generate
        
        plot.generate()
        
        assert len(plot.data) == 2  # After generate
        assert plot.data["values"] == [1, 2, 3, 4, 5]
        assert plot.data["labels"] == ["a", "b", "c", "d", "e"]
