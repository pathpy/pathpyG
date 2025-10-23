"""Unit tests for NetworkPlot class in pathpyG.visualisations."""

import numpy as np
import pandas as pd
import pytest

from pathpyG.core.graph import Graph
from pathpyG.core.index_map import IndexMap
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.core.path_data import PathData
from pathpyG.visualisations.network_plot import NetworkPlot


class TestNetworkPlot:
    def setup_method(self):
        # Simple triangle graph
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])

    def test_initialization_and_config(self):
        plot = NetworkPlot(self.g, node_color="#ff0000", edge_size=2)
        assert plot.network is self.g
        assert plot.node_args["color"] == "#ff0000"
        assert plot.edge_args["size"] == 2

    def test_node_and_edge_data_structure(self):
        plot = NetworkPlot(self.g)
        nodes = plot.data["nodes"]
        edges = plot.data["edges"]
        assert isinstance(nodes, pd.DataFrame)
        assert isinstance(edges, pd.DataFrame)
        assert set(nodes.index) == set(self.g.nodes)
        assert set(edges.index.names) == {"source", "target"}

    def test_node_default_attributes(self):
        plot = NetworkPlot(self.g)
        nodes = plot.data["nodes"]
        # Default attributes should be present
        for attr in ["color", "size", "opacity", "image"]:
            assert attr in nodes.columns

    def test_edge_default_attributes(self):
        plot = NetworkPlot(self.g)
        edges = plot.data["edges"]
        # Default attributes should be present
        for attr in ["color", "size", "opacity"]:
            assert attr in edges.columns

    def test_node_attribute_constant_assignment(self):
        """Test assigning constant values to node attributes."""
        plot = NetworkPlot(self.g, node_color="#00ff00", node_size=10, node_opacity=0.8)
        nodes = plot.data["nodes"]
        assert (nodes["color"] == "#00ff00").all()
        assert (nodes["size"] == 10).all()
        assert (nodes["opacity"] == 0.8).all()

    def test_node_attribute_list_assignment(self):
        """Test assigning lists to node attributes."""
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        sizes = [5, 10, 15]
        plot = NetworkPlot(self.g, node_color=colors, node_size=sizes)
        nodes = plot.data["nodes"]
        assert len(nodes) == 3
        assert set(nodes["color"]) == set(colors)
        assert set(nodes["size"]) == set(sizes)

    def test_node_attribute_dict_assignment(self):
        """Test assigning dictionaries mapping node IDs to attributes."""
        color_map = {"a": "#ff0000", "b": "#00ff00"}
        size_map = {"a": 10, "c": 20}
        plot = NetworkPlot(self.g, node_color=color_map, node_size=size_map)
        nodes = plot.data["nodes"]
        assert nodes.loc["a", "color"] == "#ff0000"
        assert nodes.loc["b", "color"] == "#00ff00"
        assert nodes.loc["a", "size"] == 10
        assert nodes.loc["c", "size"] == 20

    def test_node_attribute_rgb_tuple_assignment(self):
        """Test assigning RGB tuple as constant color."""
        rgb_color = (255, 128, 0)
        plot = NetworkPlot(self.g, node_color=rgb_color)
        nodes = plot.data["nodes"]
        # Should be converted to hex
        assert (nodes["color"] == "#ff8000").all()

    def test_node_attribute_numeric_color_mapping(self):
        """Test numeric values mapped to colors via colormap."""
        numeric_values = [0.0, 0.5, 1.0]
        plot = NetworkPlot(self.g, node_color=numeric_values)
        nodes = plot.data["nodes"]
        # Should have hex colors from colormap
        assert all(nodes["color"].str.startswith("#"))
        # All should be different colors
        assert len(nodes["color"].unique()) == 3

    def test_node_attribute_list_wrong_length_raises(self):
        """Test that wrong-length lists raise an error."""
        with pytest.raises(AttributeError):
            NetworkPlot(self.g, node_size=[10, 20])  # Only 2 values for 3 nodes

    def test_edge_attribute_constant_assignment(self):
        """Test assigning constant values to edge attributes."""
        plot = NetworkPlot(self.g, edge_color="#0000ff", edge_size=5, edge_opacity=0.6)
        edges = plot.data["edges"]
        assert (edges["color"] == "#0000ff").all()
        assert (edges["size"] == 5).all()
        assert (edges["opacity"] == 0.6).all()

    def test_edge_attribute_list_assignment(self):
        """Test assigning lists to edge attributes."""
        # Triangle has 3 edges (undirected will deduplicate to 3)
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        plot = NetworkPlot(self.g, edge_color=colors)
        edges = plot.data["edges"]
        assert len(edges) == 3
        assert set(edges["color"]) == set(colors)

    def test_edge_attribute_dict_assignment(self):
        """Test assigning dictionaries mapping edge tuples to attributes."""
        color_map = {("a", "b"): "#ff0000", ("b", "c"): "#00ff00"}
        size_map = {("a", "b"): 10, ("c", "a"): 20}
        plot = NetworkPlot(self.g, edge_color=color_map, edge_size=size_map)
        edges = plot.data["edges"]
        # Check at least one edge got the color
        assert "#ff0000" in edges["color"].values or "#00ff00" in edges["color"].values
        assert 10 in edges["size"].values or 20 in edges["size"].values

    def test_edge_weight_as_size(self):
        """Test that edge_weight attribute is used as size when present."""
        # Create a weighted graph
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        import torch
        g.data.edge_weight = torch.tensor([1.0, 2.0, 3.0])
        
        plot = NetworkPlot(g)
        edges = plot.data["edges"]
        # Edge sizes should come from weights
        assert set(edges["size"].unique()).issubset({1.0, 2.0, 3.0})

    def test_edge_attribute_from_network_data(self):
        """Test that edge attributes from network data are used."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        import torch
        g.data.edge_color = ["#ff0000", "#00ff00", "#0000ff"]
        g.data.edge_size = torch.tensor([5, 10, 15])
        
        plot = NetworkPlot(g)
        edges = plot.data["edges"]
        # Should use network attributes
        assert set(edges["color"]).issubset({"#ff0000", "#00ff00", "#0000ff"})
        assert set(edges["size"]).issubset({5, 10, 15})

    def test_node_attribute_from_network_data(self):
        """Test that node attributes from network data are used."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        import torch
        g.data.node_color = ["#ff0000", "#00ff00", "#0000ff"]
        g.data.node_size = torch.tensor([5, 10, 15])
        
        plot = NetworkPlot(g)
        nodes = plot.data["nodes"]
        # Should use network attributes
        assert set(nodes["color"]) == {"#ff0000", "#00ff00", "#0000ff"}
        assert set(nodes["size"]) == {5, 10, 15}

    def test_node_kwargs_override_network_data(self):
        """Test that kwargs override network data attributes."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        import torch
        g.data.node_size = torch.tensor([5, 10, 15])
        
        # Override with kwargs
        plot = NetworkPlot(g, node_size=20)
        nodes = plot.data["nodes"]
        assert (nodes["size"] == 20).all()

    def test_edge_kwargs_override_network_data(self):
        """Test that kwargs override network data attributes."""
        g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        import torch
        g.data.edge_size = torch.tensor([5, 10, 15])
        
        # Override with kwargs
        plot = NetworkPlot(g, edge_size=20)
        edges = plot.data["edges"]
        assert (edges["size"] == 20).all()

    def test_layout_integration(self):
        plot = NetworkPlot(self.g, layout="spring")
        nodes = plot.data["nodes"]
        assert "x" in nodes.columns and "y" in nodes.columns
        assert np.all((nodes["x"] >= 0) & (nodes["x"] <= 1))
        assert np.all((nodes["y"] >= 0) & (nodes["y"] <= 1))

    def test_higher_order_network(self):
        # Create a higher-order network from path data
        paths = PathData(IndexMap(["a", "b", "c", "d"]))
        paths.append_walks([["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"], ["a", "d"]], weights=[1, 1, 1, 1])
        ho_g = MultiOrderModel.from_path_data(paths, max_order=2).layers[2]
        # Create a plot for the higher-order graph
        plot = NetworkPlot(ho_g, node_color="#123456")
        nodes = plot.data["nodes"]
        # Index should be stringified tuples
        assert all(isinstance(idx, str) for idx in nodes.index)

    def test_invalid_image_path_raises(self):
        with pytest.raises(AttributeError):
            NetworkPlot(self.g, node_image="/nonexistent/path/to/image.png")


class TestNetworkPlotColorConversion:
    """Test color conversion methods in NetworkPlot."""
    
    def setup_method(self):
        self.g = Graph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
        self.plot = NetworkPlot(self.g)
    
    def test_convert_color_hex_string(self):
        """Test that hex strings are preserved."""
        assert self.plot._convert_color("#ff0000") == "#ff0000"
        assert self.plot._convert_color("#00FF00") == "#00FF00"
        assert self.plot._convert_color("#0000ff") == "#0000ff"
    
    def test_convert_color_rgb_tuple_float(self):
        """Test conversion of RGB tuples with float values (0-1)."""
        result = self.plot._convert_color((1.0, 0.0, 0.0))
        assert result == "#ff0000"
        
        result = self.plot._convert_color((0.0, 1.0, 0.0))
        assert result == "#00ff00"
        
        result = self.plot._convert_color((0.0, 0.0, 1.0))
        assert result == "#0000ff"
    
    def test_convert_color_rgb_tuple_int(self):
        """Test conversion of RGB tuples with integer values (0-255)."""
        result = self.plot._convert_color((255, 0, 0))
        assert result == "#ff0000"
        
        result = self.plot._convert_color((0, 255, 0))
        assert result == "#00ff00"
        
        result = self.plot._convert_color((0, 0, 255))
        assert result == "#0000ff"
    
    def test_convert_color_rgba_tuple(self):
        """Test that RGBA tuples use only RGB components."""
        result = self.plot._convert_color((1.0, 0.5, 0.0, 0.8))
        assert result.startswith("#")
        assert len(result) == 7  # Hex color without alpha
    
    def test_convert_color_named_colors(self):
        """Test conversion of matplotlib named colors."""
        result = self.plot._convert_color("red")
        assert result == "#ff0000"
        
        result = self.plot._convert_color("blue")
        assert result == "#0000ff"
        
        result = self.plot._convert_color("green")
        assert result.startswith("#")
    
    def test_convert_color_invalid_name_raises(self):
        """Test that invalid color names raise AttributeError."""
        with pytest.raises(AttributeError):
            self.plot._convert_color("not_a_real_color_name")
    
    def test_convert_color_none_returns_na(self):
        """Test that None values return pd.NA."""
        result = self.plot._convert_color(None)
        assert pd.isna(result)
        
        result = self.plot._convert_color(pd.NA)
        assert pd.isna(result)
    
    def test_convert_color_invalid_type_raises(self):
        """Test that invalid types raise AttributeError."""
        with pytest.raises(AttributeError):
            self.plot._convert_color(123)
        
        # Lists should fail (only tuples are valid for RGB)
        with pytest.raises(AttributeError):
            self.plot._convert_color([255, 0, 0])
    
    def test_convert_to_rgb_tuple_numeric_series(self):
        """Test conversion of numeric series to RGB tuples via colormap."""
        numeric_colors = pd.Series([0.0, 0.5, 1.0])
        result = self.plot._convert_to_rgb_tuple(numeric_colors)
        
        # Should return a series with RGB tuples
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        
        # Each value should be a tuple with RGBA components
        for val in result:
            assert isinstance(val, tuple)
            assert len(val) >= 3  # RGB or RGBA
    
    def test_convert_to_rgb_tuple_non_numeric_passthrough(self):
        """Test that non-numeric series are returned unchanged."""
        hex_colors = pd.Series(["#ff0000", "#00ff00", "#0000ff"])
        result = self.plot._convert_to_rgb_tuple(hex_colors)
        
        # Should be unchanged
        assert result.equals(hex_colors)
    
    def test_convert_to_rgb_tuple_with_custom_colormap(self):
        """Test numeric to RGB conversion respects custom colormap setting."""
        # Set custom colormap
        self.plot.config["cmap"] = "plasma"
        numeric_colors = pd.Series([0.0, 0.5, 1.0])
        result = self.plot._convert_to_rgb_tuple(numeric_colors)
        
        # Should return RGB tuples
        assert len(result) == 3
        for val in result:
            assert isinstance(val, tuple)
    
    def test_convert_to_rgb_tuple_edge_values(self):
        """Test conversion with extreme numeric values."""
        numeric_colors = pd.Series([-10.0, 0.0, 10.0])
        result = self.plot._convert_to_rgb_tuple(numeric_colors)
        
        # Should normalize and convert
        assert len(result) == 3
        for val in result:
            assert isinstance(val, tuple)
            assert len(val) >= 3
    
    def test_convert_to_rgb_tuple_single_value(self):
        """Test conversion when all values are the same."""
        numeric_colors = pd.Series([5.0, 5.0, 5.0])
        result = self.plot._convert_to_rgb_tuple(numeric_colors)
        
        # Should handle constant values
        assert len(result) == 3
        # All colors might be the same due to normalization
        unique_colors = result.unique()
        assert len(unique_colors) == 1
