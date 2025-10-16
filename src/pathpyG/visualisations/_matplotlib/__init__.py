"""Matplotlib Backend for PathpyG Visualizations.

Raster graphics backend using matplotlib for static network images.

!!! info "Output Formats"
    - **PNG**: High-quality raster images for presentations
    - **JPG**: Compressed raster images for web usage

## Basic Usage

```python
import pathpyG as pp

# Simple network visualization
edges = [("A", "B"), ("B", "C"), ("C", "A")]
g = pp.Graph.from_edge_list(edges)
pp.plot(g, backend="matplotlib")
```
<img src="../plot/network.png" alt="Example Matplotlib Backend Output" width="550"/>
"""
