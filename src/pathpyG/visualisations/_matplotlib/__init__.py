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

## Time-Unfolded Network

We also support time-unfolded static visualizations of temporal networks using the matplotlib backend.
The example uses the `node_opacity` parameter to highlight active nodes and edges at each time step.

```python
import pathpyG as pp

# Example temporal network data
tedges = [
    ("a", "b", 1),
    ("a", "b", 2),
    ("b", "a", 3),
    ("b", "c", 3),
    ("d", "c", 4),
    ("a", "b", 4),
    ("c", "b", 4),
]
t = pp.TemporalGraph.from_edge_list(tedges)

# Create temporal plot and display inline
node_opacity = {(node_id, time): 0.1 for node_id in t.nodes for time in range(t.data.time.max().item() + 2)}
node_opacity.update({(source_id, time): 1.0 for source_id, target_id, time in t.temporal_edges})
node_opacity.update({(target_id, time+1): 1.0 for source_id, target_id, time in t.temporal_edges})
pp.plot(t, backend="matplotlib", kind="unfolded", node_size=12, node_opacity=node_opacity)
```
<img src="../plot/unfolded_graph_matplotlib.png" alt="Example Matplotlib Backend Time-Unfolded Output" width="550"/>
"""
