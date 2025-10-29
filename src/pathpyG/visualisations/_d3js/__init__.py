"""D3.js Backend for PathpyG Visualizations.

Interactive web-based visualization backend using D3.js for both static and temporal networks.
Default backend providing rich interactivity, real-time exploration, and web-compatible output.

!!! info "Output Formats"
    - **HTML**: Interactive web visualizations where nodes can be dragged around and temporal graphs can be paused/played.

!!! note "Default Backend"
    D3.js is the default visualization backend for PathpyG, automatically
    selected when no specific backend is specified. No additional
    dependencies required beyond web browser.

## Basic Usage

```python
import pathpyG as pp

# Simple network visualization
edges = [("A", "B"), ("B", "C"), ("C", "A")]
g = pp.Graph.from_edge_list(edges)
pp.plot(g)  # Uses d3.js backend by default
```
<iframe src="../plot/simple_network.html" width="650" height="520"></iframe>

## Advanced Temporal Network Example

```python
import torch
import pathpyG as pp

# Temporal network with evolving properties
tedges = [
    ("a", "b", 1), ("b", "c", 1),
    ("c", "d", 2), ("d", "a", 2), 
    ("a", "c", 3), ("b", "d", 3)
]
tg = pp.TemporalGraph.from_edge_list(tedges)
tg.data["edge_color"] = torch.arange(tg.m)  # Assign a unique color index to each edge

pp.plot(
    tg,
    delta=750,  # 0.75 seconds per timestep
    node_size={("a", 1): 20, ("b", 2): 7},
    node_color=["red", "blue", "green", "orange"],
    edge_opacity=0.7,
    filename="dynamic_network.html"
)
```
<iframe src="../plot/dynamic_network.html" width="650" height="520"></iframe>

## Network Visualization with custom Images

```python
import torch
import pathpyG as pp

# Example network data
edges = [
    ("b", "a"),
    ("c", "a"),
]
mapping = pp.IndexMap(["a", "b", "c", "d"])
g = pp.Graph.from_edge_list(edges, mapping=mapping)
g.data["node_size"] = torch.tensor([25]*4)
pp.plot(
    g,
    node_size={"d": 50},
    edge_size=5,
    node_image={
        "a": "https://avatars.githubusercontent.com/u/52822508?s=48&v=4",
        "b": "https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo.png",
        "c": "https://pytorch-geometric.readthedocs.io/en/latest/_static/img/pytorch_logo.svg",
        "d": "docs/img/pathpy_logo_new.png",
    },
    show_labels=False,
)
```
<iframe src="../plot/d3js_custom_node_images.html" width="650" height="520"></iframe>

!!! tip "Deployment Options"
    - **Standalone**: Self-contained HTML with embedded resources
    - **Jupyter**: Direct display in notebook cells
    - **Web Apps**: Easy integration into existing websites

## Templates
PathpyG uses HTML templates to generate D3.js visualizations located in the `templates` directory.
Templates define the overall structure and include placeholders for dynamic content.
Currently used templates:

- `network.js`: A basic template for static and temporal networks
- `setup.js`: Loads requireJS and D3.js libraries
- `styles.css`: Basic CSS styling for the visualizations
- `static.js`: Template for static networks that initializes the network from `network.js`
- `temporal.js`: Template for temporal networks that initializes the network from `network.js` with temporal controls
- `d3.v7.min.js`: D3.js library (version 7) for using D3.js functionalities without internet connection
"""