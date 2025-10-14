# PathpyG Visualisations

This page provides an overview of the available visualisations and the supported backends.
It also describes which displaying and saving options are available as well as the supported keyword arguments for customized plot styling.

---

## Overview

The main plotting function is `pathpyG.plot()`, which can be used to create visualisations of both static and temporal networks.
The function supports multiple backends, each with its own capabilities and output formats.
The backend will be automatically chosen depending on the input data and the specified options.

The default backend is `d3.js`, which is suitable for both static and temporal networks and produces interactive visualisations that can be viewed in a web browser.

!!! example "Interactive Temporal Graph Visualisation with d3.js"

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
        ("c", "d", 5),
        ("b", "a", 5),
        ("c", "b", 6),
    ]
    t = pp.TemporalGraph.from_edge_list(tedges)

    # Create temporal plot and display inline
    pp.plot(t)
    ```
    <iframe src="plot/d3js_temporal.html" width="650" height="520"></iframe>

    ??? example "Interactive Static Graph Visualisation with d3.js"
        
        ```python
        import pathpyG as pp

        # Example network data
        edges = [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "a"),
        ]
        g = pp.Graph.from_edge_list(edges)
        pp.plot(g)
        ```
        <iframe src="plot/d3js_static.html" width="650" height="520"></iframe>

## Customisation and Other Backends

For more advanced visualisations, `PathpyG` offers customisation options for node and edge properties (like `color`, `size`, and `opacity`), as well as support for additional backends, including `manim`, `matplotlib`, and `tikz`.
We provide some usage examples below, and a detailed overview of the supported keyword arguments for each backend in section [Customisation Options](#customisation-options).

### Visualising Undirected Networks

We provide support for directed and undirected static networks.
Directed networks are visualised with arrows, while undirected networks use simple lines in all backends.
We provide an example using `matplotlib` below.

!!! example "Undirected Static Graph Visualisation with `matplotlib`"
    
    You will see below that compared to the examples above, the nodes do not have arrows indicating directionality.
    ```python
    import torch
    import pathpyG as pp

    # Example undirected network data
    edge_index = torch.tensor([[0, 1, 3, 3], [1, 2, 1, 0]])
    g = pp.Graph.from_edge_index(edge_index).to_undirected()

    # Create static plot and display inline
    pp.plot(g, backend="matplotlib")
    ```
    <img src="plot/matplotlib_undirected.png" alt="Example Matplotlib Undirected" width="320"/>

    !!! tip "Node Labels"
        In the above picture, the nodes do not have labels.
        This is because labels are automatically generated based on the node IDs provided in `g.mapping.node_ids`.
        When we created the graph using the `from_edge_index()` method, we did not provide any specific node IDs, so no IDs were assigned and no labels were generated.
        You can override the default behaviour by specifying `show_labels=True` in the `pp.plot()` function call.

### Node and Edge Customisation
#### Static Networks

In all backends, you can customise the `size`, `color`, and `opacity` of nodes and edges. 
You can specify these properties in three different ways either as arguments in the `pp.plot()` function or as attributes of the graph object:

- A single value (applied uniformly to all nodes/edges)
- A list of values with length equal to the number of nodes/edges (values are applied in order)
- A dictionary mapping node/edge IDs to values (values are applied based on the IDs)

For `color`, you can use color names (e.g., `"blue"`), HEX codes (e.g., `"#ff0000"`), or RGB tuples (e.g., `(255, 0, 0)`).
You can also pass numeric values, which will be mapped to colors using a `matplotlib` colormap (specified via `cmap`).

!!! example "Custom Node and Edge Properties"
    
    In the example below, we set custom properties for nodes and edges using all three methods.
    ```python
    import torch
    import pathpyG as pp

    # Example network data
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "d"),
        ("c", "d"),
        ("d", "a"),
    ]
    g = pp.Graph.from_edge_list(edges)

    # Add properties as attributes to the graph
    g.data["node_size"] = torch.tensor([10, 15, 20, 15])
    g.data["edge_color"] = torch.tensor([0, 1, 2, 1, 0])
    g.data["node_opacity"] = torch.zeros(g.n)

    # Create static plot with custom settings and display inline
    pp.plot(
        g,
        backend="tikz",
        node_color={"a": "red", "b": "#00FF00"},
        edge_opacity={("a", "b"): 0.1, ("a", "c"): 0.5, ("b", "d"): 1.0},
        node_opacity=1.0,  # override graph attribute
        edge_size=torch.tensor([1, 2, 3, 2, 1]),
    )
    ```
    <img src="plot/tikz_custom_properties.svg" alt="Example TikZ Custom Properties" width="320"/>

    ??? tip "Display Images inside your Nodes"
        `d3.js` additionally supports images as node representations.
        You can specify the image source using the `node_image` argument.
        The image source can be a URL or a local file path.
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
        <iframe src="plot/d3js_custom_node_images.html" width="650" height="520"></iframe>

#### Temporal Networks

For temporal networks, you can also customise the `size`, `color`, and `opacity` of nodes and edges at each timestep.
In our understanding, a temporal network has a fixed set of nodes, but edges appear at different timesteps.
Thus, all nodes exist at all times, but edges may only exist at certain timesteps.
Therefore, edge properties can be specified for each timestep where the edge exists.
In contrast, node properties can change at specified points in time, but will remain the same for all subsequent timesteps until they are changed again.

!!! example "Custom Node and Edge Properties in Temporal Networks"

## Customisation Options

| Backend       | Static Networks  | Temporal Networks  | Available File Formats| 
|---------------|------------|-------------|--------------|
| **d3.js**     | ✔️         | ✔️           | `html` |
| **manim**     | ❌         | ✔️           | `mp4`, `gif` | 
| **matplotlib**| ✔️         | ❌           | `png` |
| **tikz**      | ✔️         | ❌           | `svg`, `pdf`, `tex`|



## Keyword Arguments Overview
| Argument                  | d3.js | manim | matplotlib | tikz | Short Description                             |
| ------------------------- | :-----: | :-----: | :-----: | :-----: | --------------------------------------------- |
| **General**               |           |           |           |           |                                               |
| `delta`                   |    ✔️    |     ✔️    |   ❌      |         | Duration of timestep (ms)                      |
| `start`                   |    ✔️    |     ✔️    |   ❌      |         | Animation start timestep                      |
| `end`                     |    ✔️    |     ✔️    |   ❌      |         | Animation end timestep (last edge by default) |
| `intervals`               |    ✔️    |     ✔️    |    ❌      |         | Number of animation intervals                 |
| `dynamic_layout_interval` |    ❌    |     ✔️    |     ❌     |         | Steps between layout recalculations          |
| `background_color`        |     ❌   |     ✔️    |     ❌    |         | Background color (name, hex, RGB)   |
| `width`                   |     ✔️   |      ❌   |      ❌   |         | Width of the output               |
| `height`                  |     ✔️   |      ❌   |      ❌   |         | Height of the output                   |
| `lookahead`               |      ❌  |      ✔️  |      ❌   |    ❌     | for layout computation                |
| `lookbehind`              |     ❌   |      ✔️   |      ❌   |    ❌     | for layout computation                  |
| **Nodes**                 |           |           |           |           |                                               |
| `node_size`               |     ✔️    |     ✔️    |      ✔️   |    ✔️     | Radius of nodes (uniform or per-node)         |
| `node_color`              |     🟨    |     ✔️    |     🟨    |     🟨    | Node fill color           |
| `node_cmap`               |     ✔️    |     ✔️    |      ✔️    |     ✔️    | Colormap for scalar node values               |
| `node_opacity`            |      ✔️   |     ✔️    |      ✔️   |      ✔️   | Node fill opacity (0 transparent, 1 solid)    |
| `node_label`              |      ✔️   |      ✔️   |     ❌    |         | Label text shown with nodes  |
| **Edges**                 |           |           |           |           |                                               |
| `edge_size`               |      ✔️   |     ✔️    |    ✔️     |    ✔️     | Edge width (uniform or per-edge)              |
| `edge_color`              |     ✔️    |     ✔️    |    ✔️     |     ✔️    | Edge line color          |
| `edge_cmap`               |     ✔️    |     ✔️    |     ✔️     |    ✔️     | Colormap for scalar edge values               |
| `edge_opacity`            |     ✔️    |     ✔️    |     ✔️    |    ✔️     | Edge line opacity (0 transparent, 1 solid)    |

**Legend:** ✔️ Supported 🟨 Partially Supported ❌ Not Supported 

### Detailed Description of Keywords
The default values may differ for each individual Backend.

#### General

- `delta` (int): Duration (in milliseconds) of each animation timestep.
- `start` (int): Starting timestep of the animation sequence.
- `end`(int or None): Ending timestep; defaults to the last timestamp of the input data.
- `intervals`(int): Number of discrete animation steps.
- `dynamic_layout_interval` (int): How often (in timesteps) the layout recomputes.
- `background_color`(str or tuple): Background color of the plot, accepts color names, hex codes or RGB tuples.
- `width`  (int): width of the output
- `height` (int): height of the output
- `look_ahead` (int): timesteps in the future to include while calculating layout
- `look_behind` (int): timesteps into the past to include while calculating layout



#### Nodes

- `node_size`: Node radius; either a single float applied to all nodes or a dictionary with sizes per node ID.
- `node_color`: Fill color(s) for nodes. Can be a single color string referred to by name (`"blue"`), HEX (`"#ff0000"`), RGB(`(255,0,0)`), float, a list of colors cycling through nodes or a dictionary with color per node in one of the given formats.
**Manim** additionally supports timed node color changes in the format `{"node_id-timestep": color}` (i.e. `{a-2.0" : "yellow"}`) 
- `node_cmap`: Colormap used when node colors are numeric.
- `node_opacity`: Opacity level for nodes, either uniform or per node.
- `node_label` (dict): Assign text labels to nodes

#### Edges

- `edge_size`: Width of edges, can be uniform or specified per edge in a dictionary with size per edge ID.
- `edge_color`: Color(s) of edges; supports single or multiple colors (see `node_color` above).
- `edge_cmap`: Colormap used when edge colors are numeric.
- `edge_opacity`: Opacity for edges, uniform or per edge.

---
## Usage Examples



**manim**
```python
import pathpyG as pp

# Example network data
tedges = [('a', 'b', 1),('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4),
              ('c', 'd', 5), ('b', 'a', 5), ('c', 'b', 6)]
t = pp.TemporalGraph.from_edge_list(tedges)

# Create temporal plot with custom settings and display inline
pp.plot(
    t,
    backend="manim",
    dynamic_layout_interval=1,
    edge_color={"b-a-3.0": "red", "c-b-4.0": (220,30,50)},
    node_color = {"c-3.0" : "yellow"},
    edge_size=6,
    node_label={"a": "a", "b": "b", "c": "c", "d" : "d"},
    font_size=20,
)
```

<style>
.video-wrapper {
    position: relative;
    display: block;
    height: 0;
    padding: 0;
    overflow: hidden;
    padding-bottom: 56.25%;
  }
  .video-wrapper > iframe {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
  }
</style>

<div class="video-wrapper">
<iframe width="1280" height="720" src="demo_manim.mp4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"></iframe>
</div>


**matplotlib**
```python
import pathpyG as pp

# Example network data (static)
g = Graph.from_edge_index(torch.tensor([[0,1,0], [2,2,1]]))

# Create static plot with custom settings and display inline
pp.plot(
    g,
    backend= 'matplotlib', 
    edge_color= "grey",
    node_color = "blue"
)
```
<img src="demo_matplotlib.png" alt="Example Matplotlib" width="320"/>





---
For more details and usage examples, see [Manim Visualisation Tutorial](/tutorial/manim_tutorial),[Visualisation Tutorial](/tutorial/visualisation) and [Develop your own plot Functions](/plot_tutorial)
