# Visualisation using Manim

The `_manim`submodule provides Manim-based plotting tools for visualising temporal networks.

## Overview

---

## Classes

### `ManimPlot`

Base lass for Manim visualisations integrated with Jupyter notebooks. Defines the interface for rendering and exporting animations from data.

#### Methods

- `generate()`: Abstract method to generate the plot.
- `show(**kwargs)`: Render and display inline in Jupyter Notebook
- `save(filename: str, **kwargs)`: Save animation to disk 

**kwargs** for saving Manim plots:

- `filename` (string): Name the rendered file should be given. This keyword is necessary for saving.
- `save_as` {`gif`,`mp4`}: Saving format options. Default is `mp4`
- `save_dir` (string): Directory path to save the Output to. Default is current working directory.

For rendering and inline display use the `show()` method instead of `save()`.

### `NetworkPlot`

Base class fors static and  temporal network visualisations. Stores input data and configuration

#### Parameters

- `data (dict)`: Input network dictionary
- `**kwargs`: Custom styling parameters

### `TemporalNetworkPlot`

Animation class for temporal graphs. Supports dynamic layout, time-based color changes, and further customized styling options.

#### Keyword Arguments Overview

| Argument               | Type             | Default  | Short Description                                |
|------------------------|------------------|----------|-------------------------------------------------|
| **General**            |                  |          |                                                 |
| `delta`                | int              | 1000     | Duration of timestep (ms)                        |
| `start`                | int              | 0        | Animation start timestep                         |
| `end`                  | int / None       | None     | Animation end timestep (last edge by default)   |
| `intervals`            | int              | None     | Number of animation intervals                    |
| `dynamic_layout_interval` | int           | None     | Steps between layout recalculations              |
| `background_color`     | str              | WHITE    | Background color (name, hex, RGB, or Manim)     |
| **Nodes**              |                  |          |                                                 |
| `node_size`            | float / dict     | 0.4      | Radius of nodes (uniform or per-node)            |
| `node_color`           | str / list[str]  | BLUE     | Node fill color or list of colors                 |
| `node_cmap`            | Colormap         | None     | Colormap for scalar node values                   |
| `node_opacity`         | float / dict     | 1        | Node fill opacity (0 transparent, 1 solid)       |
| `node_color_timed`     | list[tuple]      | None     | Color Changes for Nodes at timestep
| **Edges**              |                  |          |                                                 |
| `edge_size`            | float / dict     | 0.4      | Edge width (uniform or per-edge)                  |
| `edge_color`           | str / list[str]  | GRAY     | Edge line color or list of colors                  |
| `edge_cmap`            | Colormap         | None     | Colormap for scalar edge values                    |
| `edge_opacity`         | float / dict     | 1        | Edge line opacity (0 transparent, 1 solid)        |

---
##### Detailed Descriptions

###### General

- `delta`: Duration (in milliseconds) of each animation timestep.
- `start`: Starting timestep of the animation sequence.
- `end`: Ending timestep; defaults to the last timestamp of the input data.
- `intervals`: Number of discrete animation steps.
- `dynamic_layout_interval`: How often (in timesteps) the layout recomputes.
- `background_color`: Background color of the plot, accepts color names, hex codes, RGB tuples, or Manim color constants.


###### Nodes

- `node_size`: Node radius; either a single float applied to all nodes or a dictionary with sizes per node ID.
- `node_color`: Fill color(s) for nodes. Can be a single color string referred to by name, HEX, RGB, float a list of colors cycling through nodes or a dictionary with color per node
- `node_cmap`: Colormap used when node colors are numeric.
- `node_opacity`: Opacity level for nodes, either uniform or per node.
- `node_color_timed`: List containing color changes at certain time steps for a certain node. Tuples in the list follow `('node_id',(t, color))` format to indicate for a node with node_id a change to color at time t. Color can be a single color string referred to by name, HEX, RGB or float.

###### Edges

- `edge_size`: Width of edges, can be uniform or specified per edge in a dictionary with size per edge ID.
- `edge_color`: Color(s) of edges; supports single or multiple colors (see `node_color` above).
- `edge_cmap`: Colormap used when edge colors are numeric.
- `edge_opacity`: Opacity for edges, uniform or per edge.

---


#### Notable Methods

- `construct`: Core method for generating the Manim animation
- `get_layout()`: Computed node 3D positions based on temporal windows
- `get_color_at_time()`: Determines a node´s color at a given timestep
- `compute_edge_index()`: converts input data into `(source, target, time)` tuples

## Usage Example
```python
import pathpyG as pp

# Example network data
tedges = [('a', 'b', 1),('a', 'b', 2), ('b', 'a', 3), ('b', 'c', 3), ('d', 'c', 4), ('a', 'b', 4), ('c', 'b', 4)]
t = pp.TemporalGraph.from_edge_list(tedges)

# Create temporal plot with custom settings and display inline
pp.plot(
    t,
    backend= 'manim', 
    delta = 5,
    start= 1,
    end = 10,
    background_color = '#f0f0f0',
    node_size = {"a": 0.6, "b": 0.3},
    node_color = ["red", "blue"],
    edge_color = 0.4,
    edge_opacity = 0.7,
    node_color_timed =  [('a', (1, 'yellow')), ('b', (2, 'blue')), ('c', (4, 0.1)), ('b', 4, (255,0,0)]
)
```

## Notes

- The Manim config is adjusted internally for resolution, framerate and format


