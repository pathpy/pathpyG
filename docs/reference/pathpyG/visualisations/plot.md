# PathpyG Visualisations

This page provides an overview of the available visualisations and the supported backends.
It also describes which displaying and saving options are available as well as the supported keyword arguments for customized plot styling.

---

**Methods**

- `show(**kwargs)`: Show Visualisation
- `save(filename: str, **kwargs)`: Save Visualisation to hard drive

**kwargs** for saving Manim plots:

- `filename` (`str`): Name to assign to the output file. This keyword is necessary for saving.

For display use the `show()` method instead of `save()`.


## Supported Features by Backend

| Backend       | Static Networks  | Temporal Networks  | Available File Formats| 
|---------------|------------|-------------|----------------------|
| **d3.js**     | ✔️         | ✔️           | `svg`, `html`, `json` (dynamic) |
| **manim**     | ❌         | ✔️           | `mp4`, `gif`       | 
| **matplotlib**| ✔️         | ❌           | `png`, `svg`, `pdf`, etc.|
| **tikz**      | ✔️         | ❌           | `pdf`, `tex`|



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
| **Nodes**                 |           |           |           |           |                                               |
| `node_size`               |     ✔️    |     ✔️    |      ✔️   |    ✔️     | Radius of nodes (uniform or per-node)         |
| `node_color`              |     🟨    |     ✔️    |     🟨    |     🟨    | Node fill color           |
| `node_cmap`               |     ✔️    |     ✔️    |      ✔️    |     ✔️    | Colormap for scalar node values               |
| `node_opacity`            |      ✔️   |     ✔️    |      ✔️   |      ✔️   | Node fill opacity (0 transparent, 1 solid)    |
| `node_label`              |      ✔️   |      ✔️   |     ❌    |       ?  | Label text shown with nodes  |
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

For more details and usage examples, see [Visualisation Tutorial](/tutorial/visualisation) and ["Develop your own plot Functions](/plot_tutorial)