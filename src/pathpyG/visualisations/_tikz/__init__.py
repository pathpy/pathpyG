"""TikZ Backend for PathpyG Visualizations.

Publication-quality vector graphics backend using LaTeX's TikZ package for static networks.
Ideal for academic publications and high-quality print materials.

!!! info "Output Formats"
    - **SVG**: Scalable vector graphics for web and presentations
    - **PDF**: Print-ready documents with embedded fonts  
    - **TeX**: Raw LaTeX code for document integration

!!! warning "Requirements"
    - LaTeX distribution with TikZ package
    - `dvisvgm` for SVG output (included with TeX Live)
    - `pdflatex` for PDF output

## Basic Usage

```python
import pathpyG as pp

# Simple network visualization
edges = [("A", "B"), ("B", "C"), ("C", "A")]
g = pp.Graph.from_edge_list(edges)
pp.plot(g, backend="tikz")
```
<img src="../plot/tikz_init_basic.svg" alt="Example TikZ Custom Properties" width="550"/>

## Advanced Example

```python
import pathpyG as pp
import torch

# Graph with custom styling
edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
g = pp.Graph.from_edge_list(edges)
g.data["node_size"] = torch.tensor([15, 20, 25, 20])

pp.plot(
    g,
    backend="tikz",
    node_color={"A": "red", "B": "#00FF00"},
    edge_opacity=0.7,
    curvature=0.2,
    width="8cm",
    height="6cm",
    filename="custom_network.svg"
)
```
<img src="../plot/tikz_init_advanced.svg" alt="Example TikZ Custom Properties" width="550"/>

## Templates

PathpyG uses LaTeX templates to generate TikZ visualizations. Templates define standalone LaTeX documents with placeholders for dynamic content.
Templates are located in the `pathpyG/visualisations/_tikz/templates/` directory.
Currently supported templates:
- `static.tex`: For static networks without time dynamics.
"""
