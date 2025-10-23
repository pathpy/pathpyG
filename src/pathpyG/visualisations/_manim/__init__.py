"""Manim Backend for PathpyG Visualizations.

High-quality animation backend using Manim for temporal networks and dynamic visualizations.
Perfect for creating engaging presentations, educational content, and scientific animations.

!!! info "Output Formats"
    - **MP4**: High-quality video animations for presentations
    - **GIF**: Animated graphics for web and social media

!!! warning "Requirements"
    - Manim Community Edition (`pip install manim`)
    - FFmpeg for video rendering
    - LaTeX distribution for mathematical text

## Basic Usage

```python
import pathpyG as pp

# Simple temporal network animation
tedges = [("a", "b", 1), ("b", "c", 2), ("c", "a", 3)]
tg = pp.TemporalGraph.from_edge_list(tedges)
pp.plot(tg, backend="manim", filename="temporal_network.mp4")
```

<video width="550" height="350" controls>
  <source src="../plot/temporal_network.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Advanced Example

```python
import pathpyG as pp

# Temporal network with evolving properties
tedges = [
    ("a", "b", 1), ("b", "c", 1),
    ("c", "d", 2), ("d", "a", 2), 
    ("a", "c", 3), ("b", "d", 3)
]
tg = pp.TemporalGraph.from_edge_list(tedges)

pp.plot(
    tg,
    backend="manim",
    delta=2000,                    # 2 seconds per timestep
    node_size={("a", 1): 20, ("b", 2): 7},
    node_color=["red", "blue", "green", "orange"],
    edge_opacity=0.7,
    edge_color={("a", "b", 1): "purple", ("c", "d", 2): "orange"},
    filename="dynamic_network.mp4"
)
```
<video width="550" height="350" controls>
  <source src="../plot/dynamic_network.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

!!! warning "Rendering Time"
    High-quality animations can take significant time to render.
    A 60-second animation of a medium-sized network at high quality 
    may take 5-30 minutes depending on the hardware specifications.
"""
