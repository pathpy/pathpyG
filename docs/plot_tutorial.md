# Developing your own Plots

!!! abstract "Overview"
    Add a new histogram plot to pathpyG’s visualisation stack, wire it into [`pp.plot(...)`][pathpyG.plot], and render it with Matplotlib. This guide explains the data-prep vs. rendering split and shows the minimal pieces to implement.

This tutorial shows how to add a new plotting capability to pathpyG’s visualisation backend by implementing a histogram plot. You’ll learn how plot types, backends, and configuration work together, and how to add a new plot into the public [`pp.plot(...)`][pathpyG.plot] entry point.

**What you’ll do**

- :material-family-tree: Understand the new visualisation architecture
- :material-chart-bar: Implement a new `HistogramPlot` that prepares data
- :material-vector-link: Wire it into the plot orchestrator and select backends
- :material-image-multiple: Add Matplotlib rendering support for the new type
- :material-test-tube: Use and (optionally) test your new plot

!!! tip "Scope"
    This guide focuses on Matplotlib for rendering histograms (a natural fit). You can add other backends later following the same pattern.

## Visualisation architecture at a glance

The visualisation module is built around two core abstractions and a single entry point:

- :material-database-cog: [`PathPyPlot`][pathpyG.visualisations.pathpy_plot.PathPyPlot] prepares data/config for rendering. Subclass it for each plot type.
- :material-cog: [`PlotBackend`][pathpyG.visualisations.plot_backend.PlotBackend] renders a given [`PathPyPlot`][pathpyG.visualisations.pathpy_plot.PathPyPlot] using a concrete engine ([Matplotlib][pathpyG.visualisations._matplotlib], [TikZ][pathpyG.visualisations._tikz], [d3.js][pathpyG.visualisations._d3js], [Manim][pathpyG.visualisations._manim]).
- :material-play-circle: [`plot(...)`][pathpyG.plot] is the public API. It chooses a plot class (kind) and a backend (by argument or filename extension), instantiates both, then saves or shows.

!!! info "Reference"
    See the [module overview](/reference/pathpyG/visualisations) for supported backends, formats, and styling options. For existing plot types, see [`NetworkPlot`][pathpyG.visualisations.network_plot.NetworkPlot] (static) and [`TemporalNetworkPlot`][pathpyG.visualisations.temporal_network_plot.TemporalNetworkPlot] (temporal). For existing backends, see e.g. [`MatplotlibBackend`][pathpyG.visualisations._matplotlib.backend.MatplotlibBackend] which we will be using.

## Define a new plot type: HistogramPlot

Start by creating a new subclass of [`PathPyPlot`][pathpyG.visualisations.pathpy_plot.PathPyPlot] (e.g., in `src/pathpyG/visualisations/histogram_plot.py`). Its job is to:

- Accept the input object(s) (typically a [`Graph`][pathpyG.core.graph.Graph]) and user options
- Compute or collect the values to be binned
- Populate `self.data` with a clean, backend-agnostic structure
- Update `self.config` with plot configuration (bins, labels, etc.)

!!! info "Minimal class attributes"
    Inputs: `graph: Graph`, `key: str` (what to measure), `bins: int | sequence`, plus style options via `**kwargs`.

    Data format (suggested):
    - `self.data["hist_values"]: list[float | int]` — the values to bin
    - optionally precomputed bins/edges (if you want backend-agnostic binning)
    - `self.config` should include `title`, `xlabel`, `ylabel`, and `bins`

### Example outline:

```python
# src/pathpyG/visualisations/histogram_plot.py
from __future__ import annotations
import logging
from typing import Any
from pathpyG.visualisations.pathpy_plot import PathPyPlot
from pathpyG.core.graph import Graph

logger = logging.getLogger("root")


class HistogramPlot(PathPyPlot):
    """Prepare data for histogram visualisation.

    Collects values from a Graph according to `key` and exposes them in
    `self.data["hist_values"]` for backends to render.
    """

    _kind = "histogram"

    def __init__(self, graph: Graph, key: str = "degree", bins: int | list[int] = 10, **kwargs: Any) -> None:
        super().__init__()
        self.graph = graph
        # merge kwargs into config; ensure required fields are present
        self.config.update({
            "bins": bins,
            "title": kwargs.pop("title", f"{key.title()} distribution"),
            "xlabel": kwargs.pop("xlabel", key),
            "ylabel": kwargs.pop("ylabel", "count"),
        })
        self.key = key
        self.config.update(kwargs)
        self.generate()

    def generate(self) -> None:
        # Compute values to bin based on `key`
        if self.key in ("degree", "degrees"):
            values = list(self.graph.degrees().values())
        elif self.key in ("in_degree", "indegree", "in-degrees"):
            values = list(self.graph.degrees(mode="in").values())
        elif self.key in ("out_degree", "outdegree", "out-degrees"):
            values = list(self.graph.degrees(mode="out").values())
        else:
            logger.error(f"Histogram key '{self.key}' not supported.")
            raise KeyError(self.key)

        self.data["hist_values"] = values
```

!!! note 
    - Keep the class small: gather values and fill `self.data`/`self.config`.
    - Choose names that are clear for backends (`hist_values`, `bins`, labels).

## Add the new plot to the public API

[`plot(...)`][pathpyG.plot] uses the `PLOT_CLASSES` mapping to instantiate the right plot class for a given `kind`. Extend it with your new class:

```python
# src/pathpyG/visualisations/plot_function.py
from pathpyG.visualisations.histogram_plot import HistogramPlot

PLOT_CLASSES: dict = {
    "static": NetworkPlot,
    "temporal": TemporalNetworkPlot,
    "histogram": HistogramPlot,  # add this line
}
```

??? example "Usage"

    ```python
    import pathpyG as pp

    g = pp.Graph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c")])
    # Matplotlib is the natural backend for histograms
    pp.plot(g, kind="histogram", backend="matplotlib", key="degree", bins=10, filename="degree_hist.png")
    ```

!!! tip "Backend selection"
    [`plot(...)`][pathpyG.plot] auto-selects a backend from the filename extension if you omit `backend`. For histograms, prefer PNG via Matplotlib by passing `filename="...png"` or `backend="matplotlib"`.

## Add Matplotlib support for HistogramPlot

Backends validate supported plot types. The [Matplotlib backend][pathpyG.visualisations._matplotlib] currently supports `NetworkPlot` and renders nodes/edges. We’ll extend it to also support `HistogramPlot`.

Implementation approach:

1. Add `HistogramPlot` to `SUPPORTED_KINDS` so the backend accepts the plot type.
2. Branch in [`to_fig()`](/reference/pathpyG/visualisations/_matplotlib/backend/#pathpyG.visualisations._matplotlib.backend.MatplotlibBackend.to_fig) (or factor out into a helper) to draw a histogram when the plot is a `HistogramPlot`.

Sketch of the required changes (condensed for illustration):

```python
# src/pathpyG/visualisations/_matplotlib/backend.py
from pathpyG.visualisations.histogram_plot import HistogramPlot

SUPPORTED_KINDS = {
    NetworkPlot: "static",
    HistogramPlot: "histogram",  # add support
}

class MatplotlibBackend(PlotBackend):
    ...
    def to_fig(self) -> tuple[plt.Figure, plt.Axes]:
        # If histogram: render using ax.hist
        if self._kind == "histogram":
            return self._to_fig_histogram()
        # Else: existing network rendering
        return self._to_fig_network()

    def _to_fig_histogram(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(
            figsize=(unit_str_to_float(self.config["width"], "in"), unit_str_to_float(self.config["height"], "in")),
            dpi=150,
        )
        ax.set_axis_on()
        ax.hist(self.data["hist_values"], bins=self.config.get("bins", 10), color=rgb_to_hex(self.config["node"]["color"]), alpha=0.9)
        ax.set_title(self.config.get("title", "Histogram"))
        ax.set_xlabel(self.config.get("xlabel", "value"))
        ax.set_ylabel(self.config.get("ylabel", "count"))
        return fig, ax

    def _to_fig_network(self) -> tuple[plt.Figure, plt.Axes]:
        # move existing implementation of `to_fig` here
        ...
```

!!! tip "Tips"
    - Reuse `unit_str_to_float` so sizing behaves like other plots.
    - Use a default color from `self.config["node"]["color"]` for consistency.
    - Keep the new code path fully separate from the network drawing code to avoid regressions.

??? info "If you want web or LaTeX histograms"
    The current d3.js and TikZ backends are tailored to network visualisation (they expect `nodes`/`edges` in `self.data`). To add histogram support there, you would:

    - Create a new JS or TeX template for histograms
    - Extend the backend to accept `HistogramPlot` and dispatch to the new template

    Start with Matplotlib first — it's a good starting point.

## Try it out

Once you’ve added the `HistogramPlot`, updated `PLOT_CLASSES`, and extended the Matplotlib backend as shown, you can create and save a histogram in a single call:

```python
import pathpyG as pp

g = pp.Graph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
pp.plot(
    g,
    kind="histogram",
    backend="matplotlib",  # or infer via filename extension
    key="degree",
    bins=5,
    title="Node Degree Distribution",
    filename="degree_hist.png",
)
```

In notebooks, omit `filename` to show inline.

## Testing (optional but recommended)

Create a small unit test to exercise the new path end-to-end:

```python
# tests/visualisations/test_histogram.py
import pathpyG as pp

def test_histogram_plot_matplotlib(tmp_path):
    g = pp.Graph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c")])
    out = tmp_path / "deg_hist.png"
    pp.plot(g, kind="histogram", backend="matplotlib", key="degree", bins=3, filename=str(out))
    assert out.exists()
```

## Where to look for guidance and consistency

- :material-cog-outline: Backends: see other backends like [`Matplotlib`][pathpyG.visualisations._matplotlib] and [`d3.js`][pathpyG.visualisations._d3js] for how plot instances are validated and rendered.
- :material-database-outline: Plot classes: study [`NetworkPlot`][pathpyG.visualisations.network_plot.NetworkPlot] and [`TemporalNetworkPlot`][pathpyG.visualisations.temporal_network_plot.TemporalNetworkPlot] to understand how [`PathPyPlot`][pathpyG.visualisations.pathpy_plot.PathPyPlot] subclasses fill `self.data` and `self.config`.
- :material-file-document: The [module overview](/reference/pathpyG/visualisations) explains backend selection, saving, and common styling options.

## Recap

- :material-plus-circle: New plots are [`PathPyPlot`][pathpyG.visualisations.pathpy_plot.PathPyPlot] subclasses that prepare data and config.
- :material-merge: Register your plot in `PLOT_CLASSES` so [`pp.plot(..., kind=...)`][pathpyG.plot] can instantiate it.
- :material-image-multiple: Extend at least one backend to render your plot type. For histograms, Matplotlib is a clean first target.
- :material-link-variant: Keep a small, clear data contract between your plot class and backend rendering.

With this, you have a clean, maintainable path to add new visualisations to pathpyG while leveraging the unified [`pp.plot(...)`][pathpyG.plot] API and existing backend infrastructure.
