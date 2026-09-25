import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Temporal Network Visualizations with Manim

    ## Prerequisites

    First, we need to set up our Python environment that has PyTorch, PyTorch Geometric and PathpyG installed. Depending on where you are executing this notebook, this might already be (partially) done. E.g. Google Colab has PyTorch installed by default so we only need to install the remaining dependencies. The DevContainer that is part of our GitHub Repository on the other hand already has all of the necessary dependencies installed.

    In the following, we install the packages for usage in Google Colab using Jupyter magic commands. For other environments comment in or out the commands as necessary. For more details on how to install `pathpyG` especially if you want to install it with GPU-support, we refer to our [documentation](https://www.pathpy.net/dev/getting_started/). Note that `%%capture` discards the full output of the cell to not clutter this tutorial with unnecessary installation details. If you want to print the output, you can comment `%%capture` out.
    """)
    return


@app.cell
def _():
    # %%capture
    # # !pip install torch
    # !pip install torch_geometric
    # !pip install git+https://github.com/pathpy/pathpyG.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that using the `Manim`-backend in `PathpyG` requires the installation of `Manim`. It will not be automatically installed with `PathpyG` due to its additional dependencies. We recommend using the installation instructions in the `Manim` [documentation](https://docs.manim.community/en/stable/installation/uv.html) or our provided DevContainer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motivation and Learning Objectives

    While you already learned how to visualise graphs with `PathpyG` in the [interactive graph visualisation](/tutorial/visualisation/) tutorial, those visualisations mostly considered static graphs.
    In this tutorial, you will learn how to use the `Manim` backend in `PathpyG` to generate videos (`.mp4`) or animations (`.gif`) of temporal graphs.

    ## Let's Get Started

    We need to import `PathpyG` first:
    """)
    return


@app.cell
def _():
    import os
    import tempfile

    import pathpyG as pp

    return os, pp, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we can create a temporal graph that we want to visualise:
    """)
    return


@app.cell
def _(pp):
    t = pp.TemporalGraph.from_edge_list(
        [
            ("a", "b", 1),
            ("b", "c", 5),
            ("c", "d", 9),
            ("d", "a", 9),
            ("a", "b", 10),
            ("b", "c", 10),
            ("a", "b", 8),
            ("b", "c", 13),
            ("c", "d", 17),
            ("d", "a", 19),
            ("a", "b", 20),
            ("b", "c", 18),
        ]
    )
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `plot` function using Manim as backend allows  to use a variety of customization options to visualize temporal graphs and outputs a video. Some examples are provided below, e.g. the `node_size` or the `edge_size`.
    """)
    return


@app.cell
def _(pp, t):
    pp.plot(t, backend="manim", node_size=20, edge_size=10, edge_color="red", node_color="gray");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The videos can be exported as `.mp4` or `.gif` if you provide a filename to the `plot` function.
    """)
    return


@app.cell
def _(os, pp, t, tempfile):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, "temporal_graph_manim.gif")

        pp.plot(
            t,
            backend="manim",
            node_size=20,
            edge_size=4,
            edge_color="red",
            node_color="gray",
            filename=tmp_path,
        )
        print(f"Animation saved to {tmp_path}")
        print(f"Files in temp dir: {os.listdir(tmpdirname)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dynamic Customisations and Layout

    Since we use `manim` to animate temporal networks, we can also use dynamic customisations that change over time. For example, we can change the `node_size` or `edge_size` over time.
    To change those properties dynamically, we can provide a dictionary that maps node/source-target id and the time step to the desired property value.

    Additionally, we can use dynamic layouts that are based on sliding windows. For example, we can use a `ForceAtlas2` layout that is computed based on a sliding window of the last 3 time steps. This allows us to have a layout that changes over time but is still stable enough to be visually appealing.

    The following shows an example where all of the customisations described above are used:
    """)
    return


@app.cell
def _(pp, t):
    pp.plot(
        t,
        backend="manim",
        layout_window_size=[3, 1],
        layout="fa2",
        node_size=12,
        edge_color={("b", "c", 10): "green"},
        node_color={("a", 3): "yellow", ("a", 0): "green", ("b", 3): "black"},
    );
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real-World Example

    As a final example, we show how to visualise a real-world temporal graph with `Manim`. We use [Netschleuder Online Repository](https://networks.skewed.de/) (see our next tutorial our next tutorial [Graph Learning in Netzschleuder Data](/tutorial/netzschleuder/) for more information) to obtain a temporal interaction graph between baboons and color the types of interactions in different colors.
    """)
    return


@app.cell
def _(pp):
    g = pp.io.read_netzschleuder_graph('sp_baboons', 'observational', time_attr='time')
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The timestamps are provided in seconds since epoch (converted from unix time). We first convert them to a more human-readable format (hours since the first interaction):
    """)
    return


@app.cell
def _(g):
    g.data.time -= g.data.time.min()
    g.data.time = g.data.time // (60 * 60 * 12) # convert to 12 hours intervals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset contains different types of interactions between the baboons. We can use this information to color the edges based on the interaction type:
    """)
    return


@app.cell
def _(g):
    colors = []
    for category in g.data["edge_category"]:
        match category:
            case "Affiliative":
                colors.append("red")
            case "Agonistic":
                colors.append("green")
            case "Other":
                colors.append("grey")
    return (colors,)


@app.cell
def _(colors, g, pp):
    pp.plot(
        g,
        backend="manim",
        layout_window_size=12,  # three days
        layout="kk",
        edge_color=colors,
        edge_size=2.5,
        delta=500,  # Each time step is 500 ms
    );
    return


if __name__ == "__main__":
    app.run()
