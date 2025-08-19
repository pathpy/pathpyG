# Getting Started

The following will guide you through the installation of the package and the first steps to use it.

## Prerequisites

PathpyG is available for :python_logo: Python versions 3.10 and above. It is not recommended to install it on your system Python. Instead, we recommend using a virtual environment such as [:conda_logo: conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/en/latest/). You can also set up a :docker_logo: Docker image as described in the [next section](docker_installation.md).

## Installation

Once you have an environment up and running, you can install the package simply via pip. But first make sure that you installed the necessary dependencies.

### Dependencies

This package is based on [:pytorch_logo: PyTorch](https://pytorch.org/) and [:pyg_logo: PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Please install both libraries before installing PathpyG. You can follow the installation instructions in their respective documentation ([:pytorch_logo: PyTorch](https://pytorch.org/get-started/locally/) and [:pyg_logo: PyG](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html)).

!!! warning
    We currently only support PyG version 2.5.0 and above.

### Install Stable Release

You can install the latest stable release of PathpyG via pip:

!!! warning "TODO"
    This is not yet available. We will release the first stable version soon.

```bash
pip install pathpyg
```

### Install Latest Development Version

If you want to install the latest development version, you can do so via pip directly from the GitHub repository:

```bash
pip install git+https://github.com/pathpy/pathpyG.git
```

### Optional Visualisation Backends

We provide multiple visualisation backends for PathpyG. The default backend [D3.js](https://d3js.org/) does not require any additional dependencies. We further provide a [Matplotlib](https://matplotlib.org/) backend that is installed by default. Additionally, we implemented a [Manim](https://www.manim.community/) backend that is not installed by default due to its dependencies that are required for installation. Please refer to the [Manim installation instructions](https://docs.manim.community/en/stable/installation/uv.html) for more information. Once installed, you can use the Manim backend for visualisation by setting the `backend` in the `PathpyG.plot` function to `manim`: 
```python
import pathpyg as pp

t_graph = TemporalGraph.from_edge_list([('a', 'b', 1),('b', 'a', 3), ('b', 'c', 3)])
pp.plot(t_graph, backend='manim')
```