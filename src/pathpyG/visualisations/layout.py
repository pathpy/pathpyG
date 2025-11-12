"""Network layout algorithms for node positioning.

Provides comprehensive layout computation for network visualization using various
algorithms from NetworkX and custom implementations. Supports both weighted and 
unweighted networks with flexible parameter configuration.

!!! abstract "Key Features"
    - NetworkX integration for proven algorithms
    - Custom grid layout for regular structures  
    - Weighted layout support for better positioning
    - Automatic algorithm selection and validation

!!! info "Available Algorithms"
    - All layouts that are implemented in `networkx`
        - Random layout
        - Circular layout
        - Shell layout
        - Spectral layout
        - Kamada-Kawai layout
        - Fruchterman-Reingold force-directed algorithm
        - ForceAtlas2 layout algorithm
    - Grid layout

Examples:
    Compute a spring layout for a simple graph:

    >>> from pathpyG import Graph
    >>> from pathpyG.visualisations import layout
    >>> 
    >>> g = Graph.from_edge_list([('a', 'b'), ('b', 'c')])
    >>> positions = layout(g, layout='spring', k=0.5)
    >>> print(positions)
    {'a': array(...), 'b': array(...), 'c': array(...)}
"""
#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : layout.py -- Module to layout the network
# Author    : Juergen Hackl <hackl@ibi.baug.ethz.ch>
# Creation  : 2018-07-26
# Time-stamp: <Wed 2020-04-22 15:40 juergen>
#
# Copyright (c) 2018 Juergen Hackl <hackl@ibi.baug.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.utils import to_scipy_sparse_matrix

from pathpyG.core.graph import Graph


def layout(network: Graph, layout: str = "random", weight: None | str | Iterable = None, **kwargs):
    """Generate node positions using specified layout algorithm.

    Computes 2D coordinates for all nodes in the network using various layout
    algorithms. Supports edge weighting for physics-based layouts and provides
    flexible parameter passing to underlying algorithms.

    Args:
        network: Graph instance to generate layout for
        layout: Algorithm name (see supported algorithms below)
        weight: Edge weights as attribute name, iterable, or None
        **kwargs: Algorithm-specific parameters passed to layout function

    Returns:
        dict: Node positions as {node_id: (x, y)} coordinate mapping

    Raises:
        ValueError: If weight attribute not found or weight length mismatch
        ValueError: If layout algorithm not recognized

    Examples:
        ```python
        # Basic spring layout
        pos = layout(graph, 'spring')
        
        # Weighted layout with edge attribute
        pos = layout(graph, 'kamada-kawai', weight='edge_weight')
        
        # Custom parameters
        pos = layout(graph, 'spring', k=0.3, iterations=100)
        ```

    !!! note "Supported Algorithms"
        
        | Algorithm | Aliases | Best For |
        |-----------|---------|----------|
        | `spring` | `fruchterman-reingold`, `fr` | General networks |
        | `kamada-kawai` | `kk`, `kamada` | Small/medium networks |
        | `forceatlas2` | `fa2`, `force-atlas2` | Large networks |
        | `circular` | `circle`, `ring` | Cycle structures |
        | `shell` | `concentric` | Hierarchical data |
        | `grid` | `lattice-2d` | Regular structures |
        | `spectral` | `eigen` | Community detection |
        | `random` | `rand` | Testing/baseline |
    """
    # initialize variables
    if isinstance(weight, str):
        if weight in network.edge_attrs():
            weight = network.data[weight]
        else:
            raise ValueError(f"Weight attribute '{weight}' not found in edge attributes.")
    elif isinstance(weight, Iterable) and not isinstance(weight, torch.Tensor):
        n_edges = network.m * 2 if network.is_undirected() else network.m
        if len(weight) == n_edges:  # type: ignore[arg-type]
            weight = torch.tensor(weight)
        else:
            raise ValueError("Length of weight iterable does not match number of edges in the network.")

    # create layout class
    layout_cls = Layout(
        nodes=network.nodes, edge_index=network.data.edge_index, layout_type=layout, weight=weight, **kwargs  # type: ignore[arg-type]
    )
    # return the layout
    return layout_cls.generate_layout()


class Layout(object):
    """Layout computation engine for network node positioning.

    Core class that handles algorithm selection, parameter management, and
    coordinate generation. Integrates with NetworkX for proven algorithms
    while providing custom implementations for specialized cases.

    Args:
        nodes: List of unique node identifiers
        edge_index: Tensor containing source/target indices for each edge
        layout_type: Algorithm name for position computation
        weight: Optional edge weights as tensor with shape [num_edges]
        **kwargs: Algorithm-specific parameters

    Attributes:
        nodes: Node identifier list
        edge_index: Edge connectivity tensor
        weight: Edge weight tensor (optional)
        layout_type: Selected algorithm name
        kwargs: Algorithm parameters
    """

    def __init__(self, nodes: list, edge_index: Optional[Tensor] = None, layout_type: str = "random", weight: Optional[Tensor] = None, **kwargs):
        """Initialize layout computation with network data and parameters.
        
        Args:
            nodes: List of unique node identifiers
            edge_index: Edge connectivity tensor (creates empty if None)
            layout_type: Algorithm name for position computation
            weight: Optional edge weights tensor
            **kwargs: Algorithm-specific parameters
        """
        # initialize variables
        self.nodes = nodes
        if edge_index is None:
            self.edge_index = EdgeIndex(torch.empty((2, 0), dtype=torch.long))
        else:
            self.edge_index = edge_index
        self.weight = weight
        self.layout_type = layout_type.lower()
        self.kwargs = kwargs

    def generate_layout(self):
        """Select and execute appropriate layout algorithm.
        
        Routes computation to either custom grid implementation or 
        NetworkX-based algorithms based on layout_type specification.
        
        Returns:
            dict: Node positions as {node_id: (x, y)} coordinate mapping
        """
        # method names
        names_grid = ["grid", "2d-lattice", "lattice-2d"]
        # check which layout should be plotted
        if self.layout_type in names_grid:
            self.layout = self.grid()
        else:
            self.layout = self.generate_nx_layout()

        return self.layout

    def generate_nx_layout(self):
        """Compute layout using NetworkX algorithms with weight support.
        
        Converts pathpyG network to NetworkX format, applies selected algorithm
        with proper weight handling, and returns position dictionary.
        
        Returns:
            dict: Node positions from NetworkX layout algorithm
            
        Raises:
            ValueError: If layout algorithm name not recognized
            
        !!! note "Algorithm Mapping"
            Multiple aliases map to the same underlying NetworkX function
            for user convenience and compatibility with different naming conventions.
        """
        import networkx as nx

        sp_matrix = to_scipy_sparse_matrix(self.edge_index.as_tensor(), edge_attr=self.weight, num_nodes=len(self.nodes))
        nx_network = nx.from_scipy_sparse_array(sp_matrix)
        nx_network = nx.relabel_nodes(nx_network, {i: node for i, node in enumerate(self.nodes)})

        names_rand = ["random", "rand", None]
        names_circular = ["circular", "circle", "ring", "1d-lattice", "lattice-1d"]
        names_shell = ["shell", "concentric", "concentric-circles", "shell layout"]
        names_spectral = ["spectral", "eigen", "spectral layout"]
        names_kk = ["kamada-kawai", "kamada_kawai", "kk", "kamada", "kamada layout"]
        names_fr = ["fruchterman-reingold", "fruchterman_reingold", "fr", "spring_layout", "spring layout", "spring"]
        names_forceatlas2 = ["forceatlas2", "fa2", "forceatlas", "force-atlas", "force-atlas2", "fa 2"]

        if self.layout_type in names_rand:
            layout = nx.random_layout(nx_network, **self.kwargs)
        elif self.layout_type in names_circular:
            layout = nx.circular_layout(nx_network, **self.kwargs)
        elif self.layout_type in names_shell:
            layout = nx.shell_layout(nx_network, **self.kwargs)
        elif self.layout_type in names_spectral:
            layout = nx.spectral_layout(nx_network, **self.kwargs)
        elif self.layout_type in names_kk:
            layout = nx.kamada_kawai_layout(
                nx_network, weight="weight" if self.weight is not None else None, **self.kwargs
            )
        elif self.layout_type in names_fr:
            layout = nx.spring_layout(nx_network, weight="weight" if self.weight is not None else None, **self.kwargs)
        elif self.layout_type in names_forceatlas2:
            layout = nx.forceatlas2_layout(
                nx_network, weight="weight" if self.weight is not None else None, **self.kwargs
            )
        else:
            raise ValueError(f"Layout '{self.layout_type}' not recognized.")

        return layout

    def grid(self):
        """Position nodes on regular 2D grid for lattice-like structures.

        Arranges nodes in a square grid pattern with uniform spacing.
        Useful for regular networks, lattices.

        Returns:
            dict: Grid positions as {node_id: (x, y)} coordinates
        """
        n = len(self.nodes)
        width = 1.0

        # number of nodes in horizontal/vertical direction
        k = np.floor(np.sqrt(n))
        dist = width / k

        x = (np.arange(0, n) % k) * dist
        y = -(np.floor(np.arange(0, n) / k)) * dist
        coords = np.vstack((x, y)).T

        return {node: coords[i] for i, node in enumerate(self.nodes)}
