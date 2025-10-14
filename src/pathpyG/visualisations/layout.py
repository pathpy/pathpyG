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
    """Function to generate a layout for the network.

    This function generates a layout configuration for the nodes in the
    network. Thereby, different layouts and options can be chosen. The layout
    function is directly included in the plot function or can be separately
    called.

    Currently supported algorithms are:

    - All layouts that are implemented in `networkx`
        - Random layout
        - Circular layout
        - Shell layout
        - Spectral layout
        - Kamada-Kawai layout
        - Fruchterman-Reingold force-directed algorithm
        - ForceAtlas2 layout algorithm
    - Grid layout

    The appearance of the layout can be modified by keyword arguments which will
    be explained in more detail below.

    Args:
        network (network object): Network to be drawn.
        weight (str or Iterable): Edge attribute that should be used as weight.
            If a string is provided, the attribute must be present in the edge
            attributes of the network. If an iterable is provided, it must have
            the same length as the number of edges in the network.
        layout (str): Layout algorithm that should be used.
        **kwargs (Optional dict): Attributes that will be passed to the layout function.
    """
    # initialize variables
    if isinstance(weight, str):
        if weight in network.edge_attrs():
            weight = network.data[weight]
        else:
            raise ValueError(f"Weight attribute '{weight}' not found in edge attributes.")
    elif isinstance(weight, Iterable) and not isinstance(weight, torch.Tensor):
        if len(weight) == network.m:
            weight = torch.tensor(weight)
        else:
            raise ValueError("Length of weight iterable does not match number of edges in the network.")

    # create layout class
    layout_cls = Layout(
        nodes=network.nodes, edge_index=network.data.edge_index, layout_type=layout, weight=weight, **kwargs
    )
    # return the layout
    return layout_cls.generate_layout()


class Layout(object):
    """Default class to create layouts.

    The [`Layout`][pathpyG.visualisations.layout.Layout] class is used to generate node a layout drawer and
    return the calculated node positions as a dictionary, where the keywords
    represents the node ids and the values represents a two dimensional tuple
    with the x and y coordinates for the associated nodes.

    Args:
        nodes (list): list with node ids.
            The list contain a list of unique node ids.
        edge_index (Tensor): Edge index of the network.
            The edge index is a tensor of shape [2, num_edges] and contains the
            source and target nodes of each edge.
        
        weight (Tensor): Edge weights of the network.
            The edge weights is a tensor of shape [num_edges] and contains the
            weight of each edge.
        **kwargs (dict): Keyword arguments to modify the layout. Will be passed
            to the layout function.
    """

    def __init__(self, nodes: list, edge_index: Optional[Tensor] = None, layout_type: str = "random", weight: Optional[Tensor] = None, **kwargs):
        """Initialize the Layout class."""
        # initialize variables
        self.nodes = nodes
        if edge_index is None:
            self.edge_index = EdgeIndex(torch.empty((2, 0), dtype=torch.long))
        else:
            self.edge_index = edge_index
        self.weight = weight
        self.layout_type = layout_type
        self.kwargs = kwargs

    def generate_layout(self):
        """Function to pick and generate the right layout."""
        # method names
        names_grid = ["grid", "2d-lattice", "lattice-2d"]
        # check which layout should be plotted
        if self.layout_type in names_grid:
            self.layout = self.grid()
        else:
            self.layout = self.generate_nx_layout()

        return self.layout

    def generate_nx_layout(self):
        """Function to generate a layout using networkx."""
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
        names_forceatlas2 = ["forceatlas2", "fa2", "forceatlas", "force-atlas", "force-atlas2", "fa 2", "fa 1"]

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
            if "k" not in self.kwargs:
                # set optimal distance between nodes
                self.kwargs["k"] = np.sqrt(len(self.nodes))
            layout = nx.spring_layout(nx_network, weight="weight" if self.weight is not None else None, **self.kwargs)
        elif self.layout_type in names_forceatlas2:
            layout = nx.forceatlas2_layout(
                nx_network, weight="weight" if self.weight is not None else None, **self.kwargs
            )
        else:
            raise ValueError(f"Layout '{self.layout_type}' not recognized.")

        return layout

    def grid(self):
        """Position nodes on a two-dimensional grid.

        This algorithm can be enabled with the keywords: `grid`, `lattice-2d`, `2d-lattice`, `lattice`

        Returns:
            layout (dict): A dictionary of positions keyed by node
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
