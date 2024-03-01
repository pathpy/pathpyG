"""Random Graph Generation Algorithms"""

from typing import Optional, Union, Dict

import torch

import pathpyG as pp


def Watts_Strogatz(n: int, s: int, p: float = 0.0, mapping: pp.IndexMap = None) -> pp.Graph:
    """Generate a Watts-Strogatz small-world graph.

    Args:
        n: The number of nodes in the graph.
        s: The number of edges to attach from a new node to existing nodes.
        p: The probability of rewiring each edge.
        mapping: A mapping from the node indices to node names.

    Returns:
        Graph: A Watts-Strogatz small-world graph.

    Examples:
        ```py
        g = Watts_Strogatz(100, 4, 0.1, mapping=pp.IndexMap([f"n_{i}" for i in range(100)])
        ```
    """

    nodes = torch.arange(n)

    # construct a ring lattice (dimension 1)
    edges = (
        torch.stack([torch.stack((nodes, torch.roll(nodes, shifts=-i, dims=0))) for i in range(1, s + 1)], dim=0)
        .permute(1, 0, 2)
        .reshape(2, -1)
    )

    # Rewire each link with probability p
    rand_vals = torch.rand(edges.shape[1])
    rewire_mask = rand_vals < p

    # Generate random nodes excluding the current node for each edge that needs to be rewired, also avoid duplicate edges
    for i in range(edges.shape[1]):
        if rewire_mask[i]:
            while True:
                new_node = torch.randint(n, (1,)).item()
                if new_node != edges[0, i] and new_node not in edges[1, edges[0] == edges[0, i]] and new_node not in edges[0, edges[1] == edges[0, i]]:
                    edges[1, i] = new_node
                    break

    g = pp.Graph.from_edge_index(edges, mapping=mapping)

    g = g.to_undirected()
    return g
