"""Random Graph Generation Algorithms"""

import warnings

import torch

import pathpyG as pp


def Watts_Strogatz(
    n: int,
    s: int,
    p: float = 0.0,
    undirected: bool = True,
    allow_duplicate_edges: bool = True,
    allow_self_loops: bool = True,
    mapping: pp.IndexMap | None = None,
) -> pp.Graph:
    """Generate a Watts-Strogatz small-world graph.

    Args:
        n: The number of nodes in the graph.
        s: The number of edges to attach from a new node to existing nodes.
        p: The probability of rewiring each edge.
        undirected: If True, the graph will be undirected.
        allow_duplicate_edges: If True, allow duplicate edges in the graph.
            This is faster but may result in fewer edges than requested in the undirected case
            or duplicates in the directed case.
        allow_self_loops: If True, allow self-loops in the graph.
            This is faster but may result in fewer edges than requested in the undirected case.
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

    if not allow_duplicate_edges:
        if n * (n - 1) < edges.shape[1]:
            raise ValueError(
                "The number of edges is greater than the number of possible edges in the graph. Set `allow_duplicate_edges=True` to allow this."
            )
        elif n * (n - 1) * 0.5 < edges.shape[1] and p > 0.3:
            warnings.warn(
                "Avoding duplicate in graphs with high connectivity and high rewiring probability may be slow. Consider setting `allow_duplicate_edges=True`."
            )

    # Rewire each link with probability p
    rand_vals = torch.rand(edges.shape[1])
    rewire_mask = rand_vals < p

    # Generate random nodes excluding the current node for each edge that needs to be rewired, also avoid duplicate edges
    edges[1, rewire_mask] = torch.randint(n, (rewire_mask.sum(),))

    # In the undirected case, make sure the edges all point in the same direction
    # to avoid duplicate edges pointing in opposite directions
    if undirected:
        edges = edges.sort(dim=0)[0]
    final_edges = edges

    if not allow_duplicate_edges:
        # Remove duplicate edges
        final_edges, counts = edges.unique(dim=1, return_counts=True)
        if final_edges.shape[0] < edges.shape[1]:
            for i, edge in enumerate(final_edges[:, counts > 1].T):
                for _ in range(counts[counts > 1][i] - 1):
                    while True:
                        new_edge = torch.tensor([edge[0], torch.randint(n, (1,))]).sort()[0].unsqueeze(1)
                        # Check if the new edge is already in the final edges
                        # and add it if not
                        if (new_edge != final_edges).any(dim=0).all():
                            final_edges = torch.cat((final_edges, new_edge), dim=1)
                            break

    if not allow_self_loops:
        self_loop_edges = final_edges[:, final_edges[0] == final_edges[1]]
        final_edges = final_edges[:, final_edges[0] != final_edges[1]]
        for self_loop_edge in self_loop_edges.T:
            while True:
                new_edge = torch.tensor([self_loop_edge[0], torch.randint(n, (1,))]).sort()[0].unsqueeze(1)
                # Check if the new edge is already in the final edges
                # and add it if not
                if (new_edge != final_edges).any(dim=0).all() and new_edge[0] != new_edge[1]:
                    final_edges = torch.cat((final_edges, new_edge), dim=1)
                    break

    g = pp.Graph.from_edge_index(final_edges, mapping=mapping)
    if undirected:
        g = g.to_undirected()
    return g


def Molloy_Reed(
        degree_sequence: torch.Tensor,
        undirected: bool = True,
        mapping: pp.IndexMap | None = None,
) -> pp.Graph:
    """
    Generate a random graph using the Molloy-Reed algorithm.

    Args:
        degree_sequence: The degree sequence, a list of the degrees of each node.
        undirected: If True, the graph will be undirected.
        mapping: A mapping from the node indices to node names.

    Returns:
        torch.Tensor: The adjacency matrix of the generated graph.
        dict: A mapping from node indices to degrees.

    Raises:
        ValueError: If the degree sequence is not graphical (i.e., the sum of the degrees is not even).

    Example:
        degree_sequence = torch.tensor([3, 2, 2, 1, 1, 1])
        g = molloy_reed_graph(degree_sequence)
    """

    # Check if the degree sequence is graphical (i.e., the sum of the degrees is even)
    if sum(degree_sequence) % 2 != 0:
        raise ValueError('Degree sequence is not graphical')

    # Create a list of node indices
    nodes = torch.arange(len(degree_sequence))

    # Repeat each node index according to its degree to create "stubs" (half-edges)
    stubs = nodes.repeat_interleave(degree_sequence)

    # Randomly permute the stubs to create pairs
    stubs = stubs[torch.randperm(len(stubs))]

    # Pair up the stubs to form edges
    edges = torch.stack([stubs[i::2] for i in range(2)], dim=-1)

    # Create an adjacency matrix
    adjacency_matrix = torch.zeros((len(nodes), len(nodes)))

    # Fill the adjacency matrix based on the edges
    adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
    adjacency_matrix[edges[:, 1], edges[:, 0]] = 1

    # Convert the adjacency matrix to an edge list
    final_edges = adjacency_matrix.nonzero().t()

    # Create a graph from the edge list
    g = pp.Graph.from_edge_index(final_edges, mapping=mapping)

    # If the graph is undirected, convert it to an undirected graph
    if undirected:
        g = g.to_undirected()

    return g