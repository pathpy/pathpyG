"""Utils for DBGNN models."""

from typing import Optional

import torch

from pathpyG.core.graph import Graph


def generate_bipartite_edge_index(
    g: Graph, g2: Graph, mapping: str = "last", device: Optional[torch.device] = None
) -> torch.Tensor:
    """Generate edge_index for bipartite graph connecting nodes of a second-order graph to first-order nodes.
    
    The mapping strategy determines to which first-order nodes the second-order nodes are connected:
    - "last": Connects each second-order node to the last node in its sequence.
    - "first": Connects each second-order node to the first node in its sequence.
    - "both": Connects each second-order node to both the first and last nodes in its sequence.

    !!! warning "Only for Second-Order Graphs"
        This function is intended to be used with second-order graphs only. 
        It does not support the use of higher-order graphs, such as third-order graphs or beyond.

    Args:
        g (Graph): The first-order graph.
        g2 (Graph): The second-order graph.
        mapping (str, optional): The mapping strategy to use. Options are "last", "first", or "both". Defaults to "last".
        device (torch.device, optional): The device to place the tensor on. Defaults to None.

    Returns:
        torch.Tensor: The edge_index tensor for the bipartite graph.
    """
    if mapping == "last":
        bipartide_edge_index = torch.tensor([list(range(g2.n)), [v[1] for v in g2.data.node_sequence]], device=device)
    elif mapping == "first":
        bipartide_edge_index = torch.tensor([list(range(g2.n)), [v[0] for v in g2.data.node_sequence]], device=device)
    else:
        bipartide_edge_index = torch.tensor(
            [
                list(range(g2.n)) + list(range(g2.n)),
                [v[0] for v in g2.data.node_sequence] + [v[1] for v in g2.data.node_sequence],
            ],
            device=device,
        )

    return bipartide_edge_index
