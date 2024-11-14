
from __future__ import annotations

import torch

from pathpyG.core.graph import Graph

def generate_bipartite_edge_index(g: Graph, g2: Graph, mapping: str = 'last') -> torch.Tensor:
    """Generate edge_index for bipartite graph connecting nodes of a second-order graph to first-order nodes."""

    if mapping == 'last':
        bipartide_edge_index = torch.tensor(
            [list(range(g2.n)), [v[1] for v in g2.data.node_sequence]]
            )

    elif mapping == 'first':
        bipartide_edge_index = torch.tensor(
            [list(range(g2.n)), [v[0] for v in g2.data.node_sequence]]
        )
    else:
        bipartide_edge_index = torch.tensor(
            [list(range(g2.n)) + list(range(g2.n)),
            [v[0] for v in g2.data.node_sequence] + [v[1] for v in g2.data.node_sequence]]
        )

    return bipartide_edge_index