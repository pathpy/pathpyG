from __future__ import annotations

from torch import IntTensor, equal, tensor

from pathpyG import config
from pathpyG.nn.dbgnn import (
    DBGNN
)

def test_bipartite_edge_index(simple_paths):
    g = HigherOrderGraph(simple_paths, order=1)
    print(g.data.edge_index)
    print(g.mapping)
    g2 = HigherOrderGraph(simple_paths, order=2)
    print(g2.data.edge_index)
    print(g2.mapping)

    bipartite_edge_index = DBGNN.generate_bipartite_edge_index(g, g2, mapping='last')
    print(bipartite_edge_index)
    # ensure that A,C and B,C are mapped to C, C,D is mapped to D and C,E is mapped to E

    assert equal(bipartite_edge_index, tensor([[0, 1, 2, 3], [2, 2, 3, 4]]))

    bipartite_edge_index = DBGNN.generate_bipartite_edge_index(g, g2, mapping='first')
    print(bipartite_edge_index)
    # ensure that A,C is mapped A, B,C is mapped to B, and C,D and C,E are mapped to C

    assert equal(bipartite_edge_index, tensor([[0, 1, 2, 3], [0, 1, 2, 2]]))