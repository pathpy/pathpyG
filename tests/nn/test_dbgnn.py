from __future__ import annotations

import torch
from torch import equal, tensor

from pathpyG import config
from pathpyG.nn.dbgnn import DBGNN
from pathpyG.core.multi_order_model import MultiOrderModel
from pathpyG.utils.dbgnn import generate_bipartite_edge_index


def test_bipartite_edge_index(simple_walks):
    m = MultiOrderModel.from_PathData(simple_walks, max_order=2)
    g = m.layers[1]
    print(g.data.edge_index)
    print(g.mapping)
    g2 = m.layers[2]
    print(g2.data.edge_index)
    print(g2.mapping)

    bipartite_edge_index = generate_bipartite_edge_index(g, g2, mapping="last")
    print(bipartite_edge_index)
    # ensure that A,C and B,C are mapped to C, C,D is mapped to D and C,E is mapped to E

    assert equal(bipartite_edge_index, tensor([[0, 1, 2, 3], [2, 2, 3, 4]]))

    bipartite_edge_index = generate_bipartite_edge_index(g, g2, mapping="first")
    print(bipartite_edge_index)
    # ensure that A,C is mapped A, B,C is mapped to B, and C,D and C,E are mapped to C

    assert equal(bipartite_edge_index, tensor([[0, 1, 2, 3], [0, 1, 2, 2]]))


def test_dbgnn(simple_walks):
    m = MultiOrderModel.from_PathData(simple_walks, max_order=2)
    data = m.to_dbgnn_data()
    g1 = m.layers[1]
    g2 = m.layers[2]
    data.y = torch.tensor([g1.mapping.to_idx(i) // 10 for i in g1.mapping.node_ids])

    model = DBGNN(num_features=[g1.N, g2.N], num_classes=len(data.y.unique()), hidden_dims=[16, 32, 8], p_dropout=0.4)

    out = model(data)
    assert out is not None
