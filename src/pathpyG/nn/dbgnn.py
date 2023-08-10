
from __future__ import annotations

import torch
from torch.nn import Linear, ModuleList, Module
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv

class BipartiteGraphOperator(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super(BipartiteGraphOperator, self).__init__('add')
        self.lin1 = Linear(in_ch, out_ch)
        self.lin2 = Linear(in_ch, out_ch)

    def forward(self, x, bipartite_index, N, M):
        x = (self.lin1(x[0]), self.lin2(x[1]))
        return self.propagate(bipartite_index, size=(N, M), x=x)

class DBGNN(Module):
    """Implementation of time-aware graph neural network DBGNN
    Reference paper: https://openreview.net/pdf?id=Dbkqs1EhTr

    Args:
        num_classes: int - number of classes
        num_features: list - number of features for first order and higher order nodes, e.g. [first_order_num_features, second_order_num_features]
        hidden_dims: list - number of hidden dimensions per each layer in the first/higher order network
        p_dropout: float - drop-out probability
    """
    def __init__(
        self,
        num_classes,
        num_features,
        hidden_dims,
        p_dropout=0.0
        ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.p_dropout = p_dropout

        # higher-order layers
        self.higher_order_layers = ModuleList()
        self.higher_order_layers.append(GCNConv(self.num_features[1], self.hidden_dims[0]))

        # first-order layers
        self.first_order_layers = ModuleList()
        self.first_order_layers.append(GCNConv(self.num_features[1], self.hidden_dims[0]))

        for dim in range(self.hidden_dims[1:-1]):
            # higher-order layers
            self.higher_order_layers.append(GCNConv(self.hidden_dims[dim-1], self.hidden_dims[dim]))
            # first-order layers
            self.first_order_layers.append(GCNConv(self.hidden_dims[dim-1], self.hidden_dims[dim]))

        self.bipartite_layer = BipartiteGraphOperator(self.hidden_dims[-2], self.hidden_dims[-1])

        # Linear layer
        self.lin = torch.nn.Linear(self.hidden_dims[-1], num_classes)



    def forward(self, data):

        x = data.x
        x_h = data.x_h

        # First-order convolutions
        for layer in self.first_order_layers:
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            x = F.elu(layer(x, data.edge_index, data.edge_weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        # Second-order convolutions
        for layer in self.higher_order_layers:
            x_h = F.dropout(x_h, p=self.p_dropout, training=self.training)
            x_h = F.elu(layer(x_h, data.edge_index_higher_order, data.edge_weights_higher_order))
        x_h = F.dropout(x_h, p=self.p_dropout, training=self.training)

        # Bipartite message passing
        x = torch.nn.functional.elu(self.bipartite_layer((x_h, x), data.bipartite_edge_index, N = data.num_ho_nodes, M= data.num_nodes))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        # Linear layer
        x = self.lin(x)

        return x