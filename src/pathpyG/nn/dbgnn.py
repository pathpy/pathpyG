r"""Implementation of the time-aware graph neural network DBGNN (De Bruijn Graph Neural Network) proposed by Qarkaxhija, Perri, and Scholtes (2022)[^1].

This architecture is designed for time-resolved data on dynamic graphs and captures temporal-topological patterns through causal walks.
For many temporal graphs, the static representation fails to capture important temporal patterns.
An example for two different temporal patterns that are indistinguishable in the static representation is illustrated below:

<img src="../../../../img/DBGNN_example_graph.png" alt="Illustration of an example where the static representation fails to distinguish between different temporal patterns"  style="width: 100%; max-width: 800px;"/>

Frequency and topology of edges are identical, i.e. they have the same first-order time-aggregated weighted graph (center). Due to the
arrow of time, causal walks and paths differ in the dynamic graphs: Assuming a maximum waiting time $\delta = 1$ in each node, the left example node $A$ cannot causally influence $C$ via $B$,
while such a causal path is possible in the right example. A second-order De Bruijn graph model of causal walks in the two graphs
(bottom left and right) captures this difference in the causal topology.

Building on such higher-order graph models, the DBGNN defines a GNN architecture that is able to learn patterns in the causal topology of these dynamic graphs:

<img src="../../../../img/DBGNN.png" alt="Illustration of DBGNN architecture" style="width: 100%; max-width: 800px;"/>

Illustration of DBGNN architecture with two message passing layers in first- (left, gray) and second-order De Bruijn graph (right, orange)
corresponding to the dynamic graph above (left). Red edges indicate the bipartite mapping $G_b$ of higher-order node representations to
first-order representations.

[^1]: *Qarkaxhija, L., Perri, V. & Scholtes, I. De Bruijn Goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs. in LoG vol. 198 51 (PMLR, 2022).*
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing


class BipartiteGraphOperator(MessagePassing):
    """Bipartite graph operator for message passing between higher-order and first-order nodes.

    This class implements a bipartite graph operator that performs message passing from the higher-order nodes to the first-order nodes
    in the DBGNN architecture. This corresponds to the red edges in the DBGNN illustration above.
    """

    def __init__(self, in_ch: int, out_ch: int):
        """Initialize the BipartiteGraphOperator.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
        """
        super(BipartiteGraphOperator, self).__init__("add")
        self.lin1 = Linear(in_ch, out_ch)
        self.lin2 = Linear(in_ch, out_ch)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor], bipartite_index: torch.Tensor, n_ho: int, n_fo: int
    ) -> torch.Tensor:
        """Forward pass of the BipartiteGraphOperator.

        Args:
            x: Tuple of node features for higher-order [index 0] and first-order nodes [index 1].
            bipartite_index: Edge index for the bipartite graph connecting higher-order and first-order nodes.
            n_ho: Number of higher-order nodes.
            n_fo: Number of first-order nodes.

        Returns:
            Updated node features for first-order nodes after message passing.
        """
        x = (self.lin1(x[0]), self.lin2(x[1]))
        return self.propagate(bipartite_index, size=(n_ho, n_fo), x=x)

    def message(self, x_i, x_j):
        """Message function for message passing."""
        return x_i + x_j


class DBGNN(Module):
    """Implementation of the time-aware graph neural network DBGNN for time-resolved data on dynamic graph.

    The De Bruijn Graph Neural Network (DBGNN) is a time-aware graph neural network architecture designed for dynamic graphs
    that captures temporal-topological patterns through causal walks — temporally ordered sequences that represent how nodes influence each other over time.
    The architecture uses multiple layers of higher-order De Bruijn graphs for message passing, where nodes represent walks of increasing length,
    enabling the model to learn non-Markovian patterns in the causal topology of dynamic graphs.

    ??? reference
        Proposed by Qarkaxhija, Perri, and Scholtes at the first Learning on Graphs Conference (LoG) in 2022[^1]. The paper can be found [here](https://proceedings.mlr.press/v198/qarkaxhija22a/qarkaxhija22a.pdf).

    [^1]: *Qarkaxhija, L., Perri, V. & Scholtes, I. De Bruijn Goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs. in LoG vol. 198 51 (PMLR, 2022).*
    """

    def __init__(self, num_classes: int, num_features: tuple[int, int], hidden_dims: list[int], p_dropout: float = 0.0):
        """Initialize the DBGNN model.

        Args:
            num_classes: Number of classes for the classification task
            num_features: Number of features for first order and higher order nodes, i.e. `(first_order_num_features, second_order_num_features)`
            hidden_dims: Number of hidden dimensions per layer in the both GNN parts (first-order and higher-order)
            p_dropout: Drop-out probability for the dropout layers.
        """
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
        self.first_order_layers.append(GCNConv(self.num_features[0], self.hidden_dims[0]))

        for dim in range(1, len(self.hidden_dims) - 1):
            # higher-order layers
            self.higher_order_layers.append(GCNConv(self.hidden_dims[dim - 1], self.hidden_dims[dim]))
            # first-order layers
            self.first_order_layers.append(GCNConv(self.hidden_dims[dim - 1], self.hidden_dims[dim]))

        self.bipartite_layer = BipartiteGraphOperator(self.hidden_dims[-2], self.hidden_dims[-1])

        # Linear layer
        self.lin = torch.nn.Linear(self.hidden_dims[-1], num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the DBGNN.

        Args:
            data: Input [data][torch_geometric.data.Data] object containing the graph structure and node features.
        """
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
        x = torch.nn.functional.elu(
            self.bipartite_layer((x_h, x), data.bipartite_edge_index, n_ho=data.num_ho_nodes, n_fo=data.num_nodes)
        )
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        # Linear layer
        x = self.lin(x)

        return x
