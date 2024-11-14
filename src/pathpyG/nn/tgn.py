import torch
from torch.nn import Linear
from torch_geometric.nn import TransformerConv, TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)

from sklearn.metrics import balanced_accuracy_score


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class NodePredictor(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, dropout):
        super().__init__()
        self.dropout = dropout 
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(in_channels, in_channels))
        for i in range(1, self.num_layers):
            self.layers.append(Linear(in_channels, in_channels))
        self.layers.append(Linear(in_channels, num_classes))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = torch.relu(x)
                x = torch.nn.Dropout(p=self.dropout)(x)
        return x
    

def build_gnn(data, hidden_dims, learning_rate, device):
    memory_dim = time_dim = embedding_dim = hidden_dims

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    node_pred = NodePredictor(in_channels=embedding_dim, num_classes=len(data.y.unique()), num_layers=2, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(node_pred.parameters()), lr=learning_rate)
    
    return memory, gnn, node_pred, optimizer

def filter_indices(indices, filter):
    uniques, counts = torch.cat((indices, filter)).unique(return_counts=True)
    return uniques[counts > 1]


def train_epoch_tgn(memory, gnn, node_pred, optimizer, neighbor_loader, data_loader, data, classes, assoc, device):
    
    train_split = data.train_mask
    test_split = data.test_mask

    memory.train()
    gnn.train()
    node_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    preds = torch.zeros((data.y.size(0), classes), device=data.y.device).float()

    total_loss = 0
    for batch in data_loader: 
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))


        train_nid = filter_indices(batch.n_id, train_split)
        pred_train = node_pred(z[assoc[train_nid]]).squeeze()
        pred_all = node_pred(z[assoc[batch.n_id]]).squeeze().detach()

        preds[batch.n_id] = pred_all[torch.arange(batch.n_id.size(0), device=batch.n_id.device)]

        loss = torch.nn.functional.cross_entropy(pred_train, data.y[train_nid]) 


        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    train_loss = float(torch.nn.functional.cross_entropy(preds[train_split], data.y[train_split]))

    # reset before new run
    memory.eval()
    gnn.eval()
    node_pred.eval()

    metric_train = balanced_accuracy_score(data.y[train_split].cpu().numpy(), preds[train_split].argmax(dim=-1).cpu().numpy())
    metric_test = balanced_accuracy_score(data.y[test_split].cpu().numpy(), preds[test_split].argmax(dim=-1).cpu().numpy())


    return total_loss/ data.num_events, metric_train, metric_test
