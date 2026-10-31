"""
gnn_architecture.py
--------------------
CYORPNet: encode-process-decode GNN (Pfaff et al. 2020 style, per guide
Sec. 10.1-10.2), implemented directly on batched dense tensors since
every crater graph shares the same topology (see graphs/build_graphs.py
docstring) -- this avoids a torch_geometric dependency while keeping the
exact same computational structure: per-edge MLP -> sum-aggregate to
nodes -> per-node MLP -> residual -> repeat L times -> mean-pool -> MLP
to scalar torque.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphNetLayer(nn.Module):
    """One GN block: edge update (sum aggregate) -> node update -> residual."""
    def __init__(self, H):
        super().__init__()
        self.edge_mlp = MLP(2 * H + H, 2 * H, H)
        self.node_mlp = MLP(H + H, 2 * H, H)

    def forward(self, x, edge_index, edge_attr):
        # x: (B, N, H), edge_attr: (B, E, H), edge_index: (2, E) shared
        src, dst = edge_index[0], edge_index[1]
        x_src = x[:, src, :]                       # (B, E, H)
        x_dst = x[:, dst, :]                       # (B, E, H)
        edge_in = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        new_edge_attr = self.edge_mlp(edge_in)      # (B, E, H)

        # Sum-aggregate messages into destination nodes
        B, N, H = x.shape
        agg = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, H)
        agg.scatter_add_(1, idx, new_edge_attr)

        node_in = torch.cat([x, agg], dim=-1)
        new_x = self.node_mlp(node_in)
        return new_x, new_edge_attr


class CYORPNet(nn.Module):
    def __init__(self, node_feat_dim=8, edge_feat_dim=5, H=64, L=4):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_feat_dim, H), nn.ReLU(), nn.Linear(H, H))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_feat_dim, H), nn.ReLU(), nn.Linear(H, H))
        self.layers = nn.ModuleList([GraphNetLayer(H) for _ in range(L)])
        self.decoder = nn.Sequential(
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H // 2), nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_new, edge_attr = layer(x, edge_index, edge_attr)
            x = x + x_new  # residual

        x_global = x.mean(dim=1)          # graph-level mean pool, (B, H)
        return self.decoder(x_global).squeeze(-1)  # (B,)
