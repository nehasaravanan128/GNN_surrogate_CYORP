"""
build_graphs.py
----------------
Converts each dataset record into node/edge feature tensors, per Project
Guide Sec. 9.2-9.5 (8 node features, 5 edge features, graph-level scalar
target). Because every crater uses the same (Nr, Nz) grid, the edge
topology (edge_index) is identical across all samples -- built once and
shared, which lets us store the whole dataset as dense (N_graphs, N_nodes,
F) tensors instead of per-graph PyG Data objects (no torch_geometric
dependency needed; a plain batched message-passing GNN operates directly
on these tensors -- see model/gnn_architecture.py).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

Nr, Nz = 15, 20
D = 100.0
R = D / 2.0


def build_edge_index():
    """4-neighbor grid connectivity (up/down/left/right), both directions,
    per guide Sec. 9.5."""
    src, dst, is_radial, is_depth = [], [], [], []
    for i in range(Nr):
        for j in range(Nz):
            idx = i * Nz + j
            if i + 1 < Nr:
                nb = (i + 1) * Nz + j
                src += [idx, nb]; dst += [nb, idx]
                is_radial += [1, 1]; is_depth += [0, 0]
            if j + 1 < Nz:
                nb = i * Nz + (j + 1)
                src += [idx, nb]; dst += [nb, idx]
                is_radial += [0, 0]; is_depth += [1, 1]
    return (np.array(src), np.array(dst),
            np.array(is_radial, dtype=np.float32), np.array(is_depth, dtype=np.float32))


def build_edge_attr(r, z, src, dst, is_radial, is_depth):
    """5 edge features: delta_r, delta_z, distance, is_radial, is_depth."""
    ri = np.repeat(r, Nz)   # node i's r-coord for flat index i*Nz+j -> r[i]
    zj = np.tile(z, Nr)     # node i's z-coord -> z[j]
    dr = ri[dst] - ri[src]
    dz = zj[dst] - zj[src]
    dist = np.sqrt(dr**2 + dz**2)
    feats = np.stack([dr, dz, dist, is_radial, is_depth], axis=1).astype(np.float32)
    return feats


def build_node_features(params, r, z, norm_stats):
    """8 node features per guide Sec. 9.2."""
    r_max = r.max()
    z_max = np.abs(z).max()
    feats = np.zeros((Nr * Nz, 8), dtype=np.float32)
    k_n = (params['k'] - norm_stats['k_min']) / norm_stats['k_range']
    P_n = (params['P_hours'] - norm_stats['P_min']) / norm_stats['P_range']
    lat_n = params['latitude_deg'] / 90.0
    dd_n = (params['d_over_D'] - norm_stats['dd_min']) / norm_stats['dd_range']

    idx = 0
    for i in range(Nr):
        for j in range(Nz):
            is_surface = 1.0 if j == 0 else 0.0
            is_wall = 1.0 if (r[i] <= R and j == 0) else 0.0
            feats[idx] = [r[i] / r_max, abs(z[j]) / z_max, is_surface, is_wall,
                          k_n, P_n, lat_n, dd_n]
            idx += 1
    return feats


def compute_norm_stats(dataset):
    ks = [r['params']['k'] for r in dataset]
    Ps = [r['params']['P_hours'] for r in dataset]
    dds = [r['params']['d_over_D'] for r in dataset]
    return dict(
        k_min=min(ks), k_range=max(ks) - min(ks),
        P_min=min(Ps), P_range=max(Ps) - min(Ps),
        dd_min=min(dds), dd_range=max(dds) - min(dds),
    )


def main():
    base = os.path.dirname(__file__)
    dataset = json.load(open(os.path.join(base, '..', 'dataset', 'dataset_full.json')))

    norm_stats = compute_norm_stats(dataset)
    src, dst, is_radial, is_depth = build_edge_index()

    all_x, all_y, all_split, all_dd = [], [], [], []
    for rec in dataset:
        dd = rec['params']['d_over_D']
        d = dd * D
        r = np.linspace(0.0, 1.5 * R, Nr)
        z = np.linspace(0.0, -(d + 20.0), Nz)
        x = build_node_features(rec['params'], r, z, norm_stats)
        all_x.append(x)
        all_y.append(rec['torque'])
        all_split.append(rec['split'])
        all_dd.append(dd)

    x_tensor = torch.tensor(np.stack(all_x), dtype=torch.float32)       # (Ngraphs, N, 8)
    y_tensor = torch.tensor(all_y, dtype=torch.float32)                  # (Ngraphs,)
    # edge_attr differs only via z (depth range), which differs per-sample.
    # Build it per-sample too (cheap: same op as node features).
    all_edge_attr = []
    for dd in all_dd:
        d = dd * D
        r = np.linspace(0.0, 1.5 * R, Nr)
        z = np.linspace(0.0, -(d + 20.0), Nz)
        all_edge_attr.append(build_edge_attr(r, z, src, dst, is_radial, is_depth))
    edge_attr_tensor = torch.tensor(np.stack(all_edge_attr), dtype=torch.float32)  # (Ngraphs, E, 5)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)   # (2, E) shared

    torch.save(dict(x=x_tensor, y=y_tensor, edge_attr=edge_attr_tensor,
                     edge_index=edge_index, split=all_split, norm_stats=norm_stats),
               os.path.join(base, 'all_graphs.pt'))

    with open(os.path.join(base, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Saved {len(dataset)} graphs. Nodes/graph={Nr*Nz}, Edges/graph={src.shape[0]}")
    print(f"x shape {tuple(x_tensor.shape)}, edge_attr shape {tuple(edge_attr_tensor.shape)}")


if __name__ == '__main__':
    main()
