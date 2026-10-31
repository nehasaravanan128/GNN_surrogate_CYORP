"""
train.py
--------
Trains CYORPNet on the train-pool split (random 80/20 train/val), holding
the generalization split (d/D in {0.12, 0.22}, never trained on) fully
separate, per Project Guide Sec. 8.2 / 10.3 / 10.4.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from model.gnn_architecture import CYORPNet

torch.manual_seed(0)
np.random.seed(0)

H, L = 48, 3
BATCH = 32
EPOCHS = 45
LR = 1e-3
WEIGHT_DECAY = 1e-5


def load_splits():
    base = os.path.join(os.path.dirname(__file__), '..', 'graphs')
    d = torch.load(os.path.join(base, 'all_graphs.pt'), weights_only=False)
    x, y, edge_attr, edge_index = d['x'], d['y'], d['edge_attr'], d['edge_index']
    split = np.array(d['split'])

    train_pool_idx = np.where(split == 'train_pool')[0]
    gen_idx = np.where(split == 'generalization')[0]

    rng = np.random.RandomState(0)
    rng.shuffle(train_pool_idx)
    n_val = int(0.2 * len(train_pool_idx))
    val_idx = train_pool_idx[:n_val]
    train_idx = train_pool_idx[n_val:]

    return x, y, edge_attr, edge_index, train_idx, val_idx, gen_idx


def batches(idx, batch_size, shuffle=True):
    idx = idx.copy()
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i:i + batch_size]


def run_epoch(model, x, y, edge_attr, edge_index, idx, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    for b in batches(idx, BATCH, shuffle=training):
        xb = x[b]; eab = edge_attr[b]; yb = y[b]
        if training:
            opt.zero_grad()
        pred = model(xb, edge_index, eab)
        loss = F.mse_loss(pred, yb)
        if training:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    x, y, edge_attr, edge_index, train_idx, val_idx, gen_idx = load_splits()
    print(f"train={len(train_idx)} val={len(val_idx)} gen={len(gen_idx)}")

    # Target normalization (helps training stability, guide Sec 10.4 note)
    y_mean, y_std = y[train_idx].mean().item(), y[train_idx].std().item()
    y_norm = (y - y_mean) / y_std

    model = CYORPNet(node_feat_dim=8, edge_feat_dim=5, H=H, L=L)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, 'splits.json'), 'w') as f:
        json.dump(dict(train_idx=train_idx.tolist(), val_idx=val_idx.tolist(),
                        gen_idx=gen_idx.tolist()), f)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, x, y_norm, edge_attr, edge_index, train_idx, opt)
        sched.step()
        val_loss = run_epoch(model, x, y_norm, edge_attr, edge_index, val_idx, None)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| {time.time()-t0:.0f}s elapsed")
            # Checkpoint every log point so a killed process doesn't lose progress
            torch.save(dict(state_dict=model.state_dict(), y_mean=y_mean, y_std=y_std,
                             H=H, L=L, epoch=epoch),
                       os.path.join(os.path.dirname(__file__), 'cyorp_gnn.pt'))
            with open(os.path.join(os.path.dirname(__file__), 'training_history.json'), 'w') as f:
                json.dump(history, f)

    out_dir = os.path.dirname(__file__)
    torch.save(dict(state_dict=model.state_dict(), y_mean=y_mean, y_std=y_std,
                     H=H, L=L), os.path.join(out_dir, 'cyorp_gnn.pt'))
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    with open(os.path.join(out_dir, 'splits.json'), 'w') as f:
        json.dump(dict(train_idx=train_idx.tolist(), val_idx=val_idx.tolist(),
                        gen_idx=gen_idx.tolist()), f)

    print(f"Done. Total training time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
