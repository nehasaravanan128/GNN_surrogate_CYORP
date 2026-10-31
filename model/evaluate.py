"""
evaluate.py
-----------
Computes accuracy metrics (guide Sec. 11.1) and speed benchmark
(Sec. 11.2) on the val (test) and generalization splits, and produces
the required plots (Sec. 11.3).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from model.gnn_architecture import CYORPNet

BASE = os.path.dirname(__file__)
RESULTS = os.path.join(BASE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)


def load_everything():
    d = torch.load(os.path.join(BASE, '..', 'graphs', 'all_graphs.pt'), weights_only=False)
    ck = torch.load(os.path.join(BASE, 'cyorp_gnn.pt'), weights_only=False)
    splits = json.load(open(os.path.join(BASE, 'splits.json')))
    history = json.load(open(os.path.join(BASE, 'training_history.json')))
    dataset = json.load(open(os.path.join(BASE, '..', 'dataset', 'dataset_full.json')))
    return d, ck, splits, history, dataset


def predict(model, x, edge_index, edge_attr, y_mean, y_std, idx, batch=64):
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            out = model(x[b], edge_index, edge_attr[b])
            preds.append(out.numpy() * y_std + y_mean)
    return np.concatenate(preds)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100)
    return dict(MAE=float(mae), RMSE=rmse, R2=float(r2), MAPE_pct=mape)


def main():
    d, ck, splits, history, dataset = load_everything()
    x, y, edge_attr, edge_index = d['x'], d['y'], d['edge_attr'], d['edge_index']
    split_arr = np.array(d['split'])
    val_idx = np.array(splits['val_idx'])
    gen_idx = np.array(splits['gen_idx'])
    train_idx = np.array(splits['train_idx'])

    model = CYORPNet(node_feat_dim=8, edge_feat_dim=5, H=ck['H'], L=ck['L'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    y_mean, y_std = ck['y_mean'], ck['y_std']

    val_pred = predict(model, x, edge_index, edge_attr, y_mean, y_std, val_idx)
    gen_pred = predict(model, x, edge_index, edge_attr, y_mean, y_std, gen_idx)
    val_true = y[val_idx].numpy()
    gen_true = y[gen_idx].numpy()

    val_metrics = metrics(val_true, val_pred)
    gen_metrics = metrics(gen_true, gen_pred)
    print("Test (val) set:  ", val_metrics)
    print("Generalization:  ", gen_metrics)

    # ---- Speed benchmark (guide Sec. 11.2) ----
    solver_times = [r['solver_time_sec'] for r in dataset]
    mean_solver_time = float(np.mean(solver_times))

    sample_idx = val_idx[:64] if len(val_idx) >= 64 else val_idx
    xb, eab = x[sample_idx], edge_attr[sample_idx]
    with torch.no_grad():
        _ = model(xb, edge_index, eab)  # warm up
    t0 = time.time()
    N_TRIALS = 50
    with torch.no_grad():
        for _ in range(N_TRIALS):
            _ = model(xb, edge_index, eab)
    gnn_time_per_sample = (time.time() - t0) / N_TRIALS / len(sample_idx)
    speedup = mean_solver_time / gnn_time_per_sample
    print(f"Solver mean time: {mean_solver_time*1000:.2f} ms/sample")
    print(f"GNN mean time:    {gnn_time_per_sample*1000:.4f} ms/sample")
    print(f"Speedup:          {speedup:.0f}x")

    summary = dict(val_metrics=val_metrics, gen_metrics=gen_metrics,
                    solver_ms=mean_solver_time * 1000,
                    gnn_ms=gnn_time_per_sample * 1000,
                    speedup=speedup,
                    n_train=len(train_idx), n_val=len(val_idx), n_gen=len(gen_idx))
    with open(os.path.join(RESULTS, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Plot 1: Parity plot (test set) ----
    plt.figure(figsize=(6, 6))
    plt.scatter(val_true, val_pred, alpha=0.4, s=10, color='#2b6cb0')
    lims = [min(val_true.min(), val_pred.min()), max(val_true.max(), val_pred.max())]
    plt.plot(lims, lims, 'r--', label='Perfect prediction')
    plt.xlabel('Solver Torque (N·m)')
    plt.ylabel('GNN Predicted Torque (N·m)')
    plt.title(f'GNN vs. Solver -- Test Set (R²={val_metrics["R2"]:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot1_parity_test.png'), dpi=150)
    plt.close()

    # ---- Plot 3: Generalization parity plot ----
    plt.figure(figsize=(6, 6))
    plt.scatter(gen_true, gen_pred, alpha=0.4, s=10, color='#c05621')
    lims = [min(gen_true.min(), gen_pred.min()), max(gen_true.max(), gen_pred.max())]
    plt.plot(lims, lims, 'r--', label='Perfect prediction')
    plt.xlabel('Solver Torque (N·m)')
    plt.ylabel('GNN Predicted Torque (N·m)')
    plt.title(f'GNN vs. Solver -- Held-out d/D (R²={gen_metrics["R2"]:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot3_parity_generalization.png'), dpi=150)
    plt.close()

    # ---- Plot 4: Training curve ----
    epochs = [h['epoch'] for h in history]
    tr = [h['train_loss'] for h in history]
    va = [h['val_loss'] for h in history]
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, tr, label='Train loss')
    plt.plot(epochs, va, label='Val loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized MSE loss (log scale)')
    plt.title('Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot4_training_curve.png'), dpi=150)
    plt.close()

    # ---- Plot 2: Error vs each parameter (on val set) ----
    val_records = [dataset[i] for i in val_idx]
    params_arr = {k: np.array([r['params'][k] for r in val_records])
                  for k in ['d_over_D', 'k', 'P_hours', 'latitude_deg']}
    abs_err = np.abs(val_true - val_pred)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (pname, pvals) in zip(axes.flat, params_arr.items()):
        ax.scatter(pvals, abs_err, alpha=0.4, s=10, color='#2f855a')
        ax.set_xlabel(pname)
        ax.set_ylabel('|error| (N·m)')
        ax.set_title(f'Error vs {pname}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot2_error_vs_params.png'), dpi=150)
    plt.close()

    # ---- Plot 6: Torque vs d/D (solver ground truth, from full dataset) ----
    train_records = [r for r in dataset if r['split'] == 'train_pool']
    plt.figure(figsize=(7, 5))
    for k_val in sorted(set(r['params']['k'] for r in train_records)):
        subset = [r for r in train_records
                  if r['params']['k'] == k_val and r['params']['P_hours'] == 24
                  and r['params']['latitude_deg'] == 0]
        subset.sort(key=lambda r: r['params']['d_over_D'])
        if subset:
            plt.plot([r['params']['d_over_D'] for r in subset],
                     [r['torque'] for r in subset], marker='o', label=f'k={k_val}')
    plt.xlabel('d/D')
    plt.ylabel('CYORP Torque (N·m)')
    plt.title('Solver: Torque vs. Crater Depth (P=24h, lat=0°)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot6_torque_vs_depth.png'), dpi=150)
    plt.close()

    # ---- Plot 5: Solver temperature field (representative sample) ----
    from solver.heat_solver import run_thermal_solver
    params = dict(d_over_D=0.25, k=0.1, P_hours=24, latitude_deg=30)
    _, T = run_thermal_solver(params, Nr=25, Nz=35, n_periods=5, return_field=True)
    plt.figure(figsize=(7, 5))
    im = plt.imshow(T.T, aspect='auto', origin='upper', cmap='inferno',
                     extent=[0, 75, -T.shape[1], 0])
    plt.colorbar(im, label='Temperature (K)')
    plt.xlabel('Radial distance r (m)')
    plt.ylabel('Depth index (surface at top)')
    plt.title('Solver Temperature Field (d/D=0.25, k=0.1, P=24h, lat=30°)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot5_temperature_field.png'), dpi=150)
    plt.close()

    # ---- Plot 7: Speed-accuracy tradeoff table (as a figure) ----
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    table_data = [
        ['Solver mean time/sample', f'{mean_solver_time*1000:.1f} ms'],
        ['GNN mean time/sample', f'{gnn_time_per_sample*1000:.4f} ms'],
        ['Speedup', f'{speedup:.0f}x'],
        ['Test R²', f'{val_metrics["R2"]:.3f}'],
        ['Test MAE', f'{val_metrics["MAE"]:.4f} N·m'],
        ['Generalization R²', f'{gen_metrics["R2"]:.3f}'],
        ['Generalization MAE', f'{gen_metrics["MAE"]:.4f} N·m'],
    ]
    tbl = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                    loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    plt.title('Speed / Accuracy Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, 'plot7_speed_accuracy_summary.png'), dpi=150)
    plt.close()

    print("All plots saved to results/")


if __name__ == '__main__':
    main()
