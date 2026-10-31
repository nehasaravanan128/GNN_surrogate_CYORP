"""
generate_dataset.py
--------------------
Runs the solver over the full parameter grid specified in the project
guide (Sec. 8.1), plus a held-out generalization set on two never-seen
d/D values (Sec. 8.2), and saves everything with a train/gen split flag
already attached to each record so downstream phases never have to
re-derive the split.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import product
from solver.heat_solver import run_thermal_solver

# ---- Parameter grid (guide Sec. 8.1) ----
d_over_D_train = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
d_over_D_gen = [0.12, 0.22]  # held out -- NEVER used in training (guide Sec. 8.2)
k_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
P_values_hours = [2, 5, 10, 24, 50, 100]
latitude_values = [0, 15, 30, 45, 60, 75, 90]

Nr, Nz, N_PERIODS = 25, 35, 5


def make_records(d_over_D_values, split_name):
    combos = list(product(d_over_D_values, k_values, P_values_hours, latitude_values))
    records = []
    for dd, k, P, lat in combos:
        params = dict(d_over_D=dd, k=k, P_hours=P, latitude_deg=lat)
        t0 = time.time()
        torque = run_thermal_solver(params, Nr=Nr, Nz=Nz, n_periods=N_PERIODS)
        elapsed = time.time() - t0
        records.append(dict(params=params, torque=torque,
                             solver_time_sec=elapsed, split=split_name))
    return records


def main():
    out_dir = os.path.join(os.path.dirname(__file__))
    t_start = time.time()

    train_records = make_records(d_over_D_train, 'train_pool')
    gen_records = make_records(d_over_D_gen, 'generalization')
    dataset = train_records + gen_records

    n_nan = sum(1 for r in dataset if not np.isfinite(r['torque']))
    print(f"Total samples: {len(dataset)} (train pool {len(train_records)}, "
          f"generalization {len(gen_records)})")
    print(f"NaN/Inf torques: {n_nan}")
    print(f"Total generation time: {time.time()-t_start:.1f}s")
    times = [r['solver_time_sec'] for r in dataset]
    print(f"Solver time per sample: mean={np.mean(times):.4f}s max={np.max(times):.4f}s")

    with open(os.path.join(out_dir, 'dataset_full.json'), 'w') as f:
        json.dump(dataset, f)
    print(f"Saved dataset_full.json ({len(dataset)} records)")


if __name__ == '__main__':
    main()
