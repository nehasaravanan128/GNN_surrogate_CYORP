"""
torque_calculator.py
---------------------
Integrates the crater's surface temperature field into a single CYORP
spin-torque number, per Project Guide Sec. 5.5 / 7.5.

    dtau = -(2/(3c)) * eps * sigma * T^4 * r * n_r * dA
    tau  = sum_i dtau_i   (over crater-surface cells only, r_i <= R)
"""
import numpy as np

C_LIGHT = 3.0e8
EPS = 0.9
SIGMA = 5.67e-8


def compute_torque(T, r, z, d_over_D, D):
    from .crater_geometry import surface_normal

    R = D / 2.0
    d = d_over_D * D
    dr = r[1] - r[0]

    mask = r <= R
    ri = r[mask]
    T_surf = T[mask, 0]

    n_r, n_z, dz_dr = surface_normal(ri, R, d)
    dA = 2 * np.pi * ri * np.sqrt(1 + dz_dr**2) * dr

    dtau = -(2.0 / (3.0 * C_LIGHT)) * EPS * SIGMA * T_surf**4 * ri * n_r * dA
    return float(np.sum(dtau))
