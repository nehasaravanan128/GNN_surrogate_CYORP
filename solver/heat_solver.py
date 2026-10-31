"""
heat_solver.py
--------------
Vectorized explicit finite-difference time-stepper for the 2D cylindrical
heat equation, per Project Guide Sec. 5.1-5.3, 7.3-7.4.

    dT/dt = alpha * [ d2T/dr2 + (1/r) dT/dr + d2T/dz2 ]

Timestep choice (deviation from guide, documented):
The guide's CFL formula dt < dr^2/(4*alpha) is necessary but not
sufficient here: for low-conductivity regolith (k as low as 1e-3 W/m/K),
alpha is so small that the CFL-allowed dt is *much larger* than the
rotation period P itself, which would mean the solver never resolves
the diurnal solar cycle at all (the surface BC would only be evaluated
once or twice per rotation). We therefore additionally cap
    dt <= P / STEPS_PER_PERIOD
so every run resolves the solar cycle with at least STEPS_PER_PERIOD
steps, and take dt = min(dt_cfl, dt_diurnal). This cap is what actually
binds for nearly all of the parameter grid (see accompanying report).
"""
import numpy as np

RHO = 1500.0     # kg/m^3, regolith density (fixed, guide cheat sheet)
CP = 600.0       # J/kg/K, specific heat (fixed, guide cheat sheet)
STEPS_PER_PERIOD = 120  # temporal resolution of the diurnal cycle


def thermal_diffusivity(k):
    return k / (RHO * CP)


def choose_dt(dr, dz, alpha, P, safety=0.4):
    dt_cfl = safety * min(dr, dz) ** 2 / (2.0 * alpha)
    dt_diurnal = P / STEPS_PER_PERIOD
    return min(dt_cfl, dt_diurnal)


def laplacian_step(T, alpha, dt, dr, dz, r):
    """
    One explicit-Euler update of the interior nodes using the vectorized
    central-difference cylindrical Laplacian (guide Sec. 7.3,
    'finite_difference_step_fast').
    """
    T_new = T.copy()
    r_col = r[1:-1, np.newaxis]  # (Nr-2, 1) for broadcasting over z

    d2T_dr2 = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dr**2
    dT_dr = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dr)
    d2T_dz2 = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dz**2

    laplacian = d2T_dr2 + (1.0 / r_col) * dT_dr + d2T_dz2
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * alpha * laplacian

    # Axis of symmetry (r=0): Neumann dT/dr=0, ghost-point mirror (guide Sec. 5.3/D.3)
    T_new[0, 1:-1] = T_new[1, 1:-1]

    return T_new


def apply_surface_bc(T, t, r, lat_rad, P, k, dz, in_crater, newton_iters=6):
    """
    Radiative surface energy balance (guide Sec. 5.3/7.4):
        -k * dT/dz = (1-A)*Q_sun(t) - eps*sigma*T^4

    Deviation from guide (documented): the guide's reference code solves
    this "lagged" (T[i,0] = T[i,1] + net_flux*dz/k using the *previous*
    T[i,0] to evaluate flux_out). For our grid, dz/k is large enough at
    low conductivity that this single-step update overshoots wildly and
    the radiative term (~T^4) blows up to NaN within a few steps -- an
    instance of exactly the "NaN/Inf failure mode" the guide's own
    mistakes table warns about, but caused by the surface-BC nonlinearity
    rather than the interior CFL condition.

    Fix: solve the nonlinear balance implicitly for T_surf with a few
    steps of vectorized Newton's method each timestep (unconditionally
    stable for this 0-D nonlinear equation, and converges in <6 steps
    since sigma*T^4 is smooth). This does not change the physics (same
    equation), only how it is solved.
    """
    from .solar_flux import S0, A_ALBEDO, EPS, SIGMA, solar_elevation

    elev = solar_elevation(t, lat_rad, P)
    flux_in = S0 * max(0.0, np.sin(elev)) * (1.0 - A_ALBEDO)

    T_below = T[:, 1]
    kdz = k / dz
    Ts = T[:, 0].copy()  # initial guess: previous surface temp

    for _ in range(newton_iters):
        f = EPS * SIGMA * Ts**4 + kdz * Ts - kdz * T_below - flux_in
        fp = 4.0 * EPS * SIGMA * Ts**3 + kdz
        Ts = Ts - f / fp
        Ts = np.clip(Ts, 50.0, 2000.0)  # physical bounds guard

    T[:, 0] = Ts
    return T


def apply_bottom_bc(T):
    """Zero-flux deep boundary (guide Sec. 5.3): T[:,-1] = T[:,-2]."""
    T[:, -1] = T[:, -2]
    return T


def run_thermal_solver(params, Nr=25, Nz=35, n_periods=3, return_field=False):
    """
    Run the FD thermal solver for one crater parameter set.

    params: dict with d_over_D, k, P_hours, latitude_deg
    Returns: mean CYORP torque (N.m) [, final temperature field if return_field]
    """
    from .crater_geometry import build_crater_grid
    from .torque_calculator import compute_torque

    d_over_D = params['d_over_D']
    k = params['k']
    P = params['P_hours'] * 3600.0
    lat_rad = np.radians(params['latitude_deg'])

    alpha = thermal_diffusivity(k)
    r, z, R, d, in_crater = build_crater_grid(d_over_D, Nr=Nr, Nz=Nz)
    dr = r[1] - r[0]
    dz = abs(z[1] - z[0])
    dt = choose_dt(dr, dz, alpha, P)

    T = np.full((Nr, Nz), 250.0)
    t = 0.0
    steps_per_period = max(1, int(round(P / dt)))

    torques = []
    for step in range(n_periods * steps_per_period):
        T = apply_surface_bc(T, t, r, lat_rad, P, k, dz, in_crater)
        T = laplacian_step(T, alpha, dt, dr, dz, r)
        T = apply_bottom_bc(T)
        t += dt

        if step >= (n_periods - 1) * steps_per_period:
            tau = compute_torque(T, r, z, d_over_D, 100.0)
            torques.append(tau)

    mean_torque = float(np.mean(torques))
    if return_field:
        return mean_torque, T
    return mean_torque
