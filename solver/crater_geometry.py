"""
crater_geometry.py
-------------------
Builds the 2D axisymmetric (r, z) cylindrical grid for a parabolic-bowl
crater, per Project Guide Sec. 6 and 7.2.

Grid convention:
    r in [0, 1.5*R]   (crater radius R = D/2, plus flat apron around it)
    z in [0, -(d+20)] (0 = rim level, negative = depth below rim)
    T[i, j] lives on this *rectangular* (r, z) grid -- the crater bowl
    shape only enters through (a) which surface cells are "inside" the
    crater (r <= R) for the torque integral, and (b) the analytic bowl
    slope dz/dr used for the outward-normal in the torque calculation.
    The solver itself does not warp the mesh to follow the bowl -- this
    is the same simplification specified in the project guide (Sec. 7.2-7.5).
"""
import numpy as np


def crater_z(r, R, d):
    """Parabolic bowl profile: z(r) = -d*(1-(r/R)^2) for r<=R, else 0."""
    r = np.asarray(r, dtype=float)
    z = np.where(r <= R, -d * (1.0 - (r / R) ** 2), 0.0)
    return z


def build_crater_grid(d_over_D, D=100.0, Nr=25, Nz=35):
    """
    Build the 2D cylindrical grid for a crater.

    Returns
    -------
    r : (Nr,) radial coordinates [m]
    z : (Nz,) depth coordinates [m], 0 at rim, negative downward
    R : crater radius [m]
    d : crater depth [m]
    in_crater : (Nr,) boolean, True where r <= R (used to restrict the
                torque integral to the crater interior)
    """
    R = D / 2.0
    d = d_over_D * D

    r = np.linspace(0.0, 1.5 * R, Nr)
    z = np.linspace(0.0, -(d + 20.0), Nz)
    in_crater = r <= R

    return r, z, R, d, in_crater


def surface_normal(r, R, d):
    """
    Outward normal (n_r, n_z) of the parabolic bowl at radius r,
    from dz/dr = 2*d*r/R^2 (derivative of z(r) = -d*(1-(r/R)^2)).
    Points outward + upward, matching Project Guide Sec. 7.5.
    """
    dz_dr = 2.0 * d * r / R**2
    norm_mag = np.sqrt(dz_dr**2 + 1.0)
    n_r = dz_dr / norm_mag
    n_z = 1.0 / norm_mag
    return n_r, n_z, dz_dr
