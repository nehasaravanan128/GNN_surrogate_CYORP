"""
solar_flux.py
-------------
Sun elevation angle and incoming flux, per Project Guide Sec. 5.4 / 6.3.

Simplification (documented, matches guide Sec. 7.4): the solar elevation
is computed once per timestep from (latitude, hour angle) and applied
uniformly across all surface radial cells -- there is no per-cell
ray-cast shadow check against the opposite crater wall. This is the same
shortcut the guide's own reference code takes ("simplified: no shadow
check first"). It is a known limitation, not an oversight -- see the
accompanying report for its physical consequence.
"""
import numpy as np

S0 = 1361.0      # W/m^2, solar constant at 1 AU
A_ALBEDO = 0.1
EPS = 0.9
SIGMA = 5.67e-8  # W/m^2/K^4


def solar_elevation(t, lat_rad, P):
    """
    sin(elevation) = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(omega*t)
    with delta (solar declination) = 0 (equinox), per guide Sec. 5.4/6.3.
    Returns elevation angle in radians (can be negative = below horizon).
    """
    omega = 2.0 * np.pi / P
    hour_angle = omega * t
    sin_elev = np.sin(lat_rad) * 1.0 + np.cos(lat_rad) * 1.0 * np.cos(hour_angle)
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    return np.arcsin(sin_elev)


def incident_flux(t, lat_rad, P):
    """Q_sun(t) = S0 * max(0, sin(elevation)), normal incidence, r=1 AU."""
    elev = solar_elevation(t, lat_rad, P)
    return S0 * max(0.0, np.sin(elev))
