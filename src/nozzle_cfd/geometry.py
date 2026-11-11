"""Rao parabolic bell nozzle geometry generator for 1D/2D CFD and thermal solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NozzleGeometry:
    """Discretized axisymmetric rocket thrust chamber & Rao bell nozzle geometry."""

    throat_radius_m: float
    contraction_ratio: float
    expansion_ratio: float
    bell_percent: float
    theta_n_deg: float
    theta_e_deg: float
    divergence_efficiency: float
    chamber_length_m: float
    converging_length_m: float
    diverging_length_m: float
    total_length_m: float
    throat_x_m: float
    x_m: np.ndarray
    radius_m: np.ndarray
    area_m2: np.ndarray
    area_ratio: np.ndarray
    d_area_dx_m: np.ndarray
    wall_slope_rad: np.ndarray


def _rao_bell_angles(expansion_ratio: float, bell_percent: float) -> tuple[float, float]:
    """Empirical Rao 80% bell initial (theta_n) and exit (theta_e) contour angles [deg]."""
    eps = max(2.0, min(150.0, expansion_ratio))
    log_eps = math.log10(eps)
    # 80% Rao empirical curve fit
    theta_n_80 = 21.5 + 8.5 * log_eps
    theta_e_80 = max(5.5, 14.5 - 4.8 * log_eps)

    # Shorter bells (e.g. 65%) require steeper initial & exit angles; 100% bells flatten out
    scale = 80.0 / max(55.0, min(115.0, bell_percent))
    theta_n = min(40.0, theta_n_80 * (scale**0.45))
    theta_e = min(20.0, theta_e_80 * (scale**0.55))
    return round(theta_n, 2), round(theta_e, 2)


def _eval_quadratic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    one_m = 1.0 - t
    x = (one_m**2) * p0[0] + 2.0 * one_m * t * p1[0] + (t**2) * p2[0]
    y = (one_m**2) * p0[1] + 2.0 * one_m * t * p1[1] + (t**2) * p2[1]
    return x, y


def build_rao_bell_nozzle(
    throat_radius_m: float = 0.12,
    contraction_ratio: float = 3.2,
    expansion_ratio: float = 16.0,
    bell_percent: float = 80.0,
    n_points: int = 161,
    chamber_length_ratio: float = 1.35,
) -> NozzleGeometry:
    """Generate a smooth C1-continuous converging-diverging Rao parabolic bell nozzle.

    Axial coordinate x starts at chamber head-end x = 0, passes the geometric throat at
    x = throat_x_m, and exits at x = total_length_m.
    """
    if throat_radius_m <= 0.0:
        raise ValueError("throat_radius_m must be > 0")
    if contraction_ratio <= 1.05:
        raise ValueError("contraction_ratio must be > 1.05")
    if expansion_ratio < 1.05:
        raise ValueError("expansion_ratio must be >= 1.05")
    if n_points < 41:
        raise ValueError("n_points must be >= 41 for stable CFD discretization")

    rt = float(throat_radius_m)
    rc = rt * math.sqrt(contraction_ratio)
    re = rt * math.sqrt(expansion_ratio)

    # Axial layout: cylindrical chamber + smooth cosine converging section + Rao parabolic bell
    chamber_len = chamber_length_ratio * rc
    conv_half_angle = math.radians(32.0)
    conv_len = max(0.8 * rt, (rc - rt) / math.tan(conv_half_angle) + 0.55 * rt)
    throat_x = chamber_len + conv_len

    # Diverging bell length scaled from 15-deg reference cone
    cone_15_len = (re - rt) / math.tan(math.radians(15.0))
    div_len = max(1.0 * rt, (bell_percent / 100.0) * cone_15_len)
    total_len = throat_x + div_len

    theta_n_deg, theta_e_deg = _rao_bell_angles(expansion_ratio, bell_percent)
    theta_n = math.radians(theta_n_deg)
    theta_e = math.radians(theta_e_deg)

    # Circular-arc downstream throat fillet r_fillet = 0.382 * Rt from angle 0 to theta_n
    r_fillet_down = 0.382 * rt
    xn_rel = r_fillet_down * math.sin(theta_n)
    rn = rt + r_fillet_down * (1.0 - math.cos(theta_n))

    # Build Rao quadratic Bezier control point intersection for parabolic bell from N to Exit
    p0 = (throat_x + xn_rel, rn)
    p2 = (total_len, re)
    m1 = math.tan(theta_n)
    m2 = math.tan(theta_e)
    if abs(m1 - m2) < 1e-4:
        p1 = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]))
    else:
        c1 = p0[1] - m1 * p0[0]
        c2 = p2[1] - m2 * p2[0]
        x_int = (c2 - c1) / (m1 - m2)
        # Keep control point monotonically inside [p0_x, p2_x]
        x_int = max(p0[0] + 0.05 * (p2[0] - p0[0]), min(p2[0] - 0.05 * (p2[0] - p0[0]), x_int))
        y_int = m1 * x_int + c1
        p1 = (x_int, y_int)

    # Sample high-resolution continuous backbone spline and interpolate onto uniform CFD grid
    x_chamber = np.linspace(0.0, chamber_len, 80, endpoint=False)
    r_chamber = np.full_like(x_chamber, rc)

    # Smooth raised-cosine converging contraction (zero slope at both chamber end and throat!)
    s_conv = np.linspace(0.0, 1.0, 160, endpoint=False)
    x_conv = chamber_len + s_conv * conv_len
    r_conv = rt + 0.5 * (rc - rt) * (1.0 + np.cos(math.pi * s_conv))

    # Downstream circular throat arc from throat to inflection point N
    ang_fillet = np.linspace(0.0, theta_n, 60, endpoint=False)
    x_fillet = throat_x + r_fillet_down * np.sin(ang_fillet)
    r_fillet = rt + r_fillet_down * (1.0 - np.cos(ang_fillet))

    # Rao parabolic Bezier bell from N to Exit
    t_bell = np.linspace(0.0, 1.0, 200)
    x_bell, r_bell = _eval_quadratic_bezier(p0, p1, p2, t_bell)

    x_dense = np.concatenate([x_chamber, x_conv, x_fillet, x_bell])
    r_dense = np.concatenate([r_chamber, r_conv, r_fillet, r_bell])

    # Interpolate onto uniform axial CFD grid (required for MacCormack conservation law)
    x_grid = np.linspace(0.0, total_len, n_points)
    r_grid = np.interp(x_grid, x_dense, r_dense)
    # Enforce exact geometric minimum at closest throat index for sub-percent precision
    throat_idx = int(np.argmin(r_grid))
    r_grid[throat_idx] = min(r_grid[throat_idx], rt)

    area_grid = math.pi * (r_grid**2)
    at = math.pi * (rt**2)
    area_ratio = area_grid / at
    d_area_dx = np.gradient(area_grid, x_grid)
    dr_dx = np.gradient(r_grid, x_grid)
    wall_slope = np.arctan(dr_dx)

    # Effective Rao divergence efficiency factor weighted toward contoured bell exit angle
    divergence_eff = 0.5 * (1.0 + math.cos(0.35 * theta_n + 0.65 * theta_e))

    return NozzleGeometry(
        throat_radius_m=round(rt, 5),
        contraction_ratio=round(contraction_ratio, 3),
        expansion_ratio=round(expansion_ratio, 3),
        bell_percent=round(bell_percent, 1),
        theta_n_deg=theta_n_deg,
        theta_e_deg=theta_e_deg,
        divergence_efficiency=round(divergence_eff, 4),
        chamber_length_m=round(chamber_len, 5),
        converging_length_m=round(conv_len, 5),
        diverging_length_m=round(div_len, 5),
        total_length_m=round(total_len, 5),
        throat_x_m=round(throat_x, 5),
        x_m=x_grid,
        radius_m=r_grid,
        area_m2=area_grid,
        area_ratio=area_ratio,
        d_area_dx_m=d_area_dx,
        wall_slope_rad=wall_slope,
    )
