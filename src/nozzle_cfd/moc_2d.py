"""2D Axisymmetric Supersonic Nozzle Flow & Plume Shock Diamond Reconstruction.

Combines analytical Method-of-Characteristics (MOC) throat-expansion wave rays inside the bell
with the coupled **2D Axisymmetric Curvilinear Finite-Volume Euler / Navier-Stokes Solver**
(`navier_stokes_2d.py`), resolving genuine barrel shocks, Mach discs, triple points,
shear slip-lines, and optical Numerical Schlieren gradients instead of analytical waveforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nozzle_cfd.cfd_solver import NozzleCFDResult
from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.navier_stokes_2d import NavierStokes2DResult, solve_axisymmetric_navier_stokes_2d


@dataclass(frozen=True)
class Plume2DResult:
    """Two-dimensional `(x, r)` nozzle interior + exhaust plume Mach/P/T/Schlieren field."""

    x_m: np.ndarray
    r_m: np.ndarray
    mach_2d: np.ndarray
    pressure_kpa_2d: np.ndarray
    temperature_k_2d: np.ndarray
    upper_boundary_r_m: np.ndarray
    lower_boundary_r_m: np.ndarray
    shock_diamond_wavelength_m: float
    fully_expanded_mach: float
    prandtl_meyer_exit_deg: float
    moc_wall_x_m: np.ndarray
    moc_wall_r_m: np.ndarray
    moc_rays: list[tuple[list[float], list[float]]]
    schlieren_2d: np.ndarray | None = None
    solver_backend: str = "cpp20_openmp_dll"
    solve_ms: float = 0.0
    mach_disc_detected: bool = False
    separation_detected: bool = False
    discharge_coefficient_cd: float = 1.0
    ns2d_result: NavierStokes2DResult | None = None


def prandtl_meyer_deg(mach: float, gamma: float) -> float:
    """Prandtl-Meyer supersonic expansion angle nu(M) in degrees."""
    if mach <= 1.0:
        return 0.0
    gp1_gm1 = (gamma + 1.0) / (gamma - 1.0)
    m2_m1 = max(0.0, (mach**2) - 1.0)
    nu_rad = math.sqrt(gp1_gm1) * math.atan(math.sqrt(m2_m1 / gp1_gm1)) - math.atan(
        math.sqrt(m2_m1)
    )
    return math.degrees(nu_rad)


def _compute_moc_characteristic_net(
    geometry: NozzleGeometry,
    gamma: float,
    exit_mach: float,
    n_char_lines: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[tuple[list[float], list[float]]]]:
    """Generate 2D Method-of-Characteristics wave rays inside diverging supersonic core."""
    rt = geometry.throat_radius_m
    nu_max = 0.5 * prandtl_meyer_deg(max(1.2, exit_mach), gamma)
    rays: list[tuple[list[float], list[float]]] = []
    wall_pts_x = [geometry.throat_x_m]
    wall_pts_r = [rt]

    throat_idx = int(np.argmin(geometry.area_ratio))
    x_div = geometry.x_m[throat_idx:]
    r_div = geometry.radius_m[throat_idx:]

    # Trace C- waves from throat lip down to centerline (r=0) and reflect C+ out to nozzle wall
    for k in range(1, n_char_lines + 1):
        frac = k / (n_char_lines + 1)
        m_local = 1.08 + frac * max(0.3, exit_mach - 1.08)
        mu_rad = math.asin(1.0 / m_local)
        nu_rad = math.radians(frac * nu_max)
        # C- right-running characteristic slope: tan(theta - mu)
        slope_minus = math.tan(-nu_rad - mu_rad)
        x_start = geometry.throat_x_m + 0.04 * (k / n_char_lines) * geometry.diverging_length_m
        r_start = float(np.interp(x_start, x_div, r_div))
        x_cl = x_start - r_start / min(-0.08, slope_minus)
        x_cl = min(geometry.total_length_m * 0.96, x_cl)

        # C+ reflected wave from centerline up to outer bell contour
        slope_plus = math.tan(max(0.14, mu_rad + 0.25 * nu_rad))
        x_wall = x_cl + r_start / slope_plus
        x_wall = min(geometry.total_length_m, max(x_cl + 1e-4, x_wall))
        r_wall = float(np.interp(x_wall, x_div, r_div))

        rays.append(([x_start, x_cl, x_wall], [r_start, 0.0, r_wall]))
        wall_pts_x.append(x_wall)
        wall_pts_r.append(r_wall)

    return np.asarray(wall_pts_x), np.asarray(wall_pts_r), rays


def simulate_2d_nozzle_and_plume(
    geometry: NozzleGeometry,
    cfd_1d: NozzleCFDResult,
    plume_length_diameters: float = 3.8,
    nx_plume: int = 128,
    nr_half: int = 75,
    viscous_mode: str = "euler",
) -> Plume2DResult:
    """Solve coupled 2D Axisymmetric nozzle + exhaust plume field for ParaView & visualization."""
    gamma = cfd_1d.gamma
    pc_kpa = cfd_1d.chamber_pressure_mpa * 1.0e3
    pa_kpa = cfd_1d.ambient_pressure_kpa
    re = (
        geometry.exit_radius_m
        if hasattr(geometry, "exit_radius_m")
        else float(geometry.radius_m[-1])
    )

    # Analytical fully-expanded plume reference Mach and shock-diamond wavelength estimate
    pr_ratio = max(1.01, pc_kpa / max(1e-3, pa_kpa))
    m_j = math.sqrt(max(0.0, (2.0 / (gamma - 1.0)) * ((pr_ratio ** ((gamma - 1.0) / gamma)) - 1.0)))
    d_exit = 2.0 * re
    lam_pack = 1.22 * d_exit * math.sqrt(max(0.1, m_j**2 - 1.0))

    nx1 = 72
    nr1 = 26
    nx2 = min(140, max(64, int(nx_plume)))
    plume_len_radii = 2.0 * plume_length_diameters

    ns2d = solve_axisymmetric_navier_stokes_2d(
        geom=geometry,
        propellant_key=cfd_1d.propellant,
        chamber_pressure_mpa=cfd_1d.chamber_pressure_mpa,
        ambient_pressure_kpa=pa_kpa,
        viscous_mode=viscous_mode,
        limiter="van_leer",
        nx1=nx1,
        nr1=nr1,
        include_plume=True,
        nx2=nx2,
        plume_length_radii=plume_len_radii,
        max_iters=2000,
    )

    assert ns2d.x2_c is not None and ns2d.r2_c is not None and ns2d.mach_2 is not None
    x1_ax = ns2d.x1_c[:, 0]
    x2_ax = ns2d.x2_c[:, 0]
    x_m = np.concatenate([x1_ax, x2_ax])

    r_max = float(ns2d.r2_v[-1, -1]) if ns2d.r2_v is not None else 2.2 * re
    nr_h = max(35, int(nr_half))
    r_m = np.linspace(-r_max, r_max, 2 * nr_h - 1)
    abs_r = np.abs(r_m)

    nx_tot = len(x_m)
    nr_tot = len(r_m)
    mach_2d = np.full((nr_tot, nx_tot), np.nan, dtype=np.float64)
    pres_2d = np.full((nr_tot, nx_tot), pa_kpa, dtype=np.float64)
    temp_2d = np.full((nr_tot, nx_tot), 300.0, dtype=np.float64)
    sch_2d = np.zeros((nr_tot, nx_tot), dtype=np.float64)
    upper_r = np.empty(nx_tot, dtype=np.float64)

    # Interpolate Block 1 (internal nozzle) cleanly across |r| <= R_wall(x)
    r_wall_1 = np.interp(x1_ax, geometry.x_m, geometry.radius_m)
    for i in range(nx1):
        rw = float(r_wall_1[i])
        upper_r[i] = rw
        inside = abs_r <= (rw + 1e-6)
        r_src = ns2d.r1_c[i, :]
        mach_2d[inside, i] = np.interp(abs_r[inside], r_src, ns2d.mach_1[i, :])
        pres_2d[inside, i] = np.interp(abs_r[inside], r_src, ns2d.p_kpa_1[i, :])
        temp_2d[inside, i] = np.interp(abs_r[inside], r_src, ns2d.T_k_1[i, :])
        sch_2d[inside, i] = np.interp(abs_r[inside], r_src, ns2d.schlieren_1[i, :])

    # Interpolate Block 2 (external plume) across full radial box [0, r_max]
    r_src2 = ns2d.r2_c[0, :]
    for k in range(nx2):
        idx = nx1 + k
        mach_col = np.interp(abs_r, r_src2, ns2d.mach_2[k, :])
        mach_2d[:, idx] = mach_col
        pres_2d[:, idx] = np.interp(abs_r, r_src2, ns2d.p_kpa_2[k, :])
        temp_2d[:, idx] = np.interp(abs_r, r_src2, ns2d.T_k_2[k, :])
        sch_2d[:, idx] = np.interp(abs_r, r_src2, ns2d.schlieren_2[k, :])
        # Physical shear-layer boundary tracked by core jet Mach >= 0.45 envelope
        half_pos = r_src2[ns2d.mach_2[k, :] >= 0.45]
        upper_r[idx] = float(half_pos[-1]) if len(half_pos) > 0 else re

    moc_wx, moc_wr, moc_rays = _compute_moc_characteristic_net(
        geometry, gamma, float(np.nanmean(ns2d.mach_1[-1, :]))
    )
    lam_final = (
        float(ns2d.shock_diamond_wavelength_m)
        if ns2d.shock_diamond_wavelength_m is not None
        else lam_pack
    )
    nu_exit = prandtl_meyer_deg(float(cfd_1d.mach[-1]), gamma)

    return Plume2DResult(
        x_m=x_m,
        r_m=r_m,
        mach_2d=mach_2d,
        pressure_kpa_2d=pres_2d,
        temperature_k_2d=temp_2d,
        upper_boundary_r_m=upper_r,
        lower_boundary_r_m=-upper_r,
        shock_diamond_wavelength_m=round(lam_final, 4),
        fully_expanded_mach=round(m_j, 4),
        prandtl_meyer_exit_deg=round(nu_exit, 2),
        moc_wall_x_m=moc_wx,
        moc_wall_r_m=moc_wr,
        moc_rays=moc_rays,
        schlieren_2d=sch_2d,
        solver_backend=ns2d.solver_backend,
        solve_ms=ns2d.solve_ms,
        mach_disc_detected=ns2d.mach_disc_detected,
        separation_detected=ns2d.separation_detected,
        discharge_coefficient_cd=ns2d.discharge_coefficient_cd,
        ns2d_result=ns2d,
    )
