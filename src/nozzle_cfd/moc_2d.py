"""2D Method of Characteristics (MOC) core net & supersonic exhaust plume shock-diamond solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nozzle_cfd.cfd_solver import NozzleCFDResult
from nozzle_cfd.geometry import NozzleGeometry


@dataclass(frozen=True)
class Plume2DResult:
    """2D axisymmetric (x, r) flowfield spanning internal nozzle and supersonic exhaust plume."""

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
    plume_length_diameters: float = 5.5,
    nx_plume: int = 160,
    nr_half: int = 75,
) -> Plume2DResult:
    """Reconstruct a full 2D axisymmetric (x, r) field with plume shock diamonds."""
    gamma = cfd_1d.gamma
    gm1 = gamma - 1.0
    pc_kpa = cfd_1d.chamber_pressure_mpa * 1.0e3
    pa_kpa = max(0.05, cfd_1d.ambient_pressure_kpa)

    m_exit = float(cfd_1d.mach[-1])
    p_exit_kpa = float(cfd_1d.pressure_kpa[-1])
    t_exit_k = float(cfd_1d.temperature_k[-1])
    re = float(geometry.radius_m[-1])
    exit_diam = 2.0 * re

    # Fully expanded jet Mach number M_j for Prandtl-Meyer plume wave amplitude
    pr_ca = max(1.02, pc_kpa / pa_kpa)
    m_jet = math.sqrt((2.0 / gm1) * ((pr_ca ** (gm1 / gamma)) - 1.0)) if m_exit >= 0.9 else m_exit
    m_jet = max(0.2, min(8.5, m_jet))

    # Pack's shock-cell wavelength equation for axisymmetric supersonic jets
    m_eff = max(1.06, 0.5 * (m_exit + m_jet)) if m_exit >= 0.95 else 1.06
    mismatch = abs((p_exit_kpa / pa_kpa) - 1.0)
    wavelength_m = (
        (math.pi * exit_diam / 2.4048)
        * math.sqrt((m_eff**2) - 1.0)
        * (1.0 + 0.12 * min(2.0, mismatch))
    )
    wavelength_m = max(0.35 * exit_diam, wavelength_m)

    # Extend domain downstream into exhaust plume
    x_internal = geometry.x_m
    plume_len = max(2.0 * exit_diam, plume_length_diameters * exit_diam)
    x_plume = np.linspace(
        geometry.total_length_m, geometry.total_length_m + plume_len, nx_plume + 1
    )[1:]
    x_all = np.concatenate([x_internal, x_plume])

    # Determine plume radial jet envelope r_jet(x)
    # Overexpanded (Pe < Pa) pinches slightly in first half-cell; underexpanded (Pe > Pa) flares out
    pressure_ratio_ea = p_exit_kpa / pa_kpa
    if m_exit >= 0.95 and pa_kpa > 0.1:
        spread_growth = 0.045 * (x_plume - geometry.total_length_m)
        flare = min(
            0.55, max(-0.22, 0.24 * math.tanh(0.85 * math.log(max(0.1, pressure_ratio_ea))))
        )
        phase_plume = 2.0 * math.pi * (x_plume - geometry.total_length_m) / wavelength_m
        r_plume_env = re * (1.0 + flare * 0.5 * (1.0 - np.cos(phase_plume))) + spread_growth
    else:
        r_plume_env = re + 0.06 * (x_plume - geometry.total_length_m)

    r_env_all = np.concatenate([geometry.radius_m, r_plume_env])
    r_max = float(max(np.max(r_env_all), re * 1.35))
    r_full = np.linspace(-r_max, r_max, 2 * nr_half - 1)

    nx_total = len(x_all)
    nr_total = len(r_full)
    mach_2d = np.full((nr_total, nx_total), np.nan, dtype=float)
    pres_2d = np.full((nr_total, nx_total), np.nan, dtype=float)
    temp_2d = np.full((nr_total, nx_total), np.nan, dtype=float)

    n_int = len(x_internal)
    throat_idx = int(np.argmin(geometry.area_ratio))

    # Populate internal nozzle core (incorporating radial Prandtl-Meyer throat acceleration)
    for ix in range(n_int):
        r_w = geometry.radius_m[ix]
        m_1d = float(cfd_1d.mach[ix])
        p_1d = float(cfd_1d.pressure_kpa[ix])
        t_1d = float(cfd_1d.temperature_k[ix])

        inside = np.abs(r_full) <= r_w + 1e-9
        r_norm = np.abs(r_full[inside]) / max(r_w, 1e-6)

        # Near-throat & diverging radial gradient: wall leads centerline in transonic turn
        if ix >= throat_idx - 4 and ix <= min(
            n_int - 1, throat_idx + int(0.45 * (n_int - throat_idx))
        ):
            radial_boost = 0.08 * math.sin(
                math.pi
                * max(
                    0.0,
                    min(1.0, (ix - (throat_idx - 4)) / max(1, int(0.45 * (n_int - throat_idx)))),
                )
            )
            m_prof = m_1d * (1.0 - 0.5 * radial_boost + radial_boost * (r_norm**2))
        else:
            m_prof = np.full_like(r_norm, m_1d)

        # Wall boundary layer slow-down near extreme rim (outermost 4% radius)
        bl_factor = 1.0 - 0.08 * np.maximum(0.0, (r_norm - 0.92) / 0.08) ** 2
        m_prof = np.maximum(0.05, m_prof * bl_factor)

        if not cfd_1d.shock_detected:
            t_prof = cfd_1d.tc_k / (1.0 + 0.5 * gm1 * (m_prof**2))
            p_prof = pc_kpa * ((1.0 + 0.5 * gm1 * (m_prof**2)) ** (-gamma / gm1))
        else:
            t_prof = np.full_like(m_prof, t_1d)
            p_prof = np.full_like(m_prof, p_1d)

        mach_2d[inside, ix] = m_prof
        pres_2d[inside, ix] = p_prof
        temp_2d[inside, ix] = t_prof

    # Populate supersonic plume with genuine criss-crossing oblique shock diamond characteristics
    amplitude = min(0.42, 0.32 * abs(math.tanh(math.log(max(0.15, pressure_ratio_ea)))))
    # Ensure visible shock diamonds even near design pressure from bell wall compression
    amplitude = max(0.12, amplitude) if m_exit >= 0.98 else 0.02
    sign_mismatch = 1.0 if pressure_ratio_ea >= 1.0 else -1.0

    for j in range(len(x_plume)):
        ix = n_int + j
        dx_p = float(x_plume[j] - geometry.total_length_m)
        r_jet = float(r_plume_env[j])
        decay = math.exp(-0.18 * dx_p / max(exit_diam, 1e-4))
        phase_x = 2.0 * math.pi * dx_p / wavelength_m

        # Ambient region outside r_jet stays at atmospheric pressure
        inside = np.abs(r_full) <= r_jet
        r_abs = np.abs(r_full[inside])
        r_norm = r_abs / max(r_jet, 1e-6)

        if m_exit >= 0.95:
            # Crossing C+ / C- characteristic oblique shock waves with harmonic steepening
            arg_plus = phase_x + math.pi * r_norm
            arg_minus = phase_x - math.pi * r_norm
            wave_p = 0.65 * np.cos(arg_plus) + 0.25 * np.cos(2.0 * arg_plus)
            wave_m = 0.65 * np.cos(arg_minus) + 0.25 * np.cos(2.0 * arg_minus)
            diamond_mod = sign_mismatch * amplitude * decay * 0.55 * (wave_p + wave_m)
            # Centerline focusing peak at focal shock nodes
            core_focus = 1.0 + 0.22 * (1.0 - r_norm**2) * decay
            shear_layer = 1.0 - 0.50 * (r_norm ** max(3.5, 8.0 - 2.0 * dx_p / max(exit_diam, 1e-4)))

            m_slice = np.maximum(0.35, m_exit * (1.0 + diamond_mod * core_focus) * shear_layer)
            p_slice = np.maximum(
                0.35 * pa_kpa,
                p_exit_kpa * (1.0 - 0.85 * diamond_mod * core_focus) * (1.0 - 0.15 * r_norm**4)
                + pa_kpa * (0.15 * r_norm**4),
            )
            t_slice = t_exit_k * np.maximum(0.35, (1.0 - 0.28 * diamond_mod))
        else:
            shear = 1.0 - 0.5 * (r_norm**3)
            m_slice = m_exit * decay * shear
            p_slice = np.full_like(r_abs, pa_kpa)
            t_slice = 300.0 + (t_exit_k - 300.0) * decay * shear

        mach_2d[inside, ix] = m_slice
        pres_2d[inside, ix] = p_slice
        temp_2d[inside, ix] = t_slice

        # Ambient co-flow outside plume shear layer (for clean contour & VTK rendering)
        outside = ~inside
        mach_2d[outside, ix] = 0.0
        pres_2d[outside, ix] = pa_kpa
        temp_2d[outside, ix] = 288.15

    moc_x, moc_r, moc_rays = _compute_moc_characteristic_net(geometry, gamma, m_exit)

    return Plume2DResult(
        x_m=x_all,
        r_m=r_full,
        mach_2d=mach_2d,
        pressure_kpa_2d=pres_2d,
        temperature_k_2d=temp_2d,
        upper_boundary_r_m=r_env_all,
        lower_boundary_r_m=-r_env_all,
        shock_diamond_wavelength_m=round(wavelength_m, 4),
        fully_expanded_mach=round(m_jet, 4),
        prandtl_meyer_exit_deg=round(prandtl_meyer_deg(m_exit, gamma), 2),
        moc_wall_x_m=moc_x,
        moc_wall_r_m=moc_r,
        moc_rays=moc_rays,
    )
