"""Quasi-1D Conservative MacCormack CFD Solver with Jameson Shock Capturing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.propellants import (
    compute_chamber_thermochemistry,
    mach_from_area_ratio,
    vandenkerckhove_gamma,
)


@dataclass(frozen=True)
class NozzleCFDResult:
    """Full 1D CFD axial solution compared against analytical Rankine-Hugoniot truth."""

    propellant: str
    chamber_pressure_mpa: float
    ambient_pressure_kpa: float
    gamma: float
    tc_k: float
    r_specific_j_kg_k: float
    x_m: np.ndarray
    area_ratio: np.ndarray
    mach: np.ndarray
    pressure_kpa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    mass_flow_x_kg_s: np.ndarray
    analytical_mach: np.ndarray
    analytical_pressure_kpa: np.ndarray
    shock_detected: bool
    shock_x_m: float | None
    shock_area_ratio: float | None
    shock_upstream_mach: float | None
    shock_downstream_mach: float | None
    flow_regime: str
    iterations: int
    converged: bool
    final_residual: float
    mass_conservation_error_pct: float
    mach_l2_error_pct: float
    pressure_l2_error_pct: float
    residual_history: list[float]


def _primitive_from_conservative(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    area_ratio: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover nondimensional (rho', V', T', p', M) from conservative state vector U."""
    u1_safe = np.maximum(u1, 1e-8)
    rho = np.maximum(u1_safe / area_ratio, 1e-8)
    vel = u2 / u1_safe
    # Energy relation: T' = (gamma - 1) * (U3/U1 - (gamma/2) * V'^2)
    temp = np.maximum((gamma - 1.0) * ((u3 / u1_safe) - 0.5 * gamma * (vel**2)), 1e-4)
    pres = rho * temp
    mach = vel / np.sqrt(temp)
    return rho, vel, temp, pres, mach


def _flux_from_conservative(
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Anderson Chapter 7 conservative Euler flux vector F(U)."""
    u1_safe = np.maximum(u1, 1e-8)
    f1 = u2.copy()
    kin = 0.5 * gamma * (u2**2) / u1_safe
    f2 = (u2**2) / u1_safe + ((gamma - 1.0) / gamma) * (u3 - kin)
    f3 = gamma * (u2 * u3) / u1_safe - 0.5 * gamma * (gamma - 1.0) * (u2**3) / (u1_safe**2)
    return f1, f2, f3


def exact_quasi1d_shock_solution(
    area_ratio: np.ndarray,
    throat_idx: int,
    gamma: float,
    pc_pa: float,
    tc_k: float,
    ambient_pa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int | None, float | None, float | None]:
    """Compute exact 1D isentropic + Rankine-Hugoniot normal shock analytical solution."""
    n = len(area_ratio)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # Isentropic supersonic exit values
    m_e_sup = mach_from_area_ratio(float(area_ratio[-1]), gamma, supersonic=True)
    p_e_sup = pc_pa * ((1.0 + 0.5 * gm1 * (m_e_sup**2)) ** (-gamma / gm1))

    # Post-normal-shock pressure right at nozzle exit plane
    p_e_shocked = p_e_sup * (1.0 + (2.0 * gamma / gp1) * (m_e_sup**2 - 1.0))

    # Purely isentropic subsonic limit (shock standing at throat, M <= 1 everywhere)
    m_e_sub = mach_from_area_ratio(float(area_ratio[-1]), gamma, supersonic=False)
    p_e_sub = pc_pa * ((1.0 + 0.5 * gm1 * (m_e_sub**2)) ** (-gamma / gm1))

    mach = np.ones(n, dtype=float)
    p0_local = np.full(n, pc_pa, dtype=float)

    if ambient_pa <= p_e_shocked:
        # Supersonic throughout diverging section (shock, if any, is outside in plume)
        for i in range(n):
            is_sup = i > throat_idx
            mach[i] = mach_from_area_ratio(float(area_ratio[i]), gamma, supersonic=is_sup)
        temp = tc_k / (1.0 + 0.5 * gm1 * (mach**2))
        pres = p0_local * ((1.0 + 0.5 * gm1 * (mach**2)) ** (-gamma / gm1))
        return mach, pres, temp, False, None, None, None

    if ambient_pa >= p_e_sub:
        # Unchoked subsonic venturi flow matched to ambient_pa at exit
        pr_target = min(0.9999, max(0.05, ambient_pa / pc_pa))
        m_exit_unchoked = math.sqrt((2.0 / gm1) * ((pr_target ** (-gm1 / gamma)) - 1.0))
        a_star_eff = (
            float(area_ratio[-1])
            * m_exit_unchoked
            * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * (m_exit_unchoked**2))) ** (-gp1 / (2.0 * gm1))
        )
        for i in range(n):
            mach[i] = mach_from_area_ratio(
                max(1.0, float(area_ratio[i]) / max(1e-6, a_star_eff)),
                gamma,
                supersonic=False,
            )
        temp = tc_k / (1.0 + 0.5 * gm1 * (mach**2))
        pres = pc_pa * ((1.0 + 0.5 * gm1 * (mach**2)) ** (-gamma / gm1))
        return mach, pres, temp, False, None, None, None

    # Standing normal shock inside diverging nozzle (p_e_shocked < ambient_pa < p_e_sub)
    target_exit_factor = (ambient_pa * float(area_ratio[-1])) / (
        pc_pa * ((2.0 / gp1) ** (gp1 / (2.0 * gm1)))
    )

    # Solve post-shock exit Mach from choked mass flow identity P*A = const through P02:
    # For isentropic post-shock flow to exit: Pe * Ae / (P02 * A*_2) = same constant ->
    # P02 * A*_2 = P01 * A*_1! Thus (Pe/P02)*(Ae/A*_2) = P01*A*_1/(P02*A*_2)?
    # Even simpler: directly match p_exit(shock_area_ratio) == ambient_pa via Brent!
    def exit_pressure_residual(a_shock: float) -> float:
        m1 = mach_from_area_ratio(a_shock, gamma, supersonic=True)
        # Normal shock total pressure ratio P02/P01
        t1 = (gp1 * (m1**2)) / (gm1 * (m1**2) + 2.0)
        t2 = gp1 / (2.0 * gamma * (m1**2) - gm1)
        p02_p01 = (t1 ** (gamma / gm1)) * (t2 ** (1.0 / gm1))
        # Post-shock A*_2 = A*_1 / (P02 / P01)
        a_ratio_post = float(area_ratio[-1]) * p02_p01
        if a_ratio_post <= 1.0:
            return pc_pa * p02_p01 - ambient_pa
        m_exit = mach_from_area_ratio(a_ratio_post, gamma, supersonic=False)
        p_exit = pc_pa * p02_p01 * ((1.0 + 0.5 * gm1 * (m_exit**2)) ** (-gamma / gm1))
        return p_exit - ambient_pa

    a_shock = float(brentq(exit_pressure_residual, 1.0001, float(area_ratio[-1]) - 1e-5, xtol=1e-8))
    _ = target_exit_factor
    m1_shock = mach_from_area_ratio(a_shock, gamma, supersonic=True)
    m2_shock = math.sqrt((1.0 + 0.5 * gm1 * (m1_shock**2)) / (gamma * (m1_shock**2) - 0.5 * gm1))
    t1 = (gp1 * (m1_shock**2)) / (gm1 * (m1_shock**2) + 2.0)
    t2 = gp1 / (2.0 * gamma * (m1_shock**2) - gm1)
    p02_p01 = (t1 ** (gamma / gm1)) * (t2 ** (1.0 / gm1))

    # Locate closest grid point after throat where area_ratio crosses a_shock
    div_slice = area_ratio[throat_idx:]
    shock_rel = int(np.argmin(np.abs(div_slice - a_shock)))
    shock_idx = throat_idx + shock_rel

    for i in range(n):
        if i <= throat_idx:
            mach[i] = mach_from_area_ratio(float(area_ratio[i]), gamma, supersonic=False)
        elif i < shock_idx:
            mach[i] = mach_from_area_ratio(float(area_ratio[i]), gamma, supersonic=True)
        else:
            p0_local[i] = pc_pa * p02_p01
            a_star2_ratio = max(1.00001, float(area_ratio[i]) * p02_p01)
            mach[i] = mach_from_area_ratio(a_star2_ratio, gamma, supersonic=False)

    temp = tc_k / (1.0 + 0.5 * gm1 * (mach**2))
    pres = p0_local * ((1.0 + 0.5 * gm1 * (mach**2)) ** (-gamma / gm1))
    return mach, pres, temp, True, shock_idx, m1_shock, m2_shock


def solve_quasi1d_maccormack(
    geometry: NozzleGeometry,
    propellant_key: str = "METHALOX",
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    cfl: float = 0.45,
    max_iter: int = 8000,
    tol: float = 1e-6,
    cx_viscosity: float = 0.30,
) -> NozzleCFDResult:
    """Solve Conservative Quasi-1D Euler equations using explicit MacCormack CFD.

    Captures smooth subsonic-to-supersonic acceleration as well as standing normal shock
    waves inside overexpanded nozzles using a Jameson pressure-sensor dissipation term.
    """
    _, _, tc_k, gamma, mw = compute_chamber_thermochemistry(
        propellant_key, chamber_pressure_mpa=chamber_pressure_mpa
    )
    r_spec = 8314.4626 / mw
    pc_pa = chamber_pressure_mpa * 1.0e6
    ambient_pa = max(0.01, ambient_pressure_kpa * 1.0e3)
    rho_c = pc_pa / (r_spec * tc_k)
    a_c = math.sqrt(gamma * r_spec * tc_k)

    x = geometry.x_m
    dx = float(x[1] - x[0])
    dx_nd = dx / geometry.throat_radius_m
    area_ratio = geometry.area_ratio.copy()
    dA_dx_nd = (
        geometry.d_area_dx_m * geometry.throat_radius_m / (math.pi * (geometry.throat_radius_m**2))
    )
    throat_idx = int(np.argmin(area_ratio))
    n = len(x)

    # Analytical baseline (used both to initialize smoothly or verify shock convergence)
    ana_mach, ana_pres, _, ana_shock, ana_shock_idx, _, _ = exact_quasi1d_shock_solution(
        area_ratio, throat_idx, gamma, pc_pa, tc_k, ambient_pa
    )

    # Check if back-pressure forces subsonic outflow (standing shock inside or unchoked)
    m_e_sup = mach_from_area_ratio(float(area_ratio[-1]), gamma, supersonic=True)
    p_e_sup = pc_pa * ((1.0 + 0.5 * (gamma - 1.0) * (m_e_sup**2)) ** (-gamma / (gamma - 1.0)))
    p_e_shocked = p_e_sup * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (m_e_sup**2 - 1.0))
    subsonic_outflow = ambient_pa > p_e_shocked
    pb_nd = ambient_pa / pc_pa

    # Smooth initial guess (shock emerges dynamically if Pe < Pb)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gamma_fn = vandenkerckhove_gamma(gamma)
    m_init = np.array(
        [
            mach_from_area_ratio(float(area_ratio[i]), gamma, supersonic=(i > throat_idx))
            for i in range(n)
        ],
        dtype=float,
    )
    if subsonic_outflow:
        # Linear deceleration after throat helps converge smoothly to shock position
        blend = np.linspace(0.0, 1.0, n - throat_idx)
        m_init[throat_idx:] = np.maximum(0.25, m_init[throat_idx:] * (1.0 - 0.65 * blend))

    t_init = 1.0 / (1.0 + 0.5 * gm1 * (m_init**2))
    rho_init = t_init ** (1.0 / gm1)
    v_init = m_init * np.sqrt(t_init)

    u1 = rho_init * area_ratio
    u2 = rho_init * area_ratio * v_init
    u3 = rho_init * area_ratio * (t_init / gm1 + 0.5 * gamma * (v_init**2))

    residual_history: list[float] = []
    converged = False
    final_residual = 1.0
    iters_run = 0

    for it in range(1, max_iter + 1):
        rho, vel, temp, pres, _ = _primitive_from_conservative(u1, u2, u3, area_ratio, gamma)
        a_local = np.sqrt(temp)
        dt = float(cfl * dx_nd / np.max(np.abs(vel) + a_local))

        # Adaptive dissipation: small in smooth supersonic flow, stronger if shock present
        cx = cx_viscosity if subsonic_outflow else 0.08
        pres_safe = np.maximum(pres, 1e-6)
        sensor = np.zeros(n, dtype=float)
        sensor[1:-1] = (
            cx
            * np.abs(pres_safe[2:] - 2.0 * pres_safe[1:-1] + pres_safe[:-2])
            / np.maximum(pres_safe[2:] + 2.0 * pres_safe[1:-1] + pres_safe[:-2], 1e-8)
        )

        # ---- Predictor step (forward spatial differences) ----
        f1, f2, f3 = _flux_from_conservative(u1, u2, u3, gamma)
        _, _, _, _, _ = rho, vel, temp, pres, _

        u1_p = u1.copy()
        u2_p = u2.copy()
        u3_p = u3.copy()

        j2 = (1.0 / gamma) * pres * dA_dx_nd
        s1_p = sensor[1:-1] * (u1[2:] - 2.0 * u1[1:-1] + u1[:-2])
        s2_p = sensor[1:-1] * (u2[2:] - 2.0 * u2[1:-1] + u2[:-2])
        s3_p = sensor[1:-1] * (u3[2:] - 2.0 * u3[1:-1] + u3[:-2])

        du1_p = -(f1[2:] - f1[1:-1]) / dx_nd
        du2_p = -(f2[2:] - f2[1:-1]) / dx_nd + j2[1:-1]
        du3_p = -(f3[2:] - f3[1:-1]) / dx_nd

        u1_p[1:-1] = u1[1:-1] + dt * du1_p + s1_p
        u2_p[1:-1] = u2[1:-1] + dt * du2_p + s2_p
        u3_p[1:-1] = u3[1:-1] + dt * du3_p + s3_p

        # Boundary conditions on predicted state
        # Inflow (i=0): extrapolate U2, fix stagnation T0'=1, P0'=1
        u2_p[0] = max(0.01 * area_ratio[0], 2.0 * u2_p[1] - u2_p[2])
        v0_p = min(0.85, max(0.01, u2_p[0] / max(u1[0], 1e-6)))
        t0_p = max(0.5, 1.0 - 0.5 * gm1 * (v0_p**2))
        rho0_p = t0_p ** (1.0 / gm1)
        u1_p[0] = rho0_p * area_ratio[0]
        u3_p[0] = u1_p[0] * (t0_p / gm1 + 0.5 * gamma * (v0_p**2))

        # Outflow (i=-1)
        u1_p[-1] = max(1e-6, 2.0 * u1_p[-2] - u1_p[-3])
        u2_p[-1] = max(1e-6, 2.0 * u2_p[-2] - u2_p[-3])
        if subsonic_outflow:
            u3_p[-1] = (pb_nd * area_ratio[-1]) / gm1 + 0.5 * gamma * (u2_p[-1] ** 2) / u1_p[-1]
        else:
            u3_p[-1] = max(1e-6, 2.0 * u3_p[-2] - u3_p[-3])

        # ---- Corrector step (backward spatial differences) ----
        _, _, _, pres_p, _ = _primitive_from_conservative(u1_p, u2_p, u3_p, area_ratio, gamma)
        f1_p, f2_p, f3_p = _flux_from_conservative(u1_p, u2_p, u3_p, gamma)
        pres_p_safe = np.maximum(pres_p, 1e-6)
        sensor_p = np.zeros(n, dtype=float)
        sensor_p[1:-1] = (
            cx
            * np.abs(pres_p_safe[2:] - 2.0 * pres_p_safe[1:-1] + pres_p_safe[:-2])
            / np.maximum(pres_p_safe[2:] + 2.0 * pres_p_safe[1:-1] + pres_p_safe[:-2], 1e-8)
        )
        s1_c = sensor_p[1:-1] * (u1_p[2:] - 2.0 * u1_p[1:-1] + u1_p[:-2])
        s2_c = sensor_p[1:-1] * (u2_p[2:] - 2.0 * u2_p[1:-1] + u2_p[:-2])
        s3_c = sensor_p[1:-1] * (u3_p[2:] - 2.0 * u3_p[1:-1] + u3_p[:-2])

        j2_p = (1.0 / gamma) * pres_p * dA_dx_nd
        du1_c = -(f1_p[1:-1] - f1_p[:-2]) / dx_nd
        du2_c = -(f2_p[1:-1] - f2_p[:-2]) / dx_nd + j2_p[1:-1]
        du3_c = -(f3_p[1:-1] - f3_p[:-2]) / dx_nd

        u1_new = u1.copy()
        u2_new = u2.copy()
        u3_new = u3.copy()

        u1_new[1:-1] = 0.5 * (u1[1:-1] + u1_p[1:-1] + dt * du1_c) + 0.5 * s1_c
        u2_new[1:-1] = 0.5 * (u2[1:-1] + u2_p[1:-1] + dt * du2_c) + 0.5 * s2_c
        u3_new[1:-1] = 0.5 * (u3[1:-1] + u3_p[1:-1] + dt * du3_c) + 0.5 * s3_c

        # Final boundary updates
        u2_new[0] = max(0.01 * area_ratio[0], 2.0 * u2_new[1] - u2_new[2])
        v0 = min(0.85, max(0.01, u2_new[0] / max(u1_new[0], 1e-6)))
        t0 = max(0.5, 1.0 - 0.5 * gm1 * (v0**2))
        rho0 = t0 ** (1.0 / gm1)
        u1_new[0] = rho0 * area_ratio[0]
        u3_new[0] = u1_new[0] * (t0 / gm1 + 0.5 * gamma * (v0**2))

        u1_new[-1] = max(1e-6, 2.0 * u1_new[-2] - u1_new[-3])
        u2_new[-1] = max(1e-6, 2.0 * u2_new[-2] - u2_new[-3])
        if subsonic_outflow:
            u3_new[-1] = (pb_nd * area_ratio[-1]) / gm1 + 0.5 * gamma * (u2_new[-1] ** 2) / u1_new[
                -1
            ]
        else:
            u3_new[-1] = max(1e-6, 2.0 * u3_new[-2] - u3_new[-3])

        diff = float(np.max(np.abs(u2_new - u2) / np.maximum(np.abs(u2), 1e-4)))
        u1, u2, u3 = u1_new, u2_new, u3_new
        iters_run = it

        if it == 1 or it % 50 == 0 or diff < tol:
            residual_history.append(round(diff, 8))
        final_residual = diff
        if it > 200 and diff < tol:
            converged = True
            break

    rho_nd, vel_nd, temp_nd, pres_nd, mach = _primitive_from_conservative(
        u1, u2, u3, area_ratio, gamma
    )
    pressure_kpa = (pres_nd * pc_pa) / 1.0e3
    temperature_k = temp_nd * tc_k
    density_kg_m3 = rho_nd * rho_c
    velocity_m_s = vel_nd * a_c
    at_m2 = math.pi * (geometry.throat_radius_m**2)
    mdot_x = density_kg_m3 * (area_ratio * at_m2) * velocity_m_s

    # Detect normal shock wave directly from negative Mach jump downstream of throat
    dmach = np.diff(mach)
    shock_detected = False
    shock_x_m: float | None = None
    shock_ar: float | None = None
    shock_m1: float | None = None
    shock_m2: float | None = None

    search_start = max(throat_idx + 2, 1)
    if search_start < n - 3:
        min_dm_rel = int(np.argmin(dmach[search_start:]))
        shock_cell = search_start + min_dm_rel
        # Normal shock signature: Mach drops from > 1.05 across a steep negative gradient
        if mach[max(0, shock_cell - 2)] > 1.05 and dmach[shock_cell] < -0.08:
            shock_detected = True
            shock_x_m = round(float(x[shock_cell]), 5)
            shock_ar = round(float(area_ratio[shock_cell]), 3)
            shock_m1 = round(float(np.max(mach[max(0, shock_cell - 3) : shock_cell + 1])), 3)
            shock_m2 = round(float(np.min(mach[shock_cell + 1 : min(n, shock_cell + 5)])), 3)

    if shock_detected:
        regime = "NORMAL_SHOCK_INSIDE_NOZZLE"
    elif mach[-1] < 0.95:
        regime = "UNCHOKED_SUBSONIC_VENTURI"
    elif pressure_kpa[-1] < 0.85 * ambient_pressure_kpa:
        regime = "OVEREXPANDED_SUPERSONIC_PLUME_SHOCKS"
    elif pressure_kpa[-1] > 1.15 * ambient_pressure_kpa:
        regime = "UNDEREXPANDED_SUPERSONIC_EXPANSION"
    else:
        regime = "DESIGN_ISENTROPIC_SUPERSONIC"

    # Exclude 2-cell shock-smearing band from L2 comparison for true physical error metric
    mask = np.ones(n, dtype=bool)
    if ana_shock and ana_shock_idx is not None:
        lo = max(0, ana_shock_idx - 3)
        hi = min(n, ana_shock_idx + 4)
        mask[lo:hi] = False

    # Mass conservation error outside the shock smearing layer
    valid_mdot = mdot_x[mask][1:-1] if np.sum(mask) > 6 else mdot_x[1:-1]
    mdot_mean = float(np.mean(valid_mdot))
    mass_err_pct = float(100.0 * np.std(valid_mdot) / max(abs(mdot_mean), 1e-6))

    ana_pres_kpa = ana_pres / 1.0e3
    mach_l2 = float(
        100.0
        * np.linalg.norm(mach[mask] - ana_mach[mask])
        / max(np.linalg.norm(ana_mach[mask]), 1e-6)
    )
    pres_l2 = float(
        100.0
        * np.linalg.norm(pressure_kpa[mask] - ana_pres_kpa[mask])
        / max(np.linalg.norm(ana_pres_kpa[mask]), 1e-6)
    )

    _ = gp1, gamma_fn

    return NozzleCFDResult(
        propellant=propellant_key.upper(),
        chamber_pressure_mpa=round(chamber_pressure_mpa, 3),
        ambient_pressure_kpa=round(ambient_pressure_kpa, 3),
        gamma=round(gamma, 4),
        tc_k=round(tc_k, 2),
        r_specific_j_kg_k=round(r_spec, 2),
        x_m=x,
        area_ratio=area_ratio,
        mach=mach,
        pressure_kpa=pressure_kpa,
        temperature_k=temperature_k,
        density_kg_m3=density_kg_m3,
        velocity_m_s=velocity_m_s,
        mass_flow_x_kg_s=mdot_x,
        analytical_mach=ana_mach,
        analytical_pressure_kpa=ana_pres_kpa,
        shock_detected=shock_detected,
        shock_x_m=shock_x_m,
        shock_area_ratio=shock_ar,
        shock_upstream_mach=shock_m1,
        shock_downstream_mach=shock_m2,
        flow_regime=regime,
        iterations=iters_run,
        converged=converged,
        final_residual=final_residual,
        mass_conservation_error_pct=round(mass_err_pct, 4),
        mach_l2_error_pct=round(mach_l2, 4),
        pressure_l2_error_pct=round(pres_l2, 4),
        residual_history=residual_history,
    )
