"""Finite-Volume Approximate Riemann Solvers (HLLC, AUSM+, Rusanov), MUSCL-TVD, Sod Benchmark & GCI.

Why this exists (and how it climbs beyond introductory CFD):
    1. Finite-Difference MacCormack (1969) requires tuning scalar artificial viscosity (`cx`)
       to stop oscillations at strong shock fronts.
    2. Modern aerospace CFD solvers (SU2, FUN3D, OpenFOAM/rhoCentralFoam) discretize the
       integral conservation laws directly across cell interfaces using **Godunov Finite-Volume
       Riemann Solvers** (HLLC, AUSM+) + **MUSCL-TVD Slope Limiters** (Van Leer, Minmod,
       Superbee) + **SSP-RK2** time integration.
    3. This module validates our solver against Toro's exact iterative **Sod Shock Tube**
       analytical truth AND performs formal **ASME V&V 20 / Roache Grid Convergence Index (GCI)**
       mesh-refinement verification across 3 grids.
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nozzle_cfd.cfd_solver import exact_quasi1d_shock_solution
from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.propellants import (
    compute_chamber_thermochemistry,
    mach_from_area_ratio,
)

VALID_FLUX_SCHEMES = ("hllc", "ausm_plus", "rusanov")
VALID_LIMITERS = ("first_order", "minmod", "van_leer", "superbee", "mc")


@dataclass(frozen=True)
class SodShockTubeResult:
    """Verification of 1D Finite-Volume Riemann solver against Toro's exact Sod solution."""

    flux_scheme: str
    limiter: str
    nx: int
    t_end: float
    gamma: float
    x: np.ndarray
    rho_exact: np.ndarray
    u_exact: np.ndarray
    p_exact: np.ndarray
    mach_exact: np.ndarray
    rho_num: np.ndarray
    u_num: np.ndarray
    p_num: np.ndarray
    mach_num: np.ndarray
    rho_l1_error: float
    rho_l2_error: float
    u_l2_error: float
    p_l2_error: float
    solve_ms: float


@dataclass(frozen=True)
class NozzleFVMResult:
    """High-resolution Finite-Volume HLLC / AUSM+ Quasi-1D Rocket Nozzle CFD solution."""

    propellant: str
    flux_scheme: str
    limiter: str
    solver_backend: str
    solve_ms: float
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


@dataclass(frozen=True)
class GridConvergenceResult:
    """ASME V&V 20 / Roache Grid Convergence Index (GCI) mesh-refinement verification."""

    propellant: str
    flux_scheme: str
    limiter: str
    grid_sizes: tuple[int, int, int]
    refinement_ratio: float
    exit_mach_values: tuple[float, float, float]
    mach_l2_errors_pct: tuple[float, float, float]
    pressure_l2_errors_pct: tuple[float, float, float]
    observed_order_error: float
    richardson_extrapolated_exit_mach: float
    gci_fine_pct: float
    gci_coarse_pct: float
    asymptotic_range_ratio: float
    is_grid_converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "propellant": self.propellant,
            "flux_scheme": self.flux_scheme,
            "limiter": self.limiter,
            "grid_sizes": list(self.grid_sizes),
            "refinement_ratio": self.refinement_ratio,
            "exit_mach_values": list(self.exit_mach_values),
            "mach_l2_errors_pct": list(self.mach_l2_errors_pct),
            "pressure_l2_errors_pct": list(self.pressure_l2_errors_pct),
            "observed_order_error": self.observed_order_error,
            "richardson_extrapolated_exit_mach": self.richardson_extrapolated_exit_mach,
            "gci_fine_pct": self.gci_fine_pct,
            "gci_coarse_pct": self.gci_coarse_pct,
            "asymptotic_range_ratio": self.asymptotic_range_ratio,
            "is_grid_converged": self.is_grid_converged,
        }


def compute_tvd_slopes(q_pad: np.ndarray, limiter: str = "van_leer") -> np.ndarray:
    """Compute TVD-limited slopes for cells `1 .. len(q_pad) - 2`.

    Total Variation Diminishing (TVD) limiters measure the smoothness ratio `r = dq_- / dq_+`.
    At smooth peaks or flat zones where neighbor slopes reverse sign (`r <= 0`), the slope
    is clamped to 0 so new unphysical extrema (shock wiggles) can never appear.
    """
    lim = limiter.strip().lower()
    if lim not in VALID_LIMITERS:
        raise ValueError(f"Unknown limiter '{limiter}'. Choose from: {VALID_LIMITERS}")
    if lim == "first_order":
        return np.zeros(len(q_pad) - 2, dtype=float)

    dq_minus = q_pad[1:-1] - q_pad[:-2]
    dq_plus = q_pad[2:] - q_pad[1:-1]

    same_sign = (dq_minus * dq_plus) > 0.0
    denom = np.where(np.abs(dq_plus) > 1e-14, dq_plus, np.copysign(1e-14, dq_plus + 1e-30))
    r = np.where(same_sign, dq_minus / denom, 0.0)

    if lim == "minmod":
        phi = np.maximum(0.0, np.minimum(1.0, r))
    elif lim == "van_leer":
        phi = np.where(r > 0.0, (2.0 * r) / (1.0 + r), 0.0)
    elif lim == "superbee":
        phi = np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0)))
    else:  # "mc" (Monotonized Central)
        phi = np.maximum(0.0, np.minimum(2.0 * r, np.minimum(0.5 * (1.0 + r), 2.0)))

    return np.where(same_sign, phi * dq_plus, 0.0)


def reconstruct_muscl_interfaces(
    rho_pad: np.ndarray,
    vel_pad: np.ndarray,
    pres_pad: np.ndarray,
    limiter: str = "van_leer",
    shock_damping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct primitive states (rho, u, p) cleanly on all N+1 cell interfaces."""
    s_rho = compute_tvd_slopes(rho_pad, limiter)
    s_vel = compute_tvd_slopes(vel_pad, limiter)
    s_pres = compute_tvd_slopes(pres_pad, limiter)

    if shock_damping is not None:
        s_rho = s_rho * shock_damping
        s_vel = s_vel * shock_damping
        s_pres = s_pres * shock_damping

    rho_l = np.maximum(1e-9, rho_pad[1:-2] + 0.5 * s_rho[:-1])
    rho_r = np.maximum(1e-9, rho_pad[2:-1] - 0.5 * s_rho[1:])
    vel_l = vel_pad[1:-2] + 0.5 * s_vel[:-1]
    vel_r = vel_pad[2:-1] - 0.5 * s_vel[1:]
    pres_l = np.maximum(1e-9, pres_pad[1:-2] + 0.5 * s_pres[:-1])
    pres_r = np.maximum(1e-9, pres_pad[2:-1] - 0.5 * s_pres[1:])
    return rho_l, vel_l, pres_l, rho_r, vel_r, pres_r


def compute_riemann_flux(
    rho_l: np.ndarray,
    u_l: np.ndarray,
    p_l: np.ndarray,
    rho_r: np.ndarray,
    u_r: np.ndarray,
    p_r: np.ndarray,
    gamma: float,
    scheme: str = "hllc",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate approximate Riemann interface fluxes F_{i+1/2} in vectorized form."""
    sch = scheme.strip().lower()
    if sch not in VALID_FLUX_SCHEMES:
        raise ValueError(f"Unknown flux_scheme '{scheme}'. Choose from: {VALID_FLUX_SCHEMES}")

    gm1 = gamma - 1.0
    e_l = p_l / gm1 + 0.5 * rho_l * (u_l**2)
    e_r = p_r / gm1 + 0.5 * rho_r * (u_r**2)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    f1_l = rho_l * u_l
    f2_l = rho_l * (u_l**2) + p_l
    f3_l = u_l * (e_l + p_l)

    f1_r = rho_r * u_r
    f2_r = rho_r * (u_r**2) + p_r
    f3_r = u_r * (e_r + p_r)

    if sch == "rusanov":
        s_max = np.maximum(np.abs(u_l) + a_l, np.abs(u_r) + a_r)
        u1_l, u2_l, u3_l = rho_l, rho_l * u_l, e_l
        u1_r, u2_r, u3_r = rho_r, rho_r * u_r, e_r
        return (
            0.5 * (f1_l + f1_r) - 0.5 * s_max * (u1_r - u1_l),
            0.5 * (f2_l + f2_r) - 0.5 * s_max * (u2_r - u2_l),
            0.5 * (f3_l + f3_r) - 0.5 * s_max * (u3_r - u3_l),
        )

    if sch == "ausm_plus":
        # Liou (1996) AUSM+ split Mach polynomials (alpha=3/16, beta=1/8)
        a_half = 0.5 * (a_l + a_r)
        m_l = u_l / a_half
        m_r = u_r / a_half

        m_p = np.where(
            np.abs(m_l) >= 1.0,
            0.5 * (m_l + np.abs(m_l)),
            0.25 * ((m_l + 1.0) ** 2) + 0.125 * (((m_l**2) - 1.0) ** 2),
        )
        m_m = np.where(
            np.abs(m_r) >= 1.0,
            0.5 * (m_r - np.abs(m_r)),
            -0.25 * ((m_r - 1.0) ** 2) - 0.125 * (((m_r**2) - 1.0) ** 2),
        )
        p_p = np.where(
            np.abs(m_l) >= 1.0,
            np.where(m_l > 0.0, 1.0, 0.0),
            0.25 * ((m_l + 1.0) ** 2) * (2.0 - m_l) + 0.1875 * m_l * (((m_l**2) - 1.0) ** 2),
        )
        p_m = np.where(
            np.abs(m_r) >= 1.0,
            np.where(m_r < 0.0, 1.0, 0.0),
            0.25 * ((m_r - 1.0) ** 2) * (2.0 + m_r) - 0.1875 * m_r * (((m_r**2) - 1.0) ** 2),
        )

        m_half = m_p + m_m
        p_half = p_p * p_l + p_m * p_r
        mdot_half = a_half * np.where(m_half >= 0.0, m_half * rho_l, m_half * rho_r)

        h_l = (e_l + p_l) / rho_l
        h_r = (e_r + p_r) / rho_r
        psi_u = np.where(m_half >= 0.0, u_l, u_r)
        psi_h = np.where(m_half >= 0.0, h_l, h_r)

        return mdot_half, mdot_half * psi_u + p_half, mdot_half * psi_h

    # ---- HLLC+ (Harten-Lax-van Leer-Contact with pressure-shock carbuncle stabilization) ----
    # Note: At a contact discontinuity, p_L == p_R so shock_sensor == 0 and pure HLLC contact
    # crispness is preserved 100%. At strong stationary normal shocks (|p_R-p_L|/(p_R+p_L) large),
    # light HLL blending prevents discrete shock limit-cycling (Quirk / Kim-Kim-Rho).
    sq_l = np.sqrt(rho_l)
    sq_r = np.sqrt(rho_r)
    h_l = (e_l + p_l) / rho_l
    h_r = (e_r + p_r) / rho_r
    u_roe = (sq_l * u_l + sq_r * u_r) / (sq_l + sq_r)
    h_roe = (sq_l * h_l + sq_r * h_r) / (sq_l + sq_r)
    a_roe = np.sqrt(np.maximum(1e-9, gm1 * (h_roe - 0.5 * (u_roe**2))))

    s_l = np.minimum(u_l - a_l, u_roe - a_roe)
    s_r = np.maximum(u_r + a_r, u_roe + a_roe)

    denom = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    denom_safe = np.where(np.abs(denom) > 1e-12, denom, np.copysign(1e-12, denom + 1e-30))
    s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / denom_safe

    # Star states U*_L and U*_R
    fac_l = rho_l * (s_l - u_l) / np.where(np.abs(s_l - s_star) > 1e-12, s_l - s_star, -1e-12)
    fac_r = rho_r * (s_r - u_r) / np.where(np.abs(s_r - s_star) > 1e-12, s_r - s_star, 1e-12)

    u1s_l = fac_l
    u2s_l = fac_l * s_star
    u3s_l = fac_l * (e_l / rho_l + (s_star - u_l) * (s_star + p_l / (rho_l * (s_l - u_l) - 1e-15)))

    u1s_r = fac_r
    u2s_r = fac_r * s_star
    u3s_r = fac_r * (e_r / rho_r + (s_star - u_r) * (s_star + p_r / (rho_r * (s_r - u_r) + 1e-15)))

    f1_sl = f1_l + s_l * (u1s_l - rho_l)
    f2_sl = f2_l + s_l * (u2s_l - rho_l * u_l)
    f3_sl = f3_l + s_l * (u3s_l - e_l)

    f1_sr = f1_r + s_r * (u1s_r - rho_r)
    f2_sr = f2_r + s_r * (u2s_r - rho_r * u_r)
    f3_sr = f3_r + s_r * (u3s_r - e_r)

    f1 = np.where(
        s_l >= 0.0, f1_l, np.where(s_star >= 0.0, f1_sl, np.where(s_r >= 0.0, f1_sr, f1_r))
    )
    f2 = np.where(
        s_l >= 0.0, f2_l, np.where(s_star >= 0.0, f2_sl, np.where(s_r >= 0.0, f2_sr, f2_r))
    )
    f3 = np.where(
        s_l >= 0.0, f3_l, np.where(s_star >= 0.0, f3_sl, np.where(s_r >= 0.0, f3_sr, f3_r))
    )

    # Subtle HLL shock stabilizer strictly proportional to pressure jump (0 at contact waves)
    p_jump = np.abs(p_r - p_l) / (p_r + p_l + 1e-12)
    alpha_hll = np.where(
        (s_l < 0.0) & (s_r > 0.0) & (p_jump > 0.08), np.minimum(0.35, 0.5 * (p_jump**2)), 0.0
    )
    if np.any(alpha_hll > 0.0):
        sr_sl = np.maximum(1e-9, s_r - s_l)
        u1_l, u2_l, u3_l = rho_l, rho_l * u_l, e_l
        u1_r, u2_r, u3_r = rho_r, rho_r * u_r, e_r
        f1_hll = (s_r * f1_l - s_l * f1_r + s_l * s_r * (u1_r - u1_l)) / sr_sl
        f2_hll = (s_r * f2_l - s_l * f2_r + s_l * s_r * (u2_r - u2_l)) / sr_sl
        f3_hll = (s_r * f3_l - s_l * f3_r + s_l * s_r * (u3_r - u3_l)) / sr_sl
        f1 = (1.0 - alpha_hll) * f1 + alpha_hll * f1_hll
        f2 = (1.0 - alpha_hll) * f2 + alpha_hll * f2_hll
        f3 = (1.0 - alpha_hll) * f3 + alpha_hll * f3_hll

    return f1, f2, f3


def exact_sod_shock_tube(
    x: np.ndarray,
    t_end: float = 0.20,
    gamma: float = 1.4,
    x0: float = 0.5,
    rho_l: float = 1.0,
    u_l: float = 0.0,
    p_l: float = 1.0,
    rho_r: float = 0.125,
    u_r: float = 0.0,
    p_r: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Toro's exact analytical solution for a 1D Riemann shock tube at time t_end."""
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    def _fk(p: float, rho_k: float, p_k: float, a_k: float) -> tuple[float, float]:
        if p > p_k:
            ak = 2.0 / (gp1 * rho_k)
            bk = (gm1 / gp1) * p_k
            q = math.sqrt(ak / (p + bk))
            val = (p - p_k) * q
            deriv = q * (1.0 - (p - p_k) / (2.0 * (p + bk)))
            return val, deriv
        pr = p / p_k
        val = (2.0 * a_k / gm1) * (pr ** (gm1 / (2.0 * gamma)) - 1.0)
        deriv = (1.0 / (rho_k * a_k)) * (pr ** (-gp1 / (2.0 * gamma)))
        return val, deriv

    # Two-rarefaction initial guess + Newton-Raphson for p*
    p_star = max(1e-6, 0.5 * (p_l + p_r))
    for _ in range(40):
        fl, dfl = _fk(p_star, rho_l, p_l, a_l)
        fr, dfr = _fk(p_star, rho_r, p_r, a_r)
        res = fl + fr + (u_r - u_l)
        if abs(res) < 1e-11:
            break
        dp = res / max(1e-12, dfl + dfr)
        p_star = max(1e-8, p_star - dp)

    fl, _ = _fk(p_star, rho_l, p_l, a_l)
    fr, _ = _fk(p_star, rho_r, p_r, a_r)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (fr - fl)

    rho_out = np.empty_like(x, dtype=float)
    u_out = np.empty_like(x, dtype=float)
    p_out = np.empty_like(x, dtype=float)

    for idx, x_val in enumerate(x):
        s = (float(x_val) - x0) / max(1e-12, t_end)
        if s <= u_star:
            # Left of contact discontinuity
            if p_star > p_l:
                s_l = u_l - a_l * math.sqrt(
                    (gp1 / (2.0 * gamma)) * (p_star / p_l) + gm1 / (2.0 * gamma)
                )
                if s <= s_l:
                    rho_out[idx], u_out[idx], p_out[idx] = rho_l, u_l, p_l
                else:
                    pr = p_star / p_l
                    rho_sl = rho_l * ((pr + gm1 / gp1) / ((gm1 / gp1) * pr + 1.0))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_sl, u_star, p_star
            else:
                sh_l = u_l - a_l
                a_sl = a_l * ((p_star / p_l) ** (gm1 / (2.0 * gamma)))
                st_l = u_star - a_sl
                if s <= sh_l:
                    rho_out[idx], u_out[idx], p_out[idx] = rho_l, u_l, p_l
                elif s >= st_l:
                    rho_sl = rho_l * ((p_star / p_l) ** (1.0 / gamma))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_sl, u_star, p_star
                else:
                    u_fan = (2.0 / gp1) * (a_l + 0.5 * gm1 * u_l + s)
                    a_fan = (2.0 / gp1) * (a_l + 0.5 * gm1 * (u_l - s))
                    rho_fan = rho_l * ((a_fan / a_l) ** (2.0 / gm1))
                    p_fan = p_l * ((a_fan / a_l) ** (2.0 * gamma / gm1))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_fan, u_fan, p_fan
        else:
            # Right of contact discontinuity
            if p_star > p_r:
                s_r = u_r + a_r * math.sqrt(
                    (gp1 / (2.0 * gamma)) * (p_star / p_r) + gm1 / (2.0 * gamma)
                )
                if s >= s_r:
                    rho_out[idx], u_out[idx], p_out[idx] = rho_r, u_r, p_r
                else:
                    pr = p_star / p_r
                    rho_sr = rho_r * ((pr + gm1 / gp1) / ((gm1 / gp1) * pr + 1.0))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_sr, u_star, p_star
            else:
                sh_r = u_r + a_r
                a_sr = a_r * ((p_star / p_r) ** (gm1 / (2.0 * gamma)))
                st_r = u_star + a_sr
                if s >= sh_r:
                    rho_out[idx], u_out[idx], p_out[idx] = rho_r, u_r, p_r
                elif s <= st_r:
                    rho_sr = rho_r * ((p_star / p_r) ** (1.0 / gamma))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_sr, u_star, p_star
                else:
                    u_fan = (2.0 / gp1) * (-a_r + 0.5 * gm1 * u_r + s)
                    a_fan = (2.0 / gp1) * (a_r - 0.5 * gm1 * (u_r - s))
                    rho_fan = rho_r * ((a_fan / a_r) ** (2.0 / gm1))
                    p_fan = p_r * ((a_fan / a_r) ** (2.0 * gamma / gm1))
                    rho_out[idx], u_out[idx], p_out[idx] = rho_fan, u_fan, p_fan

    mach_out = u_out / np.sqrt(gamma * p_out / rho_out)
    return rho_out, u_out, p_out, mach_out


def solve_sod_shock_tube(
    nx: int = 200,
    flux_scheme: str = "hllc",
    limiter: str = "van_leer",
    cfl: float = 0.45,
    t_end: float = 0.20,
    gamma: float = 1.4,
) -> SodShockTubeResult:
    """Solve NASA/Toro 1D Sod Shock Tube numerically via MUSCL-TVD + SSP-RK2."""
    t_start = time.perf_counter()
    dx = 1.0 / nx
    x = (np.arange(nx, dtype=float) + 0.5) * dx
    gm1 = gamma - 1.0

    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(nx, dtype=float)
    p = np.where(x < 0.5, 1.0, 0.1)

    u1 = rho.copy()
    u2 = rho * u
    u3 = p / gm1 + 0.5 * rho * (u**2)

    def _rhs(
        c1: np.ndarray, c2: np.ndarray, c3: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_i = np.maximum(1e-9, c1)
        u_i = c2 / r_i
        p_i = np.maximum(1e-9, gm1 * (c3 - 0.5 * r_i * (u_i**2)))

        r_pad = np.pad(r_i, (2, 2), mode="edge")
        u_pad = np.pad(u_i, (2, 2), mode="edge")
        p_pad = np.pad(p_i, (2, 2), mode="edge")
        rl, ul, pl, rr, ur, pr = reconstruct_muscl_interfaces(r_pad, u_pad, p_pad, limiter=limiter)
        f1, f2, f3 = compute_riemann_flux(rl, ul, pl, rr, ur, pr, gamma, scheme=flux_scheme)
        return (
            -(f1[1:] - f1[:-1]) / dx,
            -(f2[1:] - f2[:-1]) / dx,
            -(f3[1:] - f3[:-1]) / dx,
        )

    t = 0.0
    while t < t_end - 1e-12:
        r_cur = np.maximum(1e-9, u1)
        u_cur = u2 / r_cur
        p_cur = np.maximum(1e-9, gm1 * (u3 - 0.5 * r_cur * (u_cur**2)))
        a_cur = np.sqrt(gamma * p_cur / r_cur)
        dt = min(t_end - t, float(cfl * dx / np.max(np.abs(u_cur) + a_cur)))

        k1_1, k1_2, k1_3 = _rhs(u1, u2, u3)
        u1_s = u1 + dt * k1_1
        u2_s = u2 + dt * k1_2
        u3_s = u3 + dt * k1_3

        k2_1, k2_2, k2_3 = _rhs(u1_s, u2_s, u3_s)
        u1 = 0.5 * (u1 + u1_s + dt * k2_1)
        u2 = 0.5 * (u2 + u2_s + dt * k2_2)
        u3 = 0.5 * (u3 + u3_s + dt * k2_3)
        t += dt

    rho_num = np.maximum(1e-9, u1)
    u_num = u2 / rho_num
    p_num = np.maximum(1e-9, gm1 * (u3 - 0.5 * rho_num * (u_num**2)))
    mach_num = u_num / np.sqrt(gamma * p_num / rho_num)

    rho_ex, u_ex, p_ex, mach_ex = exact_sod_shock_tube(x, t_end=t_end, gamma=gamma)
    solve_ms = (time.perf_counter() - t_start) * 1000.0

    return SodShockTubeResult(
        flux_scheme=flux_scheme.lower(),
        limiter=limiter.lower(),
        nx=nx,
        t_end=t_end,
        gamma=gamma,
        x=x,
        rho_exact=rho_ex,
        u_exact=u_ex,
        p_exact=p_ex,
        mach_exact=mach_ex,
        rho_num=rho_num,
        u_num=u_num,
        p_num=p_num,
        mach_num=mach_num,
        rho_l1_error=round(float(np.mean(np.abs(rho_num - rho_ex))), 5),
        rho_l2_error=round(float(np.sqrt(np.mean((rho_num - rho_ex) ** 2))), 5),
        u_l2_error=round(float(np.sqrt(np.mean((u_num - u_ex) ** 2))), 5),
        p_l2_error=round(float(np.sqrt(np.mean((p_num - p_ex) ** 2))), 5),
        solve_ms=round(solve_ms, 2),
    )


def _load_cpp_fvm_library() -> ctypes.CDLL | None:
    """Locate optional compiled C++20 Finite-Volume shared library for zero-copy acceleration."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "cpp" / "nozzle_fvm_core.dll",
        root / "cpp" / "libnozzle_fvm_core.so",
        root / "cpp" / "build" / "nozzle_fvm_core.dll",
        root / "cpp" / "build" / "libnozzle_fvm_core.so",
    ]
    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                lib.solve_quasi1d_fvm_c.argtypes = [
                    ctypes.c_int,  # n
                    ctypes.c_double,  # dx_nd
                    ctypes.c_double,  # gamma
                    ctypes.c_double,  # pb_nd
                    ctypes.c_int,  # flux_id (0=hllc, 1=ausm_plus, 2=rusanov)
                    ctypes.c_int,  # limiter_id (0..4)
                    ctypes.c_double,  # cfl
                    ctypes.c_int,  # max_iter
                    ctypes.c_double,  # tol
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # area_ratio
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out_rho
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out_u
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out_p
                    ctypes.POINTER(ctypes.c_int),  # out_iters
                    ctypes.POINTER(ctypes.c_int),  # out_converged
                    ctypes.POINTER(ctypes.c_double),  # out_residual
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out_res_hist
                    ctypes.c_int,  # max_hist_pts
                    ctypes.POINTER(ctypes.c_int),  # out_hist_len
                ]
                lib.solve_quasi1d_fvm_c.restype = ctypes.c_int
                return lib
            except OSError:
                continue
    return None


def solve_quasi1d_fvm(
    geometry: NozzleGeometry,
    propellant_key: str = "METHALOX",
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    flux_scheme: str = "hllc",
    limiter: str = "van_leer",
    cfl: float = 0.50,
    max_iter: int = 12000,
    tol: float = 1e-6,
    backend: str = "auto",
) -> NozzleFVMResult:
    """Solve Quasi-1D Rocket Nozzle flow with Finite-Volume Riemann Fluxes + MUSCL-TVD + SSP-RK2."""
    t_start = time.perf_counter()
    flux_key = flux_scheme.strip().lower()
    lim_key = limiter.strip().lower()
    if flux_key not in VALID_FLUX_SCHEMES:
        raise ValueError(f"Unknown flux_scheme '{flux_scheme}'. Choose from: {VALID_FLUX_SCHEMES}")
    if lim_key not in VALID_LIMITERS:
        raise ValueError(f"Unknown limiter '{limiter}'. Choose from: {VALID_LIMITERS}")

    _, _, tc_k, gamma, mw = compute_chamber_thermochemistry(
        propellant_key, chamber_pressure_mpa=chamber_pressure_mpa
    )
    r_spec = 8314.4626 / mw
    pc_pa = chamber_pressure_mpa * 1.0e6
    ambient_pa = max(0.01, ambient_pressure_kpa * 1.0e3)
    rho_c = pc_pa / (r_spec * tc_k)
    u_ref = math.sqrt(r_spec * tc_k)

    x = geometry.x_m
    n = len(x)
    dx_nd = float(x[1] - x[0]) / geometry.throat_radius_m
    area = np.ascontiguousarray(geometry.area_ratio, dtype=float)
    throat_idx = int(np.argmin(area))
    gm1 = gamma - 1.0
    pb_nd = ambient_pa / pc_pa

    # Exact Rankine-Hugoniot truth (determines subsonic vs supersonic outflow regime)
    ana_mach, ana_pres, _, ana_shock, ana_shock_idx, _, _ = exact_quasi1d_shock_solution(
        area, throat_idx, gamma, pc_pa, tc_k, ambient_pa
    )
    m_e_sup = mach_from_area_ratio(float(area[-1]), gamma, supersonic=True)
    p_e_sup = pc_pa * ((1.0 + 0.5 * gm1 * (m_e_sup**2)) ** (-gamma / gm1))
    p_e_shocked = p_e_sup * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (m_e_sup**2 - 1.0))
    subsonic_outflow = ambient_pa > p_e_shocked

    residual_history: list[float] = []
    used_backend = "python_numpy"
    converged = False
    iters_run = 0
    final_residual = 1.0

    backend_choice = backend.strip().lower()
    cpp_lib = _load_cpp_fvm_library() if backend_choice in ("auto", "cpp") else None
    if backend_choice == "cpp" and cpp_lib is None:
        raise RuntimeError("C++ shared library nozzle_fvm_core not compiled yet.")

    if cpp_lib is not None:
        flux_map = {"hllc": 0, "ausm_plus": 1, "rusanov": 2}
        lim_map = {"first_order": 0, "minmod": 1, "van_leer": 2, "superbee": 3, "mc": 4}
        out_rho = np.zeros(n, dtype=np.float64)
        out_u = np.zeros(n, dtype=np.float64)
        out_p = np.zeros(n, dtype=np.float64)
        max_hist = 400
        out_hist = np.zeros(max_hist, dtype=np.float64)
        c_iters = ctypes.c_int(0)
        c_conv = ctypes.c_int(0)
        c_res = ctypes.c_double(1.0)
        c_hist_len = ctypes.c_int(0)
        rc = cpp_lib.solve_quasi1d_fvm_c(
            n,
            dx_nd,
            gamma,
            pb_nd if subsonic_outflow else -1.0,
            flux_map[flux_key],
            lim_map[lim_key],
            cfl,
            max_iter,
            tol,
            area,
            out_rho,
            out_u,
            out_p,
            ctypes.byref(c_iters),
            ctypes.byref(c_conv),
            ctypes.byref(c_res),
            out_hist,
            max_hist,
            ctypes.byref(c_hist_len),
        )
        if rc == 0:
            rho_nd, u_nd, p_nd = out_rho, out_u, out_p
            iters_run = int(c_iters.value)
            converged = bool(c_conv.value)
            final_residual = float(c_res.value)
            h_len = int(c_hist_len.value)
            residual_history = [round(float(val), 8) for val in out_hist[:h_len]]
            used_backend = "cpp20_native_dll"

    if used_backend == "python_numpy":
        # Interface areas A_{i-1/2} and A_{i+1/2} arranged for exact freestream well-balancing
        a_int = np.empty(n + 1, dtype=float)
        a_int[1:n] = 0.5 * (area[:-1] + area[1:])
        a_int[0] = area[0]
        a_int[n] = max(1.0, area[-1] + 0.5 * (area[-1] - area[-2]))
        da_cell = a_int[1:] - a_int[:-1]

        # Initialize smoothly toward expected regime
        m_init = np.array(
            [
                mach_from_area_ratio(float(area[i]), gamma, supersonic=(i > throat_idx))
                for i in range(n)
            ],
            dtype=float,
        )
        if subsonic_outflow:
            blend = np.linspace(0.0, 1.0, n - throat_idx)
            m_init[throat_idx:] = np.maximum(0.25, m_init[throat_idx:] * (1.0 - 0.65 * blend))

        t_init = 1.0 / (1.0 + 0.5 * gm1 * (m_init**2))
        p_init = t_init ** (gamma / gm1)
        rho_init = p_init / t_init
        u_init = m_init * np.sqrt(gamma * t_init)

        u1 = rho_init.copy()
        u2 = rho_init * u_init
        u3 = p_init / gm1 + 0.5 * rho_init * (u_init**2)

        h0_nd = gamma / gm1
        a_quad = 1.0 / gm1 + 2.0 / (gm1**2)

        def _apply_ghost_cells(
            r_i: np.ndarray, u_i: np.ndarray, p_i: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Characteristic subsonic Riemann inflow fixing stagnation P0=1, T0=1
            a0 = math.sqrt(gamma * float(p_i[0]) / float(r_i[0]))
            r_minus = float(u_i[0]) - (2.0 * a0 / gm1)
            b_quad = (2.0 * r_minus) / gm1
            c_quad = 0.5 * (r_minus**2) - h0_nd
            disc = max(0.0, b_quad * b_quad - 4.0 * a_quad * c_quad)
            a_in = max(0.2, (-b_quad + math.sqrt(disc)) / (2.0 * a_quad))
            u_in = max(0.005 * a_in, min(0.92 * a_in, r_minus + (2.0 * a_in / gm1)))
            t_in = min(1.0, max(0.4, (a_in * a_in) / gamma))
            p_in = t_in ** (gamma / gm1)
            r_in = p_in / t_in

            # Outflow: fixed pressure Riemann characteristic if subsonic, else extrapolate
            if subsonic_outflow:
                a_e = math.sqrt(gamma * float(p_i[-1]) / float(r_i[-1]))
                p_out = max(1e-5, pb_nd)
                s_e = float(p_i[-1]) / (float(r_i[-1]) ** gamma)
                r_out = max(1e-6, (p_out / s_e) ** (1.0 / gamma))
                u_out = max(
                    0.02, float(u_i[-1]) + (float(p_i[-1]) - p_out) / (float(r_i[-1]) * a_e)
                )
            else:
                r_out, u_out, p_out = float(r_i[-1]), float(u_i[-1]), float(p_i[-1])

            r_pad = np.empty(n + 4, dtype=float)
            u_pad = np.empty(n + 4, dtype=float)
            p_pad = np.empty(n + 4, dtype=float)
            r_pad[0], r_pad[1], r_pad[2:-2], r_pad[-2], r_pad[-1] = r_in, r_in, r_i, r_out, r_out
            u_pad[0], u_pad[1], u_pad[2:-2], u_pad[-2], u_pad[-1] = u_in, u_in, u_i, u_out, u_out
            p_pad[0], p_pad[1], p_pad[2:-2], p_pad[-2], p_pad[-1] = p_in, p_in, p_i, p_out, p_out
            return r_pad, u_pad, p_pad

        def _rhs_nozzle(
            c1: np.ndarray, c2: np.ndarray, c3: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r_i = np.maximum(1e-9, c1)
            u_i = c2 / r_i
            p_i = np.maximum(1e-9, gm1 * (c3 - 0.5 * r_i * (u_i**2)))
            r_pad, u_pad, p_pad = _apply_ghost_cells(r_i, u_i, p_i)
            # Suppress steady-state MUSCL sawtooth limit cycling strictly inside shocks
            dp_switch = np.abs(p_pad[2:] - 2.0 * p_pad[1:-1] + p_pad[:-2]) / np.maximum(
                p_pad[2:] + 2.0 * p_pad[1:-1] + p_pad[:-2], 1e-12
            )
            shock_damp = np.clip(1.0 - 8.0 * np.maximum(0.0, dp_switch - 0.03), 0.0, 1.0)
            rl, ul, pl, rr, ur, pr = reconstruct_muscl_interfaces(
                r_pad, u_pad, p_pad, limiter=lim_key, shock_damping=shock_damp
            )
            f1, f2, f3 = compute_riemann_flux(rl, ul, pl, rr, ur, pr, gamma, scheme=flux_key)
            inv_vol = 1.0 / (area * dx_nd)
            du1 = -(f1[1:] * a_int[1:] - f1[:-1] * a_int[:-1]) * inv_vol
            du2 = -(f2[1:] * a_int[1:] - f2[:-1] * a_int[:-1]) * inv_vol + p_i * da_cell * inv_vol
            du3 = -(f3[1:] * a_int[1:] - f3[:-1] * a_int[:-1]) * inv_vol
            return du1, du2, du3

        for it in range(1, max_iter + 1):
            r_cur = np.maximum(1e-9, u1)
            u_cur = u2 / r_cur
            p_cur = np.maximum(1e-9, gm1 * (u3 - 0.5 * r_cur * (u_cur**2)))
            a_cur = np.sqrt(gamma * p_cur / r_cur)
            dt = float(cfl * dx_nd / np.max(np.abs(u_cur) + a_cur))

            k1_1, k1_2, k1_3 = _rhs_nozzle(u1, u2, u3)
            u1_s = u1 + dt * k1_1
            u2_s = u2 + dt * k1_2
            u3_s = u3 + dt * k1_3

            k2_1, k2_2, k2_3 = _rhs_nozzle(u1_s, u2_s, u3_s)
            u1_n = 0.5 * (u1 + u1_s + dt * k2_1)
            u2_n = 0.5 * (u2 + u2_s + dt * k2_2)
            u3_n = 0.5 * (u3 + u3_s + dt * k2_3)

            diff = float(np.max(np.abs(u2_n - u2) / np.maximum(np.abs(u2), 1e-4)))
            u1, u2, u3 = u1_n, u2_n, u3_n
            iters_run = it
            final_residual = diff
            if it == 1 or it % 50 == 0 or diff < tol:
                residual_history.append(round(diff, 8))
            if it > 150 and diff < tol:
                converged = True
                break

        rho_nd = np.maximum(1e-9, u1)
        u_nd = u2 / rho_nd
        p_nd = np.maximum(1e-9, gm1 * (u3 - 0.5 * rho_nd * (u_nd**2)))

    solve_ms = (time.perf_counter() - t_start) * 1000.0
    t_nd = p_nd / rho_nd
    a_nd = np.sqrt(gamma * t_nd)
    mach = u_nd / a_nd

    pressure_kpa = (p_nd * pc_pa) / 1.0e3
    temperature_k = t_nd * tc_k
    density_kg_m3 = rho_nd * rho_c
    velocity_m_s = u_nd * u_ref
    at_m2 = math.pi * (geometry.throat_radius_m**2)
    mdot_x = density_kg_m3 * (area * at_m2) * velocity_m_s

    # Shock detection directly from steep compressive Mach drop downstream of throat
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
        if mach[max(0, shock_cell - 2)] > 1.05 and dmach[shock_cell] < -0.10:
            shock_detected = True
            shock_x_m = round(float(x[shock_cell]), 5)
            shock_ar = round(float(area[shock_cell]), 3)
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

    mask = np.ones(n, dtype=bool)
    if ana_shock and ana_shock_idx is not None:
        lo = max(0, ana_shock_idx - 3)
        hi = min(n, ana_shock_idx + 4)
        mask[lo:hi] = False

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

    return NozzleFVMResult(
        propellant=propellant_key.upper(),
        flux_scheme=flux_key,
        limiter=lim_key,
        solver_backend=used_backend,
        solve_ms=round(solve_ms, 2),
        chamber_pressure_mpa=round(chamber_pressure_mpa, 3),
        ambient_pressure_kpa=round(ambient_pressure_kpa, 3),
        gamma=round(gamma, 4),
        tc_k=round(tc_k, 2),
        r_specific_j_kg_k=round(r_spec, 2),
        x_m=x,
        area_ratio=area,
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


def run_grid_convergence_study(
    propellant_key: str = "METHALOX",
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    expansion_ratio: float = 12.0,
    throat_radius_m: float = 0.12,
    flux_scheme: str = "hllc",
    limiter: str = "van_leer",
    grid_sizes: tuple[int, int, int] = (81, 161, 321),
    backend: str = "auto",
) -> GridConvergenceResult:
    """Execute formal 3-grid ASME V&V 20 / Roache GCI mesh refinement study.

    Grid sizes should double between levels (`(N1, N2, N3)` -> refinement ratio `r = 2.0`),
    measuring true spatial order of convergence `p`, Richardson extrapolated exact value,
    and numerical uncertainty band `GCI_fine %`.
    """
    from nozzle_cfd.geometry import build_rao_bell_nozzle

    n_coarse, n_med, n_fine = grid_sizes
    r = (n_fine - 1) / max(1.0, float(n_med - 1))

    results: list[NozzleFVMResult] = []
    for nx in grid_sizes:
        geom = build_rao_bell_nozzle(
            throat_radius_m=throat_radius_m,
            contraction_ratio=3.2,
            expansion_ratio=expansion_ratio,
            n_points=nx,
        )
        res = solve_quasi1d_fvm(
            geom,
            propellant_key=propellant_key,
            chamber_pressure_mpa=chamber_pressure_mpa,
            ambient_pressure_kpa=ambient_pressure_kpa,
            flux_scheme=flux_scheme,
            limiter=limiter,
            max_iter=15000,
            tol=1e-7,
            backend=backend,
        )
        results.append(res)

    f3, f2, f1 = [float(item.mach[-1]) for item in results]
    e3, e2, e1 = [float(item.mach_l2_error_pct) for item in results]
    p3, p2, p1 = [float(item.pressure_l2_error_pct) for item in results]

    # Direct spatial convergence order p_obs from exact L2 error ratio across h_coarse -> h_fine
    if e1 > 1e-9 and e3 > e1:
        p_obs = max(0.8, min(3.0, math.log(e3 / e1) / max(1e-6, 2.0 * math.log(r))))
    else:
        p_obs = 2.0

    # Local exit-Mach convergence order for Richardson & asymptotic check
    dm_32 = abs(f3 - f2)
    dm_21 = abs(f2 - f1)
    p_mach = (
        max(0.8, min(3.0, math.log(dm_32 / dm_21) / max(1e-6, math.log(r))))
        if (dm_21 > 1e-9 and dm_32 > 1e-9)
        else p_obs
    )

    denom_mach = max(1e-6, (r**p_mach) - 1.0)
    fs = 1.25  # Roache conservative safety factor for 3+ grids
    rel_21 = abs(f1 - f2) / max(abs(f1), 1e-9)
    rel_32 = abs(f2 - f3) / max(abs(f2), 1e-9)
    gci_fine = fs * rel_21 / denom_mach * 100.0
    gci_coarse = fs * rel_32 / denom_mach * 100.0
    asym_ratio = gci_coarse / max(1e-6, (r**p_mach) * gci_fine)
    rich_mach = f1 + (f1 - f2) / denom_mach

    return GridConvergenceResult(
        propellant=propellant_key.upper(),
        flux_scheme=flux_scheme.lower(),
        limiter=limiter.lower(),
        grid_sizes=grid_sizes,
        refinement_ratio=round(r, 3),
        exit_mach_values=(round(f3, 5), round(f2, 5), round(f1, 5)),
        mach_l2_errors_pct=(round(e3, 4), round(e2, 4), round(e1, 4)),
        pressure_l2_errors_pct=(round(p3, 4), round(p2, 4), round(p1, 4)),
        observed_order_error=round(p_obs, 3),
        richardson_extrapolated_exit_mach=round(rich_mach, 5),
        gci_fine_pct=round(gci_fine, 4),
        gci_coarse_pct=round(gci_coarse, 4),
        asymptotic_range_ratio=round(asym_ratio, 3),
        is_grid_converged=e1 < e3 and gci_fine < 1.0,
    )
