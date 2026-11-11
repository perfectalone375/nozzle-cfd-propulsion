"""True 2D Axisymmetric Curvilinear Finite-Volume Euler & Navier-Stokes Solver.

Why Step 2 upgrades beyond 1D and analytical shock approximations:
    1. Solves the **2D Axisymmetric Compressible Euler & Navier-Stokes Equations** on a
       body-fitted curvilinear multi-block mesh (`Block 1`: inside the converging-diverging
       nozzle; `Block 2`: connected external plume box with 1-to-1 conservative face matching).
    2. Uses the **1-radian wedge finite-volume formulation**, guaranteeing exact free-stream
       preservation (Geometric Conservation Law) on curved walls without spurious radial drift.
    3. Employs a **Rotated 2D HLLC+ Riemann Solver** + **MUSCL-TVD Van Leer reconstruction**
       + **Sutherland viscosity & Prandtl/Baldwin-Lomax turbulent mixing length** + **SSP-RK2**
       time integration accelerated across all CPU cores via a **C++20/OpenMP DLL**.
    4. Directly resolves phenomena that 1D CFD misses:
       - **Curved 2D transonic sonic line** (`M = 1` bends upstream at tight throats such as
         the JPL `rc/rt = 0.625` 45-15 conical nozzle & Rao bell contours).
       - **Inviscid discharge coefficient** `Cd < 1` compared against Hall (1962) / Kliegel-Levine.
       - **Viscous no-slip boundary layers & shock-induced flow separation (`u < 0`)**.
       - **Plume oblique barrel shocks, centerline Mach discs, triple points, & Schlieren**.
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.propellants import compute_chamber_thermochemistry, mach_from_area_ratio

VALID_VISCOUS_MODES = ("euler", "laminar", "turbulent")
VALID_LIMITERS_2D = ("first_order", "minmod", "van_leer", "mc")


def compute_hall_discharge_coefficient(gamma: float, rc_rt_ratio: float) -> float:
    """Hall (1962) / ISO-9300 asymptotic series for inviscid 2D axisymmetric Cd.

    Accurate for moderate/large throat radii of curvature (`rc/rt >= 1.5`). For very tight
    throats (`rc/rt < 1.0`, e.g., JPL 45-15 `rc/rt = 0.625`), series terms diverge slowly,
    highlighting why full 2D CFD is necessary.
    """
    R = max(0.35, float(rc_rt_ratio))
    gp1 = gamma + 1.0
    t1 = gp1 / (96.0 * R * R)
    t2 = (gp1 * (8.0 * gamma - 27.0)) / (2304.0 * (R**3))
    t3 = (gp1 * (754.0 * gamma * gamma - 757.0 * gamma + 3633.0)) / (276480.0 * (R**4))
    return float(1.0 - t1 + t2 - t3)


def build_conical_nozzle_geometry(
    throat_radius_m: float = 0.05,
    contraction_ratio: float = 3.0,
    expansion_ratio: float = 4.0,
    conv_half_angle_deg: float = 45.0,
    div_half_angle_deg: float = 15.0,
    throat_rc_ratio: float = 0.625,
    n_points: int = 121,
) -> NozzleGeometry:
    """Construct an exact C1-continuous conical nozzle (such as JPL 45-15 rc/rt=0.625)."""
    rt = float(throat_radius_m)
    rc = float(throat_rc_ratio) * rt
    r_in = rt * math.sqrt(contraction_ratio)
    r_ex = rt * math.sqrt(expansion_ratio)

    beta = math.radians(conv_half_angle_deg)
    alpha = math.radians(div_half_angle_deg)

    # Tangent transition points along circular throat arc centered at (x_throat, rt + rc)
    r_arc_c = rt + rc * (1.0 - math.cos(beta))
    dx_arc_c = rc * math.sin(beta)

    # C1-continuous circular fillet connecting horizontal chamber wall r_in into converging cone
    denom_c = max(1e-6, 1.0 - math.cos(beta))
    r_conv_fillet = min(0.8 * rt, 0.7 * max(0.0, r_in - r_arc_c) / denom_c)
    r_fillet_end = r_in - r_conv_fillet * (1.0 - math.cos(beta))
    dx_fillet = r_conv_fillet * math.sin(beta)
    dx_conv_straight = max(0.0, (r_fillet_end - r_arc_c) / math.tan(beta))
    len_lead = 0.25 * r_in

    r_arc_d = rt + rc * (1.0 - math.cos(alpha))
    dx_arc_d = rc * math.sin(alpha)
    dx_div_straight = max(0.0, (r_ex - r_arc_d) / math.tan(alpha))

    x_fillet_end = len_lead + dx_fillet
    x_arc_start = x_fillet_end + dx_conv_straight
    x_throat = x_arc_start + dx_arc_c
    x_arc_end = x_throat + dx_arc_d
    x_total = x_arc_end + dx_div_straight

    x = np.linspace(0.0, x_total, n_points)
    r = np.empty_like(x)

    tan_b = math.tan(beta)
    tan_a = math.tan(alpha)

    for k, xv in enumerate(x):
        if xv <= len_lead:
            r[k] = r_in
        elif xv < x_fillet_end:
            dx = min(r_conv_fillet * 0.9999, xv - len_lead)
            r[k] = r_in - r_conv_fillet + math.sqrt(max(0.0, r_conv_fillet**2 - dx * dx))
        elif xv < x_arc_start:
            r[k] = r_fillet_end - (xv - x_fillet_end) * tan_b
        elif xv <= x_arc_end:
            dx = np.clip(xv - x_throat, -rc * 0.9999, rc * 0.9999)
            r[k] = rt + rc - math.sqrt(max(0.0, rc * rc - dx * dx))
        else:
            r[k] = r_arc_d + (xv - x_arc_end) * tan_a

    area_m2 = math.pi * (r**2)
    area_ratio = (r / rt) ** 2
    d_area_dx_m = np.gradient(area_m2, x, edge_order=2)
    wall_slope_rad = np.arctan(np.gradient(r, x, edge_order=2))
    div_eff = 0.5 * (1.0 + math.cos(alpha))
    return NozzleGeometry(
        throat_radius_m=rt,
        contraction_ratio=float(contraction_ratio),
        expansion_ratio=float(expansion_ratio),
        bell_percent=100.0,
        theta_n_deg=float(conv_half_angle_deg),
        theta_e_deg=float(div_half_angle_deg),
        divergence_efficiency=float(div_eff),
        chamber_length_m=float(len_lead),
        converging_length_m=float(x_throat - len_lead),
        diverging_length_m=float(x_total - x_throat),
        total_length_m=float(x_total),
        throat_x_m=float(x_throat),
        x_m=x,
        radius_m=r,
        area_m2=area_m2,
        area_ratio=area_ratio,
        d_area_dx_m=d_area_dx_m,
        wall_slope_rad=wall_slope_rad,
    )


@dataclass(frozen=True)
class AxisymmetricMesh2D:
    """Two-block structured curvilinear axisymmetric mesh (Block 1: nozzle, Block 2: plume)."""

    nx1: int
    nr1: int
    x1_v: np.ndarray  # shape (nx1+1, nr1+1)
    r1_v: np.ndarray  # shape (nx1+1, nr1+1)
    x1_c: np.ndarray  # shape (nx1, nr1)
    r1_c: np.ndarray  # shape (nx1, nr1)
    nx2: int
    nr2: int
    x2_v: np.ndarray | None  # shape (nx2+1, nr2+1)
    r2_v: np.ndarray | None
    x2_c: np.ndarray | None  # shape (nx2, nr2)
    r2_c: np.ndarray | None
    wall_x_m: np.ndarray
    wall_r_m: np.ndarray


def build_axisymmetric_nozzle_plume_mesh(
    geom: NozzleGeometry,
    nx1: int = 84,
    nr1: int = 28,
    include_plume: bool = True,
    nx2: int = 96,
    plume_length_radii: float = 12.0,
    plume_max_radius_ratio: float = 2.4,
    wall_clustering: float = 1.5,
) -> AxisymmetricMesh2D:
    """Build curvilinear Block 1 nozzle mesh + optional 1-to-1 matched Block 2 plume."""
    # Blend uniform + throat-clustered axial spacing smoothly
    s = np.linspace(0.0, 1.0, nx1 + 1)
    s_th = geom.throat_x_m / geom.total_length_m
    alpha_cluster = 2.6
    u_th = (
        s_th + (1.0 / alpha_cluster) * np.arcsinh(np.sinh(alpha_cluster * s_th) * (s - 0.5) * 2.0)
        if 0.05 < s_th < 0.95
        else s
    )
    # Clean monotonic blend centered tightly around throat
    w_th = np.exp(-(((s - s_th) / 0.22) ** 2))
    # Integrate smooth positive density to cluster 2x finer around throat
    ds_density = 1.0 + 1.4 * w_th
    x_raw = np.zeros_like(s)
    x_raw[1:] = np.cumsum(0.5 * (ds_density[:-1] + ds_density[1:]) * np.diff(s))
    x_line = geom.x_m[0] + (x_raw / x_raw[-1]) * geom.total_length_m
    _ = u_th

    wall_r = np.interp(x_line, geom.x_m, geom.radius_m)

    # Radial wall clustering (tanh toward eta=1 nozzle wall to resolve viscous BL & lip shear layer)
    u_r = np.linspace(0.0, 1.0, nr1 + 1)
    if wall_clustering > 0.05:
        eta = 1.0 - np.tanh(wall_clustering * (1.0 - u_r)) / math.tanh(wall_clustering)
    else:
        eta = u_r

    x1_v = np.repeat(x_line[:, None], nr1 + 1, axis=1)
    r1_v = wall_r[:, None] * eta[None, :]

    x1_c = 0.25 * (x1_v[:-1, :-1] + x1_v[1:, :-1] + x1_v[1:, 1:] + x1_v[:-1, 1:])
    r1_c = 0.25 * (r1_v[:-1, :-1] + r1_v[1:, :-1] + r1_v[1:, 1:] + r1_v[:-1, 1:])

    if not include_plume or nx2 <= 0:
        return AxisymmetricMesh2D(
            nx1=nx1,
            nr1=nr1,
            x1_v=np.ascontiguousarray(x1_v, dtype=np.float64),
            r1_v=np.ascontiguousarray(r1_v, dtype=np.float64),
            x1_c=np.ascontiguousarray(x1_c, dtype=np.float64),
            r1_c=np.ascontiguousarray(r1_c, dtype=np.float64),
            nx2=0,
            nr2=0,
            x2_v=None,
            r2_v=None,
            x2_c=None,
            r2_c=None,
            wall_x_m=x_line,
            wall_r_m=wall_r,
        )

    # Block 2: Exact 1-to-1 radial vertices at j = 0..nr1, plus ambient upper expansion
    r_exit = float(wall_r[-1])
    dr_lip = max(1e-5, float(r1_v[-1, -1] - r1_v[-1, -2]))
    r_outer_target = plume_max_radius_ratio * r_exit

    # Compute nr_outer geometrically growing cells outward from r_exit
    r_outer = [r_exit]
    step = dr_lip * 1.15
    while r_outer[-1] < r_outer_target and len(r_outer) < 40:
        nxt = min(r_outer_target, r_outer[-1] + step)
        r_outer.append(nxt)
        step *= 1.16
    if r_outer[-1] < r_outer_target - 1e-6:
        r_outer.append(r_outer_target)

    r2_line = np.concatenate([r1_v[-1, :], np.asarray(r_outer[1:], dtype=np.float64)])
    nr2 = len(r2_line) - 1

    # Axial plume line gently stretched downstream
    t2 = np.linspace(0.0, 1.0, nx2 + 1)
    s2 = 0.65 * t2 + 0.35 * (t2**1.55)
    x2_line = geom.total_length_m + s2 * (plume_length_radii * r_exit)

    x2_v = np.repeat(x2_line[:, None], nr2 + 1, axis=1)
    r2_v = np.repeat(r2_line[None, :], nx2 + 1, axis=0)
    x2_c = 0.25 * (x2_v[:-1, :-1] + x2_v[1:, :-1] + x2_v[1:, 1:] + x2_v[:-1, 1:])
    r2_c = 0.25 * (r2_v[:-1, :-1] + r2_v[1:, :-1] + r2_v[1:, 1:] + r2_v[:-1, 1:])

    return AxisymmetricMesh2D(
        nx1=nx1,
        nr1=nr1,
        x1_v=np.ascontiguousarray(x1_v, dtype=np.float64),
        r1_v=np.ascontiguousarray(r1_v, dtype=np.float64),
        x1_c=np.ascontiguousarray(x1_c, dtype=np.float64),
        r1_c=np.ascontiguousarray(r1_c, dtype=np.float64),
        nx2=nx2,
        nr2=nr2,
        x2_v=np.ascontiguousarray(x2_v, dtype=np.float64),
        r2_v=np.ascontiguousarray(r2_v, dtype=np.float64),
        x2_c=np.ascontiguousarray(x2_c, dtype=np.float64),
        r2_c=np.ascontiguousarray(r2_c, dtype=np.float64),
        wall_x_m=x_line,
        wall_r_m=wall_r,
    )


@dataclass(frozen=True)
class NavierStokes2DResult:
    """Complete 2D Axisymmetric Curvilinear Euler / Navier-Stokes solution."""

    propellant: str
    viscous_mode: str
    limiter: str
    solver_backend: str
    n_threads: int
    solve_ms: float
    iterations: int
    converged: bool
    gamma: float
    tc_k: float
    r_specific_j_kg_k: float
    chamber_pressure_mpa: float
    ambient_pressure_kpa: float
    x1_v: np.ndarray
    r1_v: np.ndarray
    x1_c: np.ndarray
    r1_c: np.ndarray
    rho_1: np.ndarray
    u_1: np.ndarray
    v_1: np.ndarray
    p_kpa_1: np.ndarray
    T_k_1: np.ndarray
    mach_1: np.ndarray
    schlieren_1: np.ndarray
    x2_v: np.ndarray | None
    r2_v: np.ndarray | None
    x2_c: np.ndarray | None
    r2_c: np.ndarray | None
    rho_2: np.ndarray | None
    u_2: np.ndarray | None
    v_2: np.ndarray | None
    p_kpa_2: np.ndarray | None
    T_k_2: np.ndarray | None
    mach_2: np.ndarray | None
    schlieren_2: np.ndarray | None
    residual_history: np.ndarray
    mass_flow_1d_kg_s: float
    mass_flow_2d_throat_kg_s: float
    mass_flow_2d_exit_kg_s: float
    discharge_coefficient_cd: float
    mass_conservation_error_pct: float
    exit_mach_centerline: float
    exit_mach_wall: float
    sonic_line_x_m: np.ndarray
    sonic_line_r_m: np.ndarray
    sonic_line_offset_mm: float
    separation_detected: bool
    separation_x_m: float | None
    mach_disc_detected: bool
    mach_disc_x_m: float | None
    shock_diamond_wavelength_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "propellant": self.propellant,
            "viscous_mode": self.viscous_mode,
            "limiter": self.limiter,
            "solver_backend": self.solver_backend,
            "n_threads": self.n_threads,
            "solve_ms": round(self.solve_ms, 2),
            "iterations": self.iterations,
            "converged": self.converged,
            "block1_shape": list(self.mach_1.shape),
            "block2_shape": list(self.mach_2.shape) if self.mach_2 is not None else None,
            "mass_flow_1d_kg_s": round(self.mass_flow_1d_kg_s, 3),
            "mass_flow_2d_throat_kg_s": round(self.mass_flow_2d_throat_kg_s, 3),
            "mass_flow_2d_exit_kg_s": round(self.mass_flow_2d_exit_kg_s, 3),
            "discharge_coefficient_cd": round(self.discharge_coefficient_cd, 4),
            "mass_conservation_error_pct": round(self.mass_conservation_error_pct, 4),
            "exit_mach_centerline": round(self.exit_mach_centerline, 4),
            "exit_mach_wall": round(self.exit_mach_wall, 4),
            "sonic_line_offset_mm": round(self.sonic_line_offset_mm, 3),
            "separation_detected": self.separation_detected,
            "separation_x_m": round(self.separation_x_m, 4)
            if self.separation_x_m is not None
            else None,
            "mach_disc_detected": self.mach_disc_detected,
            "mach_disc_x_m": round(self.mach_disc_x_m, 4)
            if self.mach_disc_x_m is not None
            else None,
            "shock_diamond_wavelength_m": round(self.shock_diamond_wavelength_m, 4)
            if self.shock_diamond_wavelength_m is not None
            else None,
        }


_CPP_NS2D_LIB: ctypes.CDLL | None = None
_CPP_NS2D_TRIED = False


def _load_cpp_ns2d_library() -> ctypes.CDLL | None:
    """Locate and bind compiled C++20/OpenMP 2D Axisymmetric Navier-Stokes shared library."""
    global _CPP_NS2D_LIB, _CPP_NS2D_TRIED
    if _CPP_NS2D_TRIED:
        return _CPP_NS2D_LIB
    _CPP_NS2D_TRIED = True

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "cpp" / "nozzle_fvm2d_core.dll",
        repo_root / "cpp" / "libnozzle_fvm2d_core.so",
        repo_root / "cpp" / "libnozzle_fvm2d_core.dylib",
    ]
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)
    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                fn = lib.solve_axisymmetric_ns2d_c
                fn.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    ctypes.c_int,
                    ctypes.c_int,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_double,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_dp,
                    ctypes.c_int,
                    c_ip,
                    c_ip,
                    c_dp,
                    c_dp,
                    c_dp,
                    c_ip,
                ]
                fn.restype = ctypes.c_int
                _CPP_NS2D_LIB = lib
                return lib
            except OSError:
                continue
    return None


def _build_initial_fields(
    mesh: AxisymmetricMesh2D,
    geom: NozzleGeometry,
    gamma: float,
    r_spec: float,
    tc_k: float,
    pc_pa: float,
    pa_pa: float,
    uniform_quiescent: bool = False,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
]:
    """Initialize Block 1 and Block 2 primitive fields cleanly."""
    nx1, nr1 = mesh.nx1, mesh.nr1
    if uniform_quiescent:
        p0 = pa_pa
        t0 = 300.0
        r0 = p0 / (r_spec * t0)
        b1 = (
            np.full((nx1, nr1), r0, dtype=np.float64),
            np.zeros((nx1, nr1), dtype=np.float64),
            np.zeros((nx1, nr1), dtype=np.float64),
            np.full((nx1, nr1), p0, dtype=np.float64),
        )
        if mesh.nx2 > 0:
            nx2, nr2 = mesh.nx2, mesh.nr2
            b2 = (
                np.full((nx2, nr2), r0, dtype=np.float64),
                np.zeros((nx2, nr2), dtype=np.float64),
                np.zeros((nx2, nr2), dtype=np.float64),
                np.full((nx2, nr2), p0, dtype=np.float64),
            )
            return b1, b2
        return b1, None

    gm1 = gamma - 1.0
    x1_ax = mesh.x1_c[:, 0]
    rw_ax = np.interp(x1_ax, geom.x_m, geom.radius_m)
    drw_dx = np.gradient(rw_ax, x1_ax, edge_order=2)
    ar_ax = np.maximum(1.0, (rw_ax / geom.throat_radius_m) ** 2)

    m_ax = np.array(
        [
            mach_from_area_ratio(float(a), gamma, supersonic=(xv >= geom.throat_x_m))
            for xv, a in zip(x1_ax, ar_ax, strict=True)
        ],
        dtype=np.float64,
    )
    fac = 1.0 + 0.5 * gm1 * (m_ax**2)
    t_ax = tc_k / fac
    p_ax = pc_pa * (fac ** (-gamma / gm1))
    rho_ax = p_ax / (r_spec * t_ax)
    vmag_ax = m_ax * np.sqrt(gamma * r_spec * t_ax)

    # Align radial velocity component smoothly with local streamline slope eta * dR/dx
    eta_c = mesh.r1_c / np.maximum(1e-9, rw_ax[:, None])
    slope = eta_c * drw_dx[:, None]
    inv_s = 1.0 / np.sqrt(1.0 + slope * slope)

    rho1 = np.repeat(rho_ax[:, None], nr1, axis=1)
    u1 = vmag_ax[:, None] * inv_s
    v1 = vmag_ax[:, None] * slope * inv_s
    p1 = np.repeat(p_ax[:, None], nr1, axis=1)

    if mesh.nx2 <= 0 or mesh.x2_c is None or mesh.r2_c is None:
        return (
            np.ascontiguousarray(rho1),
            np.ascontiguousarray(u1),
            np.ascontiguousarray(v1),
            np.ascontiguousarray(p1),
        ), None

    nx2, nr2 = mesh.nx2, mesh.nr2
    r_exit = float(rw_ax[-1])
    t_amb = 300.0
    rho_amb = pa_pa / (r_spec * t_amb)
    a_amb = math.sqrt(gamma * r_spec * t_amb)
    u_coflow = 0.04 * a_amb

    rho2 = np.full((nx2, nr2), rho_amb, dtype=np.float64)
    u2 = np.full((nx2, nr2), u_coflow, dtype=np.float64)
    v2 = np.zeros((nx2, nr2), dtype=np.float64)
    p2 = np.full((nx2, nr2), pa_pa, dtype=np.float64)

    # Smooth core jet continuation matching nozzle exit state across r < r_exit
    in_core = mesh.r2_c < (0.98 * r_exit)
    # Linear relaxation toward ambient static pressure in initial seed (CFD resolves shocks)
    p2 = np.where(in_core, np.maximum(0.5 * p1[-1, -1], 0.5 * (p1[-1, 0] + pa_pa)), pa_pa)
    u2 = np.where(in_core, u1[-1, 0], u_coflow)
    rho2 = np.where(in_core, rho1[-1, 0] * (p2 / p1[-1, 0]) ** (1.0 / gamma), rho_amb)

    return (
        np.ascontiguousarray(rho1, dtype=np.float64),
        np.ascontiguousarray(u1, dtype=np.float64),
        np.ascontiguousarray(v1, dtype=np.float64),
        np.ascontiguousarray(p1, dtype=np.float64),
    ), (
        np.ascontiguousarray(rho2, dtype=np.float64),
        np.ascontiguousarray(u2, dtype=np.float64),
        np.ascontiguousarray(v2, dtype=np.float64),
        np.ascontiguousarray(p2, dtype=np.float64),
    )


def _hllc_rotated_flux_vec(
    rho_l: np.ndarray,
    u_l: np.ndarray,
    v_l: np.ndarray,
    p_l: np.ndarray,
    rho_r: np.ndarray,
    u_r: np.ndarray,
    v_r: np.ndarray,
    p_r: np.ndarray,
    nx_f: np.ndarray,
    nr_f: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 2D rotated HLLC+ flux matching C++ kernel bitwise-structurally."""
    gm1 = gamma - 1.0
    un_l = u_l * nx_f + v_l * nr_f
    ut_l = -u_l * nr_f + v_l * nx_f
    un_r = u_r * nx_f + v_r * nr_f
    ut_r = -u_r * nr_f + v_r * nx_f

    rl = np.maximum(1e-8, rho_l)
    rr = np.maximum(1e-8, rho_r)
    pl = np.maximum(1.0, p_l)
    pr = np.maximum(1.0, p_r)

    al = np.sqrt(gamma * pl / rl)
    ar = np.sqrt(gamma * pr / rr)
    e_l = pl / (gm1 * rl) + 0.5 * (u_l * u_l + v_l * v_l)
    e_r = pr / (gm1 * rr) + 0.5 * (u_r * u_r + v_r * v_r)
    h_l = e_l + pl / rl
    h_r = e_r + pr / rr

    sql = np.sqrt(rl)
    sqr = np.sqrt(rr)
    inv_sq = 1.0 / (sql + sqr)
    un_roe = (sql * un_l + sqr * un_r) * inv_sq
    ut_roe = (sql * ut_l + sqr * ut_r) * inv_sq
    h_roe = (sql * h_l + sqr * h_r) * inv_sq
    a_roe = np.sqrt(np.maximum(1e-9, gm1 * (h_roe - 0.5 * (un_roe**2 + ut_roe**2))))

    s_l = np.minimum(un_l - al, un_roe - a_roe)
    s_r = np.maximum(un_r + ar, un_roe + a_roe)

    fl0 = rl * un_l
    fln = rl * un_l * un_l + pl
    flt = rl * un_l * ut_l
    fl3 = (rl * e_l + pl) * un_l

    fr0 = rr * un_r
    frn = rr * un_r * un_r + pr
    frt = rr * un_r * ut_r
    fr3 = (rr * e_r + pr) * un_r

    denom = rl * (s_l - un_l) - rr * (s_r - un_r)
    denom_s = np.where(np.abs(denom) > 1e-12, denom, np.copysign(1e-12, denom + 1e-30))
    s_star = (pr - pl + rl * un_l * (s_l - un_l) - rr * un_r * (s_r - un_r)) / denom_s

    fac_l = rl * (s_l - un_l) / np.where(np.abs(s_l - s_star) > 1e-12, s_l - s_star, -1e-12)
    fac_r = rr * (s_r - un_r) / np.where(np.abs(s_r - s_star) > 1e-12, s_r - s_star, 1e-12)

    u_e_l = fac_l * (e_l + (s_star - un_l) * (s_star + pl / (rl * (s_l - un_l) - 1e-15)))
    u_e_r = fac_r * (e_r + (s_star - un_r) * (s_star + pr / (rr * (s_r - un_r) + 1e-15)))

    f0_sl = fl0 + s_l * (fac_l - rl)
    fn_sl = fln + s_l * (fac_l * s_star - rl * un_l)
    ft_sl = flt + s_l * (fac_l * ut_l - rl * ut_l)
    f3_sl = fl3 + s_l * (u_e_l - rl * e_l)

    f0_sr = fr0 + s_r * (fac_r - rr)
    fn_sr = frn + s_r * (fac_r * s_star - rr * un_r)
    ft_sr = frt + s_r * (fac_r * ut_r - rr * ut_r)
    f3_sr = fr3 + s_r * (u_e_r - rr * e_r)

    f0 = np.where(s_l >= 0.0, fl0, np.where(s_star >= 0.0, f0_sl, np.where(s_r >= 0.0, f0_sr, fr0)))
    fn = np.where(s_l >= 0.0, fln, np.where(s_star >= 0.0, fn_sl, np.where(s_r >= 0.0, fn_sr, frn)))
    ft = np.where(s_l >= 0.0, flt, np.where(s_star >= 0.0, ft_sl, np.where(s_r >= 0.0, ft_sr, frt)))
    f3 = np.where(s_l >= 0.0, fl3, np.where(s_star >= 0.0, f3_sl, np.where(s_r >= 0.0, f3_sr, fr3)))

    p_jump = np.abs(pr - pl) / (pr + pl + 1e-12)
    alpha = np.where(
        (s_l < 0.0) & (s_r > 0.0) & (p_jump > 0.12),
        np.minimum(0.40, 0.55 * p_jump**2),
        0.0,
    )
    if np.any(alpha > 0.0):
        sr_sl = np.maximum(1e-9, s_r - s_l)
        fh0 = (s_r * fl0 - s_l * fr0 + s_l * s_r * (rr - rl)) / sr_sl
        fhn = (s_r * fln - s_l * frn + s_l * s_r * (rr * un_r - rl * un_l)) / sr_sl
        fht = (s_r * flt - s_l * frt + s_l * s_r * (rr * ut_r - rl * ut_l)) / sr_sl
        fh3 = (s_r * fl3 - s_l * fr3 + s_l * s_r * (rr * e_r - rl * e_l)) / sr_sl
        f0 = (1.0 - alpha) * f0 + alpha * fh0
        fn = (1.0 - alpha) * fn + alpha * fhn
        ft = (1.0 - alpha) * ft + alpha * fht
        f3 = (1.0 - alpha) * f3 + alpha * fh3

    fx = fn * nx_f - ft * nr_f
    fr = fn * nr_f + ft * nx_f
    return f0, fx, fr, f3


def _limiter_vec(dl: np.ndarray, dr: np.ndarray, limiter: str) -> np.ndarray:
    if limiter == "first_order":
        return np.zeros_like(dl)
    active = (dl * dr) > 0.0
    if limiter == "minmod":
        raw = np.sign(dl) * np.minimum(np.abs(dl), np.abs(dr))
    elif limiter == "mc":
        raw = np.sign(dl) * np.minimum(
            np.minimum(2.0 * np.abs(dl), 2.0 * np.abs(dr)), 0.5 * np.abs(dl + dr)
        )
    else:
        raw = (2.0 * dl * dr) / (dl + dr + np.copysign(1e-30, dl + dr))
    return np.where(active, raw, 0.0)


def _solve_axisymmetric_ns2d_python(
    mesh: AxisymmetricMesh2D,
    rho1: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    p1: np.ndarray,
    gamma: float,
    r_gas: float,
    tc_k: float,
    pc_pa: float,
    pa_pa: float,
    cfl: float,
    inflow_slope: float,
    max_iters: int,
    tol: float,
    limiter: str,
    local_time_step: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    bool,
    float,
    float,
]:
    """Pure NumPy vectorized 2D axisymmetric FVM Euler solver (for parity testing & fallback)."""
    nx, nr = mesh.nx1, mesh.nr1
    xv, rv = mesh.x1_v, mesh.r1_v

    x0, r0 = xv[:-1, :-1], rv[:-1, :-1]
    x1, r1 = xv[1:, :-1], rv[1:, :-1]
    x2, r2 = xv[1:, 1:], rv[1:, 1:]
    x3, r3 = xv[:-1, 1:], rv[:-1, 1:]

    a1 = 0.5 * ((x1 - x0) * (r2 - r0) - (r1 - r0) * (x2 - x0))
    a2 = 0.5 * ((x2 - x0) * (r3 - r0) - (r2 - r0) * (x3 - x0))
    area = np.maximum(1e-18, a1 + a2)
    rg_num = a1 * (r0 + r1 + r2) / 3.0 + a2 * (r0 + r2 + r3) / 3.0
    rg = np.maximum(1e-9, rg_num / area)
    xc = (a1 * (x0 + x1 + x2) / 3.0 + a2 * (x0 + x2 + x3) / 3.0) / area
    vol = rg * area

    d_i = np.hypot(0.5 * (x1 + x2 - x0 - x3), 0.5 * (r1 + r2 - r0 - r3))
    d_j = np.hypot(0.5 * (x3 + x2 - x0 - x1), 0.5 * (r3 + r2 - r0 - r1))
    h_min = np.maximum(1e-7, np.minimum(d_i, d_j))

    dx_i = xv[:, 1:] - xv[:, :-1]
    dr_i = rv[:, 1:] - rv[:, :-1]
    ds_i = np.maximum(1e-12, np.hypot(dx_i, dr_i))
    i_nx, i_nr = dr_i / ds_i, -dx_i / ds_i
    i_sf = 0.5 * (rv[:, :-1] + rv[:, 1:]) * ds_i

    dx_j = xv[1:, :] - xv[:-1, :]
    dr_j = rv[1:, :] - rv[:-1, :]
    ds_j = np.maximum(1e-12, np.hypot(dx_j, dr_j))
    j_nx, j_nr = -dr_j / ds_j, dx_j / ds_j
    j_sf = 0.5 * (rv[:-1, :] + rv[1:, :]) * ds_j

    gm1 = gamma - 1.0
    rho = rho1.copy()
    u = u1.copy()
    v = v1.copy()
    p = p1.copy()

    def p2c(ro: np.ndarray, vx: np.ndarray, vr: np.ndarray, pr: np.ndarray):
        return ro, ro * vx, ro * vr, pr / gm1 + 0.5 * ro * (vx * vx + vr * vr)

    def c2p(r_c: np.ndarray, ru_c: np.ndarray, rv_c: np.ndarray, re_c: np.ndarray):
        rn = np.maximum(1e-8, r_c)
        ux = ru_c / rn
        vx = rv_c / rn
        pn = np.maximum(1.0, gm1 * np.maximum(1.0 / gm1, re_c - 0.5 * rn * (ux * ux + vx * vx)))
        return rn, ux, vx, pn

    U0, U1, U2, U3 = p2c(rho, u, v, p)
    rho_ref = pc_pa / (r_gas * tc_k)
    a_ref = math.sqrt(gamma * r_gas * tc_k)
    scale = max(1e-6, rho_ref * a_ref)
    res_hist: list[float] = []
    converged = False
    iters = 0
    init_res = -1.0

    y_frac = (np.arange(nr, dtype=np.float64) + 0.5) / nr
    slopes_in = inflow_slope * y_frac
    inv_s_in = 1.0 / np.sqrt(1.0 + slopes_in * slopes_in)

    def step_rhs(rc_: np.ndarray, uc_: np.ndarray, vc_: np.ndarray, pc_: np.ndarray):
        # I-faces
        rl_i = np.empty((nx + 1, nr), dtype=np.float64)
        ul_i = np.empty_like(rl_i)
        vl_i = np.empty_like(rl_i)
        pl_i = np.empty_like(rl_i)
        rr_i = np.empty_like(rl_i)
        ur_i = np.empty_like(rl_i)
        vr_i = np.empty_like(rl_i)
        pr_i = np.empty_like(rl_i)

        # Inlet (i=0)
        a0 = np.sqrt(gamma * pc_[0, :] / rc_[0, :])
        m0 = np.clip(np.hypot(uc_[0, :], vc_[0, :]) / np.maximum(1e-6, a0), 0.0, 0.95)
        f0_m = 1.0 + 0.5 * gm1 * m0 * m0
        tb = tc_k / f0_m
        pb = pc_pa * (f0_m ** (-gamma / gm1))
        rb = pb / (r_gas * tb)
        vb = m0 * np.sqrt(gamma * r_gas * tb)
        rl_i[0], ul_i[0], vl_i[0], pl_i[0] = rb, vb * inv_s_in, vb * slopes_in * inv_s_in, pb
        rr_i[0], ur_i[0], vr_i[0], pr_i[0] = rc_[0], uc_[0], vc_[0], pc_[0]

        # Outlet (i=nx)
        rl_i[nx], ul_i[nx], vl_i[nx], pl_i[nx] = rc_[-1], uc_[-1], vc_[-1], pc_[-1]
        ax_e = np.sqrt(gamma * pc_[-1] / rc_[-1])
        un_e = uc_[-1] * i_nx[nx] + vc_[-1] * i_nr[nx]
        ut_e = -uc_[-1] * i_nr[nx] + vc_[-1] * i_nx[nx]
        sup_e = un_e >= ax_e
        fwd_sub = (~sup_e) & (un_e >= 0.0)
        rho_sub = np.maximum(1e-8, rc_[-1] * (pa_pa / pc_[-1]) ** (1.0 / gamma))
        a_sub = np.sqrt(gamma * pa_pa / rho_sub)
        un_sub = np.maximum(0.0, un_e + 2.0 * (ax_e - a_sub) / gm1)
        u_sub = un_sub * i_nx[nx] - ut_e * i_nr[nx]
        v_sub = un_sub * i_nr[nx] + ut_e * i_nx[nx]
        rho_rev = np.maximum(1e-8, rc_[-1] * (pa_pa / pc_[-1]))
        rr_i[nx] = np.where(sup_e, rc_[-1], np.where(fwd_sub, rho_sub, rho_rev))
        pr_i[nx] = np.where(sup_e, pc_[-1], pa_pa)
        ur_i[nx] = np.where(sup_e, uc_[-1], np.where(fwd_sub, u_sub, 0.5 * uc_[-1]))
        vr_i[nx] = np.where(sup_e, vc_[-1], np.where(fwd_sub, v_sub, 0.5 * vc_[-1]))

        # Interior I-faces (1..nx-1) via MUSCL-TVD
        def rec_interior_i(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q_ext = np.concatenate([q[:1, :], q, q[-1:, :]], axis=0)
            dq_back = q_ext[1:-1] - q_ext[:-2]
            dq_fwd = q_ext[2:] - q_ext[1:-1]
            slope = _limiter_vec(dq_back, dq_fwd, limiter)
            return q[:-1] + 0.5 * slope[:-1], q[1:] - 0.5 * slope[1:]

        rl_i[1:nx], rr_i[1:nx] = rec_interior_i(rc_)
        ul_i[1:nx], ur_i[1:nx] = rec_interior_i(uc_)
        vl_i[1:nx], vr_i[1:nx] = rec_interior_i(vc_)
        pl_i[1:nx], pr_i[1:nx] = rec_interior_i(pc_)
        bad_l = (rl_i[1:nx] < 0.1 * rc_[:-1]) | (pl_i[1:nx] < 0.1 * pc_[:-1])
        bad_r = (rr_i[1:nx] < 0.1 * rc_[1:]) | (pr_i[1:nx] < 0.1 * pc_[1:])
        rl_i[1:nx] = np.where(bad_l, rc_[:-1], rl_i[1:nx])
        pl_i[1:nx] = np.where(bad_l, pc_[:-1], pl_i[1:nx])
        rr_i[1:nx] = np.where(bad_r, rc_[1:], rr_i[1:nx])
        pr_i[1:nx] = np.where(bad_r, pc_[1:], pr_i[1:nx])

        fi0, fi1, fi2, fi3 = _hllc_rotated_flux_vec(
            rl_i, ul_i, vl_i, pl_i, rr_i, ur_i, vr_i, pr_i, i_nx, i_nr, gamma
        )

        # J-faces
        fj0 = np.zeros((nx, nr + 1), dtype=np.float64)
        fj1 = np.zeros_like(fj0)
        fj2 = np.zeros_like(fj0)
        fj3 = np.zeros_like(fj0)

        fj1[:, 0] = j_nx[:, 0] * pc_[:, 0]
        fj2[:, 0] = j_nr[:, 0] * pc_[:, 0]
        fj1[:, nr] = j_nx[:, nr] * pc_[:, -1]
        fj2[:, nr] = j_nr[:, nr] * pc_[:, -1]

        def rec_interior_j(
            q: np.ndarray, mirror_sign_bot: float = 1.0, mirror_sign_top: float = 1.0
        ) -> tuple[np.ndarray, np.ndarray]:
            q_ext = np.concatenate(
                [q[:, :1] * mirror_sign_bot, q, q[:, -1:] * mirror_sign_top], axis=1
            )
            dq_b = q_ext[:, 1:-1] - q_ext[:, :-2]
            dq_f = q_ext[:, 2:] - q_ext[:, 1:-1]
            sl = _limiter_vec(dq_b, dq_f, limiter)
            return q[:, :-1] + 0.5 * sl[:, :-1], q[:, 1:] - 0.5 * sl[:, 1:]

        rl_j, rr_j = rec_interior_j(rc_, 1.0, 1.0)
        ul_j, ur_j = rec_interior_j(uc_, 1.0, 1.0)
        vl_j, vr_j = rec_interior_j(vc_, -1.0, -1.0)
        pl_j, pr_j = rec_interior_j(pc_, 1.0, 1.0)
        bad_jl = (rl_j < 0.1 * rc_[:, :-1]) | (pl_j < 0.1 * pc_[:, :-1])
        bad_jr = (rr_j < 0.1 * rc_[:, 1:]) | (pr_j < 0.1 * pc_[:, 1:])
        rl_j = np.where(bad_jl, rc_[:, :-1], rl_j)
        pl_j = np.where(bad_jl, pc_[:, :-1], pl_j)
        rr_j = np.where(bad_jr, rc_[:, 1:], rr_j)
        pr_j = np.where(bad_jr, pc_[:, 1:], pr_j)

        fj0[:, 1:nr], fj1[:, 1:nr], fj2[:, 1:nr], fj3[:, 1:nr] = _hllc_rotated_flux_vec(
            rl_j, ul_j, vl_j, pl_j, rr_j, ur_j, vr_j, pr_j, j_nx[:, 1:nr], j_nr[:, 1:nr], gamma
        )

        r_rhs0 = (
            fi0[:-1] * i_sf[:-1]
            - fi0[1:] * i_sf[1:]
            + fj0[:, :-1] * j_sf[:, :-1]
            - fj0[:, 1:] * j_sf[:, 1:]
        )
        r_rhs1 = (
            fi1[:-1] * i_sf[:-1]
            - fi1[1:] * i_sf[1:]
            + fj1[:, :-1] * j_sf[:, :-1]
            - fj1[:, 1:] * j_sf[:, 1:]
        )
        r_rhs2 = (
            fi2[:-1] * i_sf[:-1]
            - fi2[1:] * i_sf[1:]
            + fj2[:, :-1] * j_sf[:, :-1]
            - fj2[:, 1:] * j_sf[:, 1:]
            + pc_ * area
        )
        r_rhs3 = (
            fi3[:-1] * i_sf[:-1]
            - fi3[1:] * i_sf[1:]
            + fj3[:, :-1] * j_sf[:, :-1]
            - fj3[:, 1:] * j_sf[:, 1:]
        )
        return r_rhs0, r_rhs1, r_rhs2, r_rhs3, fi0

    for it in range(max_iters):
        if local_time_step:
            a_loc = np.sqrt(gamma * p / rho)
            dt_mat = cfl * h_min / np.maximum(1e-6, np.hypot(u, v) + a_loc)
        else:
            a_loc = np.sqrt(gamma * p / rho)
            dt_mat = float(np.min(cfl * h_min / np.maximum(1e-6, np.hypot(u, v) + a_loc)))

        r0, r1, r2, r3, _ = step_rhs(rho, u, v, p)
        inv_vol = 1.0 / vol
        U0_1 = U0 + dt_mat * inv_vol * r0
        U1_1 = U1 + dt_mat * inv_vol * r1
        U2_1 = U2 + dt_mat * inv_vol * r2
        U3_1 = U3 + dt_mat * inv_vol * r3
        rs, us, vs, ps = c2p(U0_1, U1_1, U2_1, U3_1)

        if local_time_step:
            a_loc_s = np.sqrt(gamma * ps / rs)
            dt_mat_s = cfl * h_min / np.maximum(1e-6, np.hypot(us, vs) + a_loc_s)
        else:
            dt_mat_s = dt_mat

        s0, s1, s2, s3, _ = step_rhs(rs, us, vs, ps)
        U0 = 0.5 * (U0 + U0_1 + dt_mat_s * inv_vol * s0)
        U1 = 0.5 * (U1 + U1_1 + dt_mat_s * inv_vol * s1)
        U2 = 0.5 * (U2 + U2_1 + dt_mat_s * inv_vol * s2)
        U3 = 0.5 * (U3 + U3_1 + dt_mat_s * inv_vol * s3)
        rho, u, v, p = c2p(U0, U1, U2, U3)

        cell_res = np.abs(s1) * inv_vol / scale * h_min
        r_cur = float(np.sqrt(np.mean(cell_res * cell_res)))
        if init_res < 0.0 and r_cur > 1e-16:
            init_res = r_cur
        r_rel = r_cur / init_res if init_res > 0.0 else r_cur
        res_hist.append(r_cur)
        iters = it + 1
        if it > 80 and (r_cur < tol or r_rel < 5e-4):
            converged = True
            break

    T = p / (rho * r_gas)
    a_end = np.sqrt(gamma * p / rho)
    M = np.hypot(u, v) / np.maximum(1e-6, a_end)
    drho_dx = np.gradient(rho, xc[:, 0], axis=0, edge_order=1)
    drho_dr = np.gradient(rho, axis=1) / np.maximum(1e-6, np.gradient(rg, axis=1))
    schlieren = np.hypot(drho_dx, drho_dr)

    _, _, _, _, fi0_final = step_rhs(rho, u, v, p)
    i_th = int(np.argmin(rv[:, -1]))
    mdot_th = float(2.0 * math.pi * np.sum(fi0_final[i_th] * i_sf[i_th]))
    mdot_ex = float(2.0 * math.pi * np.sum(fi0_final[nx] * i_sf[nx]))

    return rho, u, v, p, T, M, schlieren, np.asarray(res_hist), iters, converged, mdot_th, mdot_ex


def solve_axisymmetric_navier_stokes_2d(
    geom: NozzleGeometry,
    propellant_key: str = "METHALOX",
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    viscous_mode: str = "turbulent",
    limiter: str = "van_leer",
    nx1: int = 84,
    nr1: int = 28,
    include_plume: bool = True,
    nx2: int = 96,
    plume_length_radii: float = 12.0,
    cfl: float = 0.42,
    max_iters: int = 2400,
    tol: float = 2e-5,
    wall_temp_k: float | None = None,
    mu_ref_pa_s: float = 1.716e-5,
    use_cpp: bool = True,
    uniform_quiescent: bool = False,
) -> NavierStokes2DResult:
    """Solve 2D Axisymmetric Curvilinear Euler / Navier-Stokes across nozzle & external plume."""
    vmode = viscous_mode.lower()
    if vmode not in VALID_VISCOUS_MODES:
        raise ValueError(f"Unsupported viscous_mode '{viscous_mode}'; choose {VALID_VISCOUS_MODES}")
    lim = limiter.lower()
    if lim not in VALID_LIMITERS_2D:
        raise ValueError(f"Unsupported limiter '{limiter}'; choose {VALID_LIMITERS_2D}")

    if propellant_key.strip().upper() in ("COLD_GAS_N2", "AIR_COLD", "N2"):
        prop_key = propellant_key.strip().upper()
        gamma = 1.4
        r_spec = 8314.4626 / (28.964 if prop_key == "AIR_COLD" else 28.0134)
        tc_k = 300.0
    else:
        _, _, tc_k, gamma, mw = compute_chamber_thermochemistry(
            propellant_key, chamber_pressure_mpa=chamber_pressure_mpa
        )
        prop_key = propellant_key.strip().upper()
        r_spec = 8314.4626 / mw
    pc_pa = chamber_pressure_mpa * 1.0e6
    pa_pa = ambient_pressure_kpa * 1.0e3
    if uniform_quiescent:
        tc_k = 300.0
        pc_pa = pa_pa

    clustering = 0.0 if (vmode == "euler" or uniform_quiescent) else 1.5
    mesh = build_axisymmetric_nozzle_plume_mesh(
        geom,
        nx1=nx1,
        nr1=nr1,
        include_plume=include_plume,
        nx2=nx2,
        plume_length_radii=plume_length_radii,
        wall_clustering=clustering,
    )

    inflow_slope = float(
        (geom.radius_m[2] - geom.radius_m[0]) / max(1e-6, geom.x_m[2] - geom.x_m[0])
    )
    init1, init2 = _build_initial_fields(
        mesh, geom, gamma, r_spec, tc_k, pc_pa, pa_pa, uniform_quiescent=uniform_quiescent
    )
    rho1, u1, v1, p1 = init1

    cpp_lib = _load_cpp_ns2d_library() if use_cpp else None
    t_start = time.perf_counter()

    limiter_map = {"first_order": 0, "minmod": 1, "van_leer": 2, "mc": 3}
    viscous_map = {"euler": 0, "laminar": 1, "turbulent": 2}

    rho2 = u2 = v2 = p2 = T2 = M2 = sch2 = None

    if cpp_lib is not None:
        solver_backend = "cpp20_openmp_dll"
        c_dp = ctypes.POINTER(ctypes.c_double)

        T1 = np.empty_like(rho1)
        M1 = np.empty_like(rho1)
        sch1 = np.empty_like(rho1)

        if mesh.nx2 > 0 and init2 is not None:
            rho2, u2, v2, p2 = init2
            T2 = np.empty_like(rho2)
            M2 = np.empty_like(rho2)
            sch2 = np.empty_like(rho2)

            p_x2v = mesh.x2_v.ctypes.data_as(c_dp)
            p_r2v = mesh.r2_v.ctypes.data_as(c_dp)
            p_rho2_in, p_u2_in, p_v2_in, p_p2_in = (
                rho2.ctypes.data_as(c_dp),
                u2.ctypes.data_as(c_dp),
                v2.ctypes.data_as(c_dp),
                p2.ctypes.data_as(c_dp),
            )
            p_rho2_o, p_u2_o, p_v2_o, p_p2_o, p_T2_o, p_M2_o, p_s2_o = (
                rho2.ctypes.data_as(c_dp),
                u2.ctypes.data_as(c_dp),
                v2.ctypes.data_as(c_dp),
                p2.ctypes.data_as(c_dp),
                T2.ctypes.data_as(c_dp),
                M2.ctypes.data_as(c_dp),
                sch2.ctypes.data_as(c_dp),
            )
        else:
            null_dp = c_dp()
            p_x2v = p_r2v = null_dp
            p_rho2_in = p_u2_in = p_v2_in = p_p2_in = null_dp
            p_rho2_o = p_u2_o = p_v2_o = p_p2_o = p_T2_o = p_M2_o = p_s2_o = null_dp

        res_hist_buf = np.zeros(max_iters, dtype=np.float64)
        out_iters = ctypes.c_int(0)
        out_conv = ctypes.c_int(0)
        out_mdot_t = ctypes.c_double(0.0)
        out_mdot_e = ctypes.c_double(0.0)
        out_ms = ctypes.c_double(0.0)
        out_nth = ctypes.c_int(1)

        rc = cpp_lib.solve_axisymmetric_ns2d_c(
            nx1,
            nr1,
            mesh.x1_v.ctypes.data_as(c_dp),
            mesh.r1_v.ctypes.data_as(c_dp),
            rho1.ctypes.data_as(c_dp),
            u1.ctypes.data_as(c_dp),
            v1.ctypes.data_as(c_dp),
            p1.ctypes.data_as(c_dp),
            mesh.nx2,
            mesh.nr2,
            p_x2v,
            p_r2v,
            p_rho2_in,
            p_u2_in,
            p_v2_in,
            p_p2_in,
            float(gamma),
            float(r_spec),
            float(tc_k),
            float(pc_pa),
            float(pa_pa),
            float(mu_ref_pa_s),
            0.72,
            float(cfl),
            float(inflow_slope),
            int(max_iters),
            float(tol),
            int(limiter_map[lim]),
            int(viscous_map[vmode]),
            1,
            1 if wall_temp_k is not None else 0,
            float(wall_temp_k if wall_temp_k is not None else 900.0),
            rho1.ctypes.data_as(c_dp),
            u1.ctypes.data_as(c_dp),
            v1.ctypes.data_as(c_dp),
            p1.ctypes.data_as(c_dp),
            T1.ctypes.data_as(c_dp),
            M1.ctypes.data_as(c_dp),
            sch1.ctypes.data_as(c_dp),
            p_rho2_o,
            p_u2_o,
            p_v2_o,
            p_p2_o,
            p_T2_o,
            p_M2_o,
            p_s2_o,
            res_hist_buf.ctypes.data_as(c_dp),
            int(max_iters),
            ctypes.byref(out_iters),
            ctypes.byref(out_conv),
            ctypes.byref(out_mdot_t),
            ctypes.byref(out_mdot_e),
            ctypes.byref(out_ms),
            ctypes.byref(out_nth),
        )
        if rc != 0:
            raise RuntimeError(f"C++20 2D NS DLL returned error code {rc}")
        iters = int(out_iters.value)
        converged = bool(out_conv.value)
        mdot_th = float(out_mdot_t.value)
        mdot_ex = float(out_mdot_e.value)
        solve_ms = float(out_ms.value)
        n_threads = int(out_nth.value)
        res_hist = res_hist_buf[:iters].copy()
    else:
        solver_backend = "python_numpy_fvm2d"
        if vmode != "euler":
            # Python fallback runs inviscid Euler; native C++20/OpenMP handles viscous/turbulent
            solver_backend = "python_numpy_euler2d"
        rho1, u1, v1, p1, T1, M1, sch1, res_hist, iters, converged, mdot_th, mdot_ex = (
            _solve_axisymmetric_ns2d_python(
                mesh,
                rho1,
                u1,
                v1,
                p1,
                gamma=gamma,
                r_gas=r_spec,
                tc_k=tc_k,
                pc_pa=pc_pa,
                pa_pa=pa_pa,
                cfl=cfl,
                inflow_slope=inflow_slope,
                max_iters=max_iters,
                tol=tol,
                limiter=lim,
            )
        )
        solve_ms = (time.perf_counter() - t_start) * 1000.0
        n_threads = 1

    # Theoretical 1D choked mass flow rate
    a_t = math.pi * (geom.throat_radius_m**2)
    g_fn = math.sqrt(gamma) * ((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
    mdot_1d = pc_pa * a_t * g_fn / math.sqrt(r_spec * tc_k)
    cd = mdot_th / max(1e-9, mdot_1d)
    mass_err_pct = 100.0 * abs(mdot_ex - mdot_th) / max(1e-9, abs(mdot_th))

    # Extract 2D curved sonic line (M = 1.0 crossing along every radial j-row)
    sonic_x = np.empty(nr1, dtype=np.float64)
    sonic_r = np.empty(nr1, dtype=np.float64)
    for j in range(nr1):
        m_row = M1[:, j]
        x_row = mesh.x1_c[:, j]
        r_row = mesh.r1_c[:, j]
        crossings = np.where((m_row[:-1] < 1.0) & (m_row[1:] >= 1.0))[0]
        if len(crossings) > 0:
            k = int(crossings[0])
            w = (1.0 - m_row[k]) / max(1e-9, m_row[k + 1] - m_row[k])
            sonic_x[j] = x_row[k] + w * (x_row[k + 1] - x_row[k])
            sonic_r[j] = r_row[k] + w * (r_row[k + 1] - r_row[k])
        else:
            k = int(np.argmin(np.abs(m_row - 1.0)))
            sonic_x[j] = x_row[k]
            sonic_r[j] = r_row[k]
    sonic_offset_mm = float((sonic_x[0] - sonic_x[-1]) * 1000.0)

    # Detect viscous boundary-layer separation inside diverging section (u < -0.5 m/s)
    div_mask = mesh.x1_c[:, 0] > (geom.throat_x_m + 0.1 * geom.throat_radius_m)
    wall_u_div = u1[div_mask, -1]
    x_div = mesh.x1_c[div_mask, -1]
    sep_indices = np.where(wall_u_div < -0.5)[0]
    separation_detected = bool(len(sep_indices) > 0)
    separation_x_m = float(x_div[sep_indices[0]]) if separation_detected else None

    # Detect plume Mach disc & shock-diamond wavelength along Block 2 centerline
    mach_disc_detected = False
    mach_disc_x_m = None
    wavelength_m = None
    if M2 is not None and mesh.x2_c is not None and mesh.nx2 > 10:
        m_cl = M2[:, 0]
        x_cl = mesh.x2_c[:, 0]
        # Mach disc: sharp supersonic-to-subsonic normal shock drop on centerline
        disc_idx = np.where((m_cl[:-1] > 1.35) & (m_cl[1:] < 0.98))[0]
        if len(disc_idx) > 0:
            mach_disc_detected = True
            mach_disc_x_m = float(x_cl[disc_idx[0] + 1])

        # Peak-to-peak / trough-to-trough centerline Mach spacing (macro shock cells only)
        win = max(4, len(m_cl) // 10)
        span_m = float(np.max(m_cl) - np.min(m_cl))
        major_peaks: list[int] = []
        major_troughs: list[int] = []
        for idx in range(2, len(m_cl) - 2):
            lo = max(0, idx - win)
            hi = min(len(m_cl), idx + win + 1)
            seg = m_cl[lo:hi]
            amp = max(1.0, span_m)
            if m_cl[idx] >= float(np.max(seg)) and (m_cl[idx] - float(np.min(seg))) > 0.08 * amp:
                if not major_peaks or (idx - major_peaks[-1]) >= win:
                    major_peaks.append(idx)
            if m_cl[idx] <= float(np.min(seg)) and (float(np.max(seg)) - m_cl[idx]) > 0.08 * amp:
                if not major_troughs or (idx - major_troughs[-1]) >= win:
                    major_troughs.append(idx)
        pts = np.asarray(major_troughs if len(major_troughs) >= 2 else major_peaks, dtype=int)
        if len(pts) >= 2:
            wavelength_m = float(np.median(np.diff(x_cl[pts[:4]])))

    return NavierStokes2DResult(
        propellant=prop_key,
        viscous_mode=vmode,
        limiter=lim,
        solver_backend=solver_backend,
        n_threads=n_threads,
        solve_ms=solve_ms,
        iterations=iters,
        converged=converged,
        gamma=gamma,
        tc_k=tc_k,
        r_specific_j_kg_k=r_spec,
        chamber_pressure_mpa=chamber_pressure_mpa,
        ambient_pressure_kpa=ambient_pressure_kpa,
        x1_v=mesh.x1_v,
        r1_v=mesh.r1_v,
        x1_c=mesh.x1_c,
        r1_c=mesh.r1_c,
        rho_1=rho1,
        u_1=u1,
        v_1=v1,
        p_kpa_1=p1 / 1.0e3,
        T_k_1=T1,
        mach_1=M1,
        schlieren_1=sch1,
        x2_v=mesh.x2_v,
        r2_v=mesh.r2_v,
        x2_c=mesh.x2_c,
        r2_c=mesh.r2_c,
        rho_2=rho2,
        u_2=u2,
        v_2=v2,
        p_kpa_2=(p2 / 1.0e3) if p2 is not None else None,
        T_k_2=T2,
        mach_2=M2,
        schlieren_2=sch2,
        residual_history=res_hist,
        mass_flow_1d_kg_s=mdot_1d,
        mass_flow_2d_throat_kg_s=mdot_th,
        mass_flow_2d_exit_kg_s=mdot_ex,
        discharge_coefficient_cd=cd,
        mass_conservation_error_pct=mass_err_pct,
        exit_mach_centerline=float(M1[-1, 0]),
        exit_mach_wall=float(M1[-1, -1]),
        sonic_line_x_m=sonic_x,
        sonic_line_r_m=sonic_r,
        sonic_line_offset_mm=sonic_offset_mm,
        separation_detected=separation_detected,
        separation_x_m=separation_x_m,
        mach_disc_detected=mach_disc_detected,
        mach_disc_x_m=mach_disc_x_m,
        shock_diamond_wavelength_m=wavelength_m,
    )


def _write_single_vts_block(
    path: Path,
    xv: np.ndarray,
    rv: np.ndarray,
    mach: np.ndarray,
    p_kpa: np.ndarray,
    temp_k: np.ndarray,
    rho: np.ndarray,
    schlieren: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Path:
    """Write a curvilinear ParaView XML StructuredGrid (.vts) file with CellData fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx_p, nr_p = xv.shape
    nx, nr = nx_p - 1, nr_p - 1

    # ParaView .vts orders structured coordinates fastest along first axis (i then j then k)
    pts = []
    for j in range(nr_p):
        for i in range(nx_p):
            pts.append(f"{xv[i, j]:.6e} {rv[i, j]:.6e} 0.0")

    def fmt_cell(arr: np.ndarray) -> str:
        vals = [f"{arr[i, j]:.6e}" for j in range(nr) for i in range(nx)]
        return " ".join(vals)

    vel_vals = [f"{u[i, j]:.6e} {v[i, j]:.6e} 0.0" for j in range(nr) for i in range(nx)]

    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">',
        f'  <StructuredGrid WholeExtent="0 {nx} 0 {nr} 0 0">',
        f'    <Piece Extent="0 {nx} 0 {nr} 0 0">',
        "      <Points>",
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
        "          " + " ".join(pts),
        "        </DataArray>",
        "      </Points>",
        '      <CellData Scalars="Mach" Vectors="Velocity">',
        '        <DataArray type="Float64" Name="Mach" format="ascii">'
        + fmt_cell(mach)
        + "</DataArray>",
        '        <DataArray type="Float64" Name="Pressure_kPa" format="ascii">'
        + fmt_cell(p_kpa)
        + "</DataArray>",
        '        <DataArray type="Float64" Name="Temperature_K" format="ascii">'
        + fmt_cell(temp_k)
        + "</DataArray>",
        '        <DataArray type="Float64" Name="Density_kg_m3" format="ascii">'
        + fmt_cell(rho)
        + "</DataArray>",
        '        <DataArray type="Float64" Name="Numerical_Schlieren" format="ascii">'
        + fmt_cell(schlieren)
        + "</DataArray>",
        '        <DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">'
        + " ".join(vel_vals)
        + "</DataArray>",
        "      </CellData>",
        "    </Piece>",
        "  </StructuredGrid>",
        "</VTKFile>",
    ]
    path.write_text("\n".join(xml), encoding="utf-8")
    return path


def export_axisymmetric_vts(
    ns2d: NavierStokes2DResult,
    output_dir: str | Path,
    stem: str = "nozzle_ns2d",
) -> tuple[Path, Path | None]:
    """Export curvilinear 2D Euler/Navier-Stokes blocks to ParaView XML `.vts` files."""
    out = Path(output_dir)
    interior_p = _write_single_vts_block(
        out / f"{stem}_interior.vts",
        ns2d.x1_v,
        ns2d.r1_v,
        ns2d.mach_1,
        ns2d.p_kpa_1,
        ns2d.T_k_1,
        ns2d.rho_1,
        ns2d.schlieren_1,
        ns2d.u_1,
        ns2d.v_1,
    )
    plume_p = None
    if ns2d.x2_v is not None and ns2d.mach_2 is not None:
        plume_p = _write_single_vts_block(
            out / f"{stem}_plume.vts",
            ns2d.x2_v,
            ns2d.r2_v,
            ns2d.mach_2,
            ns2d.p_kpa_2,
            ns2d.T_k_2,
            ns2d.rho_2,
            ns2d.schlieren_2,
            ns2d.u_2,
            ns2d.v_2,
        )
    return interior_p, plume_p
