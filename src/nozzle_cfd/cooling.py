"""Bartz convective nozzle heat-transfer and regenerative cooling wall analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nozzle_cfd.cfd_solver import NozzleCFDResult
from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.propellants import (
    PROPELLANTS,
    compute_chamber_thermochemistry,
    vandenkerckhove_gamma,
)


@dataclass(frozen=True)
class NozzleWallMaterial:
    """Thermal and structural properties for thrust chamber liner alloys."""

    name: str
    display_name: str
    conductivity_w_m_k: float
    max_allowable_temp_k: float
    wall_thickness_mm: float


WALL_MATERIALS: dict[str, NozzleWallMaterial] = {
    "GRCOP_42": NozzleWallMaterial(
        name="GRCOP_42",
        display_name="GRCop-42 (Cu-4Cr-2Nb NASA Alloy)",
        conductivity_w_m_k=310.0,
        max_allowable_temp_k=1050.0,
        wall_thickness_mm=0.9,
    ),
    "CUC_ZR": NozzleWallMaterial(
        name="CUC_ZR",
        display_name="CuCrZr Regeneratively Cooled Liner",
        conductivity_w_m_k=320.0,
        max_allowable_temp_k=920.0,
        wall_thickness_mm=0.9,
    ),
    "INCONEL_718": NozzleWallMaterial(
        name="INCONEL_718",
        display_name="Inconel 718 Superalloy",
        conductivity_w_m_k=21.5,
        max_allowable_temp_k=1250.0,
        wall_thickness_mm=1.8,
    ),
}


@dataclass(frozen=True)
class CoolingAnalysisResult:
    """Axial thermal load distribution and regenerative cooling wall survival margins."""

    propellant: str
    wall_material: str
    coolant_temp_k: float
    coolant_htc_w_m2_k: float
    x_m: np.ndarray
    hg_w_m2_k: np.ndarray
    adiabatic_wall_temp_k: np.ndarray
    heat_flux_mw_m2: np.ndarray
    hot_wall_temp_k: np.ndarray
    coolant_wall_temp_k: np.ndarray
    peak_heat_flux_mw_m2: float
    peak_heat_flux_x_m: float
    peak_hot_wall_temp_k: float
    max_allowable_temp_k: float
    thermal_margin_k: float
    wall_survives: bool
    total_heat_load_mw: float


def analyze_bartz_cooling(
    geometry: NozzleGeometry,
    cfd_result: NozzleCFDResult,
    material_key: str = "GRCOP_42",
    coolant_bulk_temp_k: float = 135.0,
    coolant_htc_w_m2_k: float = 95000.0,
) -> CoolingAnalysisResult:
    """Evaluate axial convective heat flux and liner temperatures via the Bartz equation."""
    mat_key = material_key.strip().upper()
    if mat_key not in WALL_MATERIALS:
        valid = ", ".join(sorted(WALL_MATERIALS))
        raise ValueError(f"Unknown wall material '{material_key}'. Choose from: {valid}")

    mat = WALL_MATERIALS[mat_key]
    spec, _, tc_k, gamma, mw = compute_chamber_thermochemistry(
        cfd_result.propellant, chamber_pressure_mpa=cfd_result.chamber_pressure_mpa
    )
    _ = spec
    prop = PROPELLANTS[cfd_result.propellant]

    r_spec = 8314.4626 / mw
    cp = gamma * r_spec / (gamma - 1.0)
    pr = prop.prandtl
    mu_0 = prop.viscosity_ref_pa_s
    r_rec = pr ** (1.0 / 3.0)

    pc_pa = cfd_result.chamber_pressure_mpa * 1.0e6
    c_star = math.sqrt(r_spec * tc_k) / vandenkerckhove_gamma(gamma)
    dt_m = 2.0 * geometry.throat_radius_m
    rc_curv_m = 1.15 * geometry.throat_radius_m

    mach = np.maximum(0.03, cfd_result.mach)
    ar = np.maximum(1.0, geometry.area_ratio)

    # Adiabatic recovery temperature T_aw(x)
    gm1_half = 0.5 * (gamma - 1.0)
    m2 = mach**2
    t_aw = tc_k * (1.0 + r_rec * gm1_half * m2) / (1.0 + gm1_half * m2)

    # Solve coupled wall temperature iteratively (3 fixed-point steps converge < 0.05 K)
    t_wg = np.full_like(t_aw, 750.0)
    r_wall = (mat.wall_thickness_mm * 1.0e-3) / mat.conductivity_w_m_k
    r_cool = 1.0 / max(100.0, coolant_htc_w_m2_k)

    bartz_base = (
        (0.026 / (dt_m**0.2))
        * (((mu_0**0.2) * cp) / (pr**0.6))
        * ((pc_pa / c_star) ** 0.8)
        * ((dt_m / rc_curv_m) ** 0.1)
    )

    hg = np.zeros_like(t_aw)
    for _ in range(5):
        film_factor = 1.0 + gm1_half * m2
        sigma = ((0.5 * (t_wg / tc_k) * film_factor + 0.5) ** -0.68) * (film_factor**-0.12)
        hg = bartz_base * ((1.0 / ar) ** 0.9) * sigma
        r_total = (1.0 / np.maximum(hg, 1.0)) + r_wall + r_cool
        q_flux = np.maximum(0.0, (t_aw - coolant_bulk_temp_k) / r_total)
        t_wg = coolant_bulk_temp_k + q_flux * (r_wall + r_cool)

    q_flux_w_m2 = hg * np.maximum(0.0, t_aw - t_wg)
    q_flux_mw_m2 = q_flux_w_m2 / 1.0e6
    t_wc = coolant_bulk_temp_k + q_flux_w_m2 * r_cool

    # Total integrated thermal power transferred through axisymmetric nozzle wall (Watts -> MW)
    circumference = 2.0 * math.pi * geometry.radius_m
    total_heat_load_w = float(np.trapezoid(q_flux_w_m2 * circumference, geometry.x_m))

    peak_idx = int(np.argmax(q_flux_mw_m2))
    peak_q = float(q_flux_mw_m2[peak_idx])
    peak_x = float(geometry.x_m[peak_idx])
    peak_twg = float(np.max(t_wg))
    margin_k = mat.max_allowable_temp_k - peak_twg

    return CoolingAnalysisResult(
        propellant=cfd_result.propellant,
        wall_material=mat.display_name,
        coolant_temp_k=coolant_bulk_temp_k,
        coolant_htc_w_m2_k=coolant_htc_w_m2_k,
        x_m=geometry.x_m,
        hg_w_m2_k=hg,
        adiabatic_wall_temp_k=t_aw,
        heat_flux_mw_m2=q_flux_mw_m2,
        hot_wall_temp_k=t_wg,
        coolant_wall_temp_k=t_wc,
        peak_heat_flux_mw_m2=round(peak_q, 3),
        peak_heat_flux_x_m=round(peak_x, 4),
        peak_hot_wall_temp_k=round(peak_twg, 1),
        max_allowable_temp_k=mat.max_allowable_temp_k,
        thermal_margin_k=round(margin_k, 1),
        wall_survives=margin_k > 0.0,
        total_heat_load_mw=round(total_heat_load_w / 1.0e6, 3),
    )
