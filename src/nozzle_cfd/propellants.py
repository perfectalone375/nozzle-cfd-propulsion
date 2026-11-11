"""Multi-propellant rocket combustion thermochemistry and 1D nozzle performance."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from scipy.optimize import brentq

G0 = 9.80665  # Standard gravity [m/s^2]
R_UNIVERSAL = 8314.4626  # Universal gas constant [J/(kmol*K)]


@dataclass(frozen=True)
class PropellantSpec:
    """Thermochemical specification for a rocket propellant combination."""

    name: str
    display_name: str
    oxidizer: str
    fuel: str
    nominal_of_ratio: float
    of_min: float
    of_max: float
    tc_nominal_k: float
    gamma_nominal: float
    mw_nominal_kg_kmol: float
    prandtl: float
    viscosity_ref_pa_s: float
    notes: str


PROPELLANTS: dict[str, PropellantSpec] = {
    "METHALOX": PropellantSpec(
        name="METHALOX",
        display_name="LOX / Liquid Methane (Methalox)",
        oxidizer="LOX",
        fuel="LCH4",
        nominal_of_ratio=3.6,
        of_min=2.8,
        of_max=4.2,
        tc_nominal_k=3560.0,
        gamma_nominal=1.21,
        mw_nominal_kg_kmol=20.8,
        prandtl=0.72,
        viscosity_ref_pa_s=9.8e-5,
        notes="Clean-burning reusable booster/upper-stage propellant (Raptor / BE-4 class).",
    ),
    "KEROLOX": PropellantSpec(
        name="KEROLOX",
        display_name="LOX / RP-1 Kerosene (Kerolox)",
        oxidizer="LOX",
        fuel="RP-1",
        nominal_of_ratio=2.65,
        of_min=2.1,
        of_max=3.1,
        tc_nominal_k=3670.0,
        gamma_nominal=1.19,
        mw_nominal_kg_kmol=23.2,
        prandtl=0.74,
        viscosity_ref_pa_s=1.02e-4,
        notes="High-density booster propellant (Merlin / F-1 / RD-180 class).",
    ),
    "HYDROLOX": PropellantSpec(
        name="HYDROLOX",
        display_name="LOX / Liquid Hydrogen (Hydrolox)",
        oxidizer="LOX",
        fuel="LH2",
        nominal_of_ratio=6.0,
        of_min=4.5,
        of_max=7.5,
        tc_nominal_k=3500.0,
        gamma_nominal=1.20,
        mw_nominal_kg_kmol=13.4,
        prandtl=0.69,
        viscosity_ref_pa_s=8.9e-5,
        notes="Highest chemical Isp upper-stage & sustainer propellant (RS-25 / RL-10 class).",
    ),
    "SOLID_APCP": PropellantSpec(
        name="SOLID_APCP",
        display_name="APCP / HTPB / Al Solid Composite",
        oxidizer="AP (NH4ClO4)",
        fuel="HTPB + 18% Al",
        nominal_of_ratio=5.2,
        of_min=4.5,
        of_max=6.0,
        tc_nominal_k=3380.0,
        gamma_nominal=1.17,
        mw_nominal_kg_kmol=26.8,
        prandtl=0.78,
        viscosity_ref_pa_s=1.08e-4,
        notes="High-thrust composite solid booster grain (SLS SRB / P120C class).",
    ),
}


@dataclass(frozen=True)
class PropellantPerformance:
    """Evaluated chamber & nozzle performance metrics for a given operating point."""

    propellant: str
    display_name: str
    of_ratio: float
    chamber_pressure_mpa: float
    ambient_pressure_kpa: float
    expansion_ratio: float
    throat_radius_m: float
    combustion_efficiency: float
    nozzle_efficiency: float
    tc_k: float
    gamma: float
    mw_kg_kmol: float
    r_specific_j_kg_k: float
    gamma_function: float
    c_star_ideal_m_s: float
    c_star_actual_m_s: float
    exit_mach: float
    exit_pressure_kpa: float
    exit_temperature_k: float
    exit_velocity_m_s: float
    mass_flow_kg_s: float
    thrust_coefficient: float
    thrust_kn: float
    isp_s: float
    isp_vacuum_s: float
    thrust_vacuum_kn: float
    expansion_regime: str
    separation_risk: bool
    summerfield_ratio_pe_pa: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_chamber_thermochemistry(
    propellant_key: str,
    chamber_pressure_mpa: float = 10.0,
    of_ratio: float | None = None,
) -> tuple[PropellantSpec, float, float, float, float]:
    """Compute realistic chamber Tc, gamma, Mw, and effective O/F using NASA CEA trends.

    Adjusts equilibrium flame temperature slightly upward with higher chamber pressure
    (suppressed endothermic dissociation) and models realistic quadratic drop-off away
    from optimal O/F mixture ratio.
    """
    key = propellant_key.strip().upper()
    if key not in PROPELLANTS:
        valid = ", ".join(sorted(PROPELLANTS))
        raise ValueError(f"Unknown propellant '{propellant_key}'. Choose from: {valid}")
    if chamber_pressure_mpa <= 0.1:
        raise ValueError("chamber_pressure_mpa must be > 0.1 MPa")

    spec = PROPELLANTS[key]
    effective_of = spec.nominal_of_ratio if of_ratio is None else float(of_ratio)
    effective_of = max(spec.of_min * 0.8, min(spec.of_max * 1.2, effective_of))

    # Normalized off-design mixture ratio offset
    half_span = max(0.25, 0.5 * (spec.of_max - spec.of_min))
    delta_of = (effective_of - spec.nominal_of_ratio) / half_span

    # Dissociation pressure correction (~+1.8% Tc per decade of Pc above 7 MPa)
    pressure_factor = 1.0 + 0.018 * math.log10(max(chamber_pressure_mpa, 0.5) / 10.0)
    tc_k = spec.tc_nominal_k * pressure_factor * (1.0 - 0.065 * (delta_of**2))
    tc_k = max(2100.0, tc_k)

    # Rich mixtures lower average molecular weight (more unburned H2/CO/CH4); lean raises it
    mw = spec.mw_nominal_kg_kmol * (1.0 + 0.085 * delta_of)
    mw = max(8.0, mw)

    # Higher Pc / cooler rich mixture raises gamma slightly
    gamma = (
        spec.gamma_nominal
        + 0.008 * abs(delta_of)
        + 0.004 * math.log10(max(chamber_pressure_mpa, 0.5) / 10.0)
    )
    gamma = max(1.12, min(1.36, gamma))

    return spec, effective_of, tc_k, gamma, mw


def vandenkerckhove_gamma(gamma: float) -> float:
    """Compute Vandenkerckhove function Gamma(gamma) for choked mass flow."""
    return math.sqrt(gamma) * ((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))


def area_mach_function(mach: float, gamma: float) -> float:
    """Isentropic area ratio A/A* for a given Mach number and gamma."""
    if mach <= 0.0:
        raise ValueError("mach must be positive")
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * (mach**2))
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / mach) * (term**exponent)


def mach_from_area_ratio(
    area_ratio: float,
    gamma: float,
    supersonic: bool = True,
) -> float:
    """Invert the isentropic area-Mach relation accurately via Brent root finding."""
    if area_ratio < 1.0 - 1e-12:
        raise ValueError(f"area_ratio must be >= 1.0, got {area_ratio}")
    if abs(area_ratio - 1.0) <= 1e-9:
        return 1.0

    def residual(m: float) -> float:
        return area_mach_function(m, gamma) - area_ratio

    if supersonic:
        lo = 1.0 + 1e-7
        hi = 25.0
        while residual(hi) < 0.0 and hi < 100.0:
            hi *= 1.5
        return float(brentq(residual, lo, hi, xtol=1e-10))

    lo = 1e-5
    hi = 1.0 - 1e-7
    return float(brentq(residual, lo, hi, xtol=1e-10))


def classify_expansion_regime(pe_kpa: float, pa_kpa: float) -> tuple[str, bool, float]:
    """Classify nozzle expansion and evaluate Summerfield flow separation margin."""
    if pa_kpa <= 1e-6:
        return "VACUUM_UNDEREXPANDED", False, float("inf")
    ratio = pe_kpa / pa_kpa
    if ratio >= 1.15:
        return "UNDEREXPANDED", False, ratio
    if 0.85 <= ratio < 1.15:
        return "OPTIMALLY_EXPANDED", False, ratio
    if 0.40 <= ratio < 0.85:
        return "OVEREXPANDED_ATTACHED", False, ratio
    return "OVEREXPANDED_SEPARATED_RISK", True, ratio


def evaluate_propellant_performance(
    propellant_key: str,
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    expansion_ratio: float = 16.0,
    throat_radius_m: float = 0.12,
    of_ratio: float | None = None,
    combustion_efficiency: float = 0.985,
    nozzle_efficiency: float = 0.985,
) -> PropellantPerformance:
    """Evaluate 1D + divergence-corrected rocket nozzle propulsion performance."""
    if expansion_ratio < 1.0:
        raise ValueError("expansion_ratio must be >= 1.0")
    if throat_radius_m <= 0.0:
        raise ValueError("throat_radius_m must be > 0")

    spec, eff_of, tc_k, gamma, mw = compute_chamber_thermochemistry(
        propellant_key,
        chamber_pressure_mpa=chamber_pressure_mpa,
        of_ratio=of_ratio,
    )
    r_spec = R_UNIVERSAL / mw
    big_gamma = vandenkerckhove_gamma(gamma)
    c_star_ideal = math.sqrt(r_spec * tc_k) / big_gamma
    c_star_actual = combustion_efficiency * c_star_ideal

    pc_pa = chamber_pressure_mpa * 1.0e6
    pa_pa = max(0.0, ambient_pressure_kpa * 1.0e3)
    at_m2 = math.pi * (throat_radius_m**2)

    m_exit = mach_from_area_ratio(expansion_ratio, gamma, supersonic=True)
    isentropic_factor = 1.0 + 0.5 * (gamma - 1.0) * (m_exit**2)
    t_exit = tc_k / isentropic_factor
    p_exit_pa = pc_pa * (isentropic_factor ** (-gamma / (gamma - 1.0)))
    p_exit_kpa = p_exit_pa / 1.0e3

    # Momentum thrust coefficient + pressure thrust term
    pr_ratio = p_exit_pa / pc_pa
    cf_momentum_ideal = math.sqrt(
        (2.0 * (gamma**2) / (gamma - 1.0))
        * ((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
        * (1.0 - (pr_ratio ** ((gamma - 1.0) / gamma)))
    )
    cf_momentum = nozzle_efficiency * cf_momentum_ideal
    cf_vac = cf_momentum + expansion_ratio * pr_ratio
    cf = cf_momentum + expansion_ratio * (p_exit_pa - pa_pa) / pc_pa

    mdot = (pc_pa * at_m2) / c_star_actual
    thrust_n = pc_pa * at_m2 * cf
    thrust_vac_n = pc_pa * at_m2 * cf_vac

    ve_effective = thrust_n / mdot
    isp_s = ve_effective / G0
    isp_vac_s = (thrust_vac_n / mdot) / G0

    regime, sep_risk, pe_pa_ratio = classify_expansion_regime(p_exit_kpa, ambient_pressure_kpa)

    return PropellantPerformance(
        propellant=spec.name,
        display_name=spec.display_name,
        of_ratio=round(eff_of, 3),
        chamber_pressure_mpa=round(chamber_pressure_mpa, 3),
        ambient_pressure_kpa=round(ambient_pressure_kpa, 3),
        expansion_ratio=round(expansion_ratio, 3),
        throat_radius_m=round(throat_radius_m, 4),
        combustion_efficiency=combustion_efficiency,
        nozzle_efficiency=nozzle_efficiency,
        tc_k=round(tc_k, 2),
        gamma=round(gamma, 4),
        mw_kg_kmol=round(mw, 3),
        r_specific_j_kg_k=round(r_spec, 2),
        gamma_function=round(big_gamma, 5),
        c_star_ideal_m_s=round(c_star_ideal, 2),
        c_star_actual_m_s=round(c_star_actual, 2),
        exit_mach=round(m_exit, 4),
        exit_pressure_kpa=round(p_exit_kpa, 3),
        exit_temperature_k=round(t_exit, 2),
        exit_velocity_m_s=round(ve_effective, 2),
        mass_flow_kg_s=round(mdot, 3),
        thrust_coefficient=round(cf, 4),
        thrust_kn=round(thrust_n / 1.0e3, 3),
        isp_s=round(isp_s, 2),
        isp_vacuum_s=round(isp_vac_s, 2),
        thrust_vacuum_kn=round(thrust_vac_n / 1.0e3, 3),
        expansion_regime=regime,
        separation_risk=sep_risk,
        summerfield_ratio_pe_pa=round(pe_pa_ratio, 4) if math.isfinite(pe_pa_ratio) else 9999.0,
    )


def run_propellant_trade_study(
    chamber_pressure_mpa: float = 10.0,
    ambient_pressure_kpa: float = 101.325,
    expansion_ratio: float = 16.0,
    throat_radius_m: float = 0.12,
) -> list[PropellantPerformance]:
    """Compare all built-in rocket propellants at identical chamber & nozzle geometry."""
    results = [
        evaluate_propellant_performance(
            key,
            chamber_pressure_mpa=chamber_pressure_mpa,
            ambient_pressure_kpa=ambient_pressure_kpa,
            expansion_ratio=expansion_ratio,
            throat_radius_m=throat_radius_m,
        )
        for key in ("HYDROLOX", "METHALOX", "KEROLOX", "SOLID_APCP")
    ]
    return sorted(results, key=lambda item: item.isp_s, reverse=True)
