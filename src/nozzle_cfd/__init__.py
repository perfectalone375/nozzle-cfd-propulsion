"""Multi-propellant rocket propulsion and supersonic nozzle CFD digital twin."""

from nozzle_cfd.cfd_solver import (
    NozzleCFDResult,
    exact_quasi1d_shock_solution,
    solve_quasi1d_maccormack,
)
from nozzle_cfd.cooling import (
    WALL_MATERIALS,
    CoolingAnalysisResult,
    NozzleWallMaterial,
    analyze_bartz_cooling,
)
from nozzle_cfd.geometry import NozzleGeometry, build_rao_bell_nozzle
from nozzle_cfd.moc_2d import Plume2DResult, simulate_2d_nozzle_and_plume
from nozzle_cfd.propellants import (
    G0,
    PROPELLANTS,
    PropellantPerformance,
    PropellantSpec,
    evaluate_propellant_performance,
    mach_from_area_ratio,
    run_propellant_trade_study,
)

__version__ = "0.1.0"

__all__ = [
    "G0",
    "PROPELLANTS",
    "WALL_MATERIALS",
    "CoolingAnalysisResult",
    "NozzleCFDResult",
    "NozzleGeometry",
    "NozzleWallMaterial",
    "Plume2DResult",
    "PropellantPerformance",
    "PropellantSpec",
    "analyze_bartz_cooling",
    "build_rao_bell_nozzle",
    "evaluate_propellant_performance",
    "exact_quasi1d_shock_solution",
    "mach_from_area_ratio",
    "run_propellant_trade_study",
    "simulate_2d_nozzle_and_plume",
    "solve_quasi1d_maccormack",
]
