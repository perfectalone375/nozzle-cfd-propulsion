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
from nozzle_cfd.navier_stokes_2d import (
    AxisymmetricMesh2D,
    NavierStokes2DResult,
    build_axisymmetric_nozzle_plume_mesh,
    build_conical_nozzle_geometry,
    compute_hall_discharge_coefficient,
    export_axisymmetric_vts,
    solve_axisymmetric_navier_stokes_2d,
)
from nozzle_cfd.propellants import (
    G0,
    PROPELLANTS,
    PropellantPerformance,
    PropellantSpec,
    evaluate_propellant_performance,
    mach_from_area_ratio,
    run_propellant_trade_study,
)
from nozzle_cfd.riemann_fvm import (
    GridConvergenceResult,
    NozzleFVMResult,
    SodShockTubeResult,
    exact_sod_shock_tube,
    run_grid_convergence_study,
    solve_quasi1d_fvm,
    solve_sod_shock_tube,
)

__version__ = "0.3.0"

__all__ = [
    "G0",
    "PROPELLANTS",
    "WALL_MATERIALS",
    "AxisymmetricMesh2D",
    "CoolingAnalysisResult",
    "GridConvergenceResult",
    "NavierStokes2DResult",
    "NozzleCFDResult",
    "NozzleFVMResult",
    "NozzleGeometry",
    "NozzleWallMaterial",
    "Plume2DResult",
    "PropellantPerformance",
    "PropellantSpec",
    "SodShockTubeResult",
    "analyze_bartz_cooling",
    "build_axisymmetric_nozzle_plume_mesh",
    "build_conical_nozzle_geometry",
    "build_rao_bell_nozzle",
    "compute_hall_discharge_coefficient",
    "evaluate_propellant_performance",
    "exact_quasi1d_shock_solution",
    "exact_sod_shock_tube",
    "export_axisymmetric_vts",
    "mach_from_area_ratio",
    "run_grid_convergence_study",
    "run_propellant_trade_study",
    "simulate_2d_nozzle_and_plume",
    "solve_axisymmetric_navier_stokes_2d",
    "solve_quasi1d_fvm",
    "solve_quasi1d_maccormack",
    "solve_sod_shock_tube",
]
