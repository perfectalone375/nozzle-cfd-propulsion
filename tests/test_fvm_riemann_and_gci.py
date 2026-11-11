"""Verification tests for Step 1 Finite-Volume Riemann solvers (HLLC+, AUSM+, Rusanov),
MUSCL-TVD limiters, Toro's Sod Shock Tube, Python/C++20 parity, and ASME V&V 20 GCI.
"""

from __future__ import annotations

import json

import numpy as np

from nozzle_cfd.cfd_solver import solve_quasi1d_maccormack
from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.mcp_server import handle_jsonrpc_request
from nozzle_cfd.riemann_fvm import (
    compute_riemann_flux,
    exact_sod_shock_tube,
    reconstruct_muscl_interfaces,
    run_grid_convergence_study,
    solve_quasi1d_fvm,
    solve_sod_shock_tube,
)


def test_exact_sod_shock_tube_invariants() -> None:
    x = np.linspace(0.005, 0.995, 200)
    rho, u, p, mach = exact_sod_shock_tube(x, t_end=0.20, gamma=1.4)

    # Star region sits around x ~ 0.60 at t = 0.20 (Toro Table 4.1: p* = 0.30313, u* = 0.92745)
    idx_star = int(np.argmin(np.abs(x - 0.60)))
    assert abs(float(p[idx_star]) - 0.30313) < 5e-4
    assert abs(float(u[idx_star]) - 0.92745) < 5e-4
    assert np.all(rho > 0.12)
    assert np.all(p > 0.09)
    assert float(np.max(mach)) > 0.85


def test_muscl_tvd_and_hllc_beat_first_order_rusanov() -> None:
    sod_1st = solve_sod_shock_tube(nx=200, flux_scheme="rusanov", limiter="first_order")
    sod_min = solve_sod_shock_tube(nx=200, flux_scheme="hllc", limiter="minmod")
    sod_vl = solve_sod_shock_tube(nx=200, flux_scheme="hllc", limiter="van_leer")
    sod_sb = solve_sod_shock_tube(nx=200, flux_scheme="hllc", limiter="superbee")
    sod_ausm = solve_sod_shock_tube(nx=200, flux_scheme="ausm_plus", limiter="van_leer")

    # Strictly sharper contact & shock capture with higher-resolution TVD limiters
    assert sod_sb.rho_l1_error < sod_vl.rho_l1_error < sod_min.rho_l1_error < sod_1st.rho_l1_error
    assert sod_ausm.rho_l1_error < sod_1st.rho_l1_error * 0.25

    # TVD monotonicity guarantee: no unphysical overshoot above initial left state
    assert float(np.max(sod_vl.p_num)) <= 1.0 + 1e-6
    assert float(np.min(sod_vl.rho_num)) >= 0.125 - 1e-6


def test_well_balanced_variable_area_quiescent_identity() -> None:
    geom = build_rao_bell_nozzle(expansion_ratio=12.0, n_points=81)
    area = geom.area_ratio
    dx = float(geom.x_m[1] - geom.x_m[0]) / geom.throat_radius_m
    a_int = np.empty(len(area) + 1)
    a_int[1:-1] = 0.5 * (area[:-1] + area[1:])
    a_int[0] = area[0]
    a_int[-1] = area[-1] + 0.5 * (area[-1] - area[-2])
    da_cell = a_int[1:] - a_int[:-1]

    rho = np.ones_like(area)
    vel = np.zeros_like(area)
    pres = np.ones_like(area)

    rl, ul, pl, rr, ur, pr = reconstruct_muscl_interfaces(
        np.pad(rho, (2, 2), mode="edge"),
        np.pad(vel, (2, 2), mode="edge"),
        np.pad(pres, (2, 2), mode="edge"),
        limiter="van_leer",
    )
    f1, f2, f3 = compute_riemann_flux(rl, ul, pl, rr, ur, pr, gamma=1.21, scheme="hllc")
    du2 = -(f2[1:] * a_int[1:] - f2[:-1] * a_int[:-1]) / (area * dx) + pres * da_cell / (area * dx)

    assert float(np.max(np.abs(f1))) < 1e-12
    assert float(np.max(np.abs(f3))) < 1e-12
    assert float(np.max(np.abs(du2))) < 1e-12


def test_hllc_and_ausm_capture_shock_with_sub_percent_error() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.10, contraction_ratio=3.0, expansion_ratio=8.0, n_points=121
    )
    mac = solve_quasi1d_maccormack(geom, "METHALOX", 1.0, 380.0)
    hllc = solve_quasi1d_fvm(geom, "METHALOX", 1.0, 380.0, flux_scheme="hllc", limiter="van_leer")
    ausm = solve_quasi1d_fvm(
        geom, "METHALOX", 1.0, 380.0, flux_scheme="ausm_plus", limiter="van_leer"
    )

    for res in (hllc, ausm):
        assert res.converged is True
        assert res.shock_detected is True
        assert res.shock_upstream_mach is not None and res.shock_upstream_mach > 2.2
        assert res.shock_downstream_mach is not None and res.shock_downstream_mach < 0.65
        assert res.mach_l2_error_pct < 0.5
        assert res.mass_conservation_error_pct < 0.3
        # Strictly beats 1969 MacCormack FD error
        assert res.mach_l2_error_pct < mac.mach_l2_error_pct


def test_python_and_cpp20_dll_exact_parity() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.10, contraction_ratio=3.0, expansion_ratio=8.0, n_points=121
    )
    py_res = solve_quasi1d_fvm(
        geom, "METHALOX", 1.0, 380.0, flux_scheme="hllc", limiter="van_leer", backend="python"
    )
    cpp_res = solve_quasi1d_fvm(
        geom, "METHALOX", 1.0, 380.0, flux_scheme="hllc", limiter="van_leer", backend="cpp"
    )

    assert py_res.solver_backend == "python_numpy"
    assert cpp_res.solver_backend == "cpp20_native_dll"
    assert float(np.max(np.abs(py_res.mach - cpp_res.mach))) < 1e-4
    assert float(np.max(np.abs(py_res.pressure_kpa - cpp_res.pressure_kpa))) < 0.05


def test_asme_vv20_grid_convergence_gci() -> None:
    gci = run_grid_convergence_study(
        propellant_key="METHALOX",
        expansion_ratio=12.0,
        flux_scheme="hllc",
        limiter="van_leer",
        grid_sizes=(81, 161, 321),
    )
    e_coarse, e_med, e_fine = gci.mach_l2_errors_pct
    assert e_fine < e_med < e_coarse
    assert gci.observed_order_error >= 0.9
    assert gci.gci_fine_pct < 0.25
    assert gci.is_grid_converged is True


def test_mcp_fvm_and_gci_tools() -> None:
    sod_call = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "run_fvm_riemann_benchmark",
                "arguments": {"benchmark": "sod", "flux_scheme": "hllc", "limiter": "superbee"},
            },
        }
    )
    assert sod_call is not None and sod_call["result"]["isError"] is False
    sod_data = json.loads(sod_call["result"]["content"][0]["text"])
    assert sod_data["rho_l1_error"] < 0.004

    gci_call = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "run_grid_convergence_study",
                "arguments": {"expansion_ratio": 12.0},
            },
        }
    )
    assert gci_call is not None and gci_call["result"]["isError"] is False
    gci_data = json.loads(gci_call["result"]["content"][0]["text"])
    assert gci_data["is_grid_converged"] is True
