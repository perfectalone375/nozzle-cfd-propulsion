"""Verification suite for Step 2: 2D Axisymmetric Curvilinear Navier-Stokes & Schlieren."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.mcp_server import handle_jsonrpc_request
from nozzle_cfd.navier_stokes_2d import (
    build_conical_nozzle_geometry,
    compute_hall_discharge_coefficient,
    export_axisymmetric_vts,
    solve_axisymmetric_navier_stokes_2d,
)
from nozzle_cfd.visualize import generate_2d_ns_schlieren_plot


def test_curvilinear_wedge_gcl_freesream_preservation() -> None:
    """Uniform quiescent gas on curved Rao bell must satisfy 1-radian wedge GCL."""
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.06, contraction_ratio=3.0, expansion_ratio=6.0, n_points=81
    )
    res = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="COLD_GAS_N2",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=1000.0,
        viscous_mode="euler",
        nx1=40,
        nr1=16,
        include_plume=False,
        max_iters=120,
        uniform_quiescent=True,
    )
    assert float(np.max(np.abs(res.u_1))) < 1e-8
    assert float(np.max(np.abs(res.v_1))) < 1e-8
    assert float(np.std(res.p_kpa_1)) < 1e-7


def test_jpl_45_15_and_hall_curved_sonic_line_and_cd() -> None:
    """2D Euler captures JPL 45-15 wall transonic overshoot & matches Hall (1962) Cd."""
    jpl = build_conical_nozzle_geometry(
        throat_radius_m=0.05,
        contraction_ratio=3.0,
        expansion_ratio=4.0,
        conv_half_angle_deg=45.0,
        div_half_angle_deg=15.0,
        throat_rc_ratio=0.625,
        n_points=101,
    )
    res_jpl = solve_axisymmetric_navier_stokes_2d(
        jpl,
        propellant_key="COLD_GAS_N2",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=2.0,
        viscous_mode="euler",
        nx1=64,
        nr1=22,
        include_plume=False,
        max_iters=2200,
    )
    assert res_jpl.converged is True
    assert res_jpl.mass_conservation_error_pct < 0.25
    assert res_jpl.sonic_line_offset_mm > 5.0

    # Throat wall Mach must exceed centerline Mach due to convex throat curvature
    i_th = int(np.argmin(jpl.radius_m))
    x_th = float(jpl.x_m[i_th])
    i_grid_th = int(np.argmin(np.abs(res_jpl.x1_c[:, 0] - x_th)))
    assert float(res_jpl.mach_1[i_grid_th, -1]) > float(res_jpl.mach_1[i_grid_th, 0]) + 0.15

    # Check Hall (1962) gentle throat curvature Cd agreement (< 1.5%)
    hall_noz = build_conical_nozzle_geometry(
        throat_radius_m=0.05,
        contraction_ratio=2.8,
        expansion_ratio=3.5,
        conv_half_angle_deg=35.0,
        div_half_angle_deg=15.0,
        throat_rc_ratio=2.0,
        n_points=101,
    )
    res_hall = solve_axisymmetric_navier_stokes_2d(
        hall_noz,
        propellant_key="COLD_GAS_N2",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=2.0,
        viscous_mode="euler",
        nx1=64,
        nr1=22,
        include_plume=False,
        max_iters=2200,
    )
    cd_theory = compute_hall_discharge_coefficient(1.4, 2.0)
    assert abs(res_hall.discharge_coefficient_cd - cd_theory) < 0.015


def test_coupled_nozzle_plume_mass_conservation_and_schlieren() -> None:
    """Two-block coupled nozzle + external plume preserves mass flow & captures shock diamonds."""
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12, contraction_ratio=3.2, expansion_ratio=12.0, n_points=101
    )
    res = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="METHALOX",
        chamber_pressure_mpa=10.0,
        ambient_pressure_kpa=101.325,
        viscous_mode="euler",
        nx1=56,
        nr1=20,
        include_plume=True,
        nx2=80,
        plume_length_radii=7.6,
        max_iters=1600,
    )
    assert res.mass_conservation_error_pct < 0.5
    assert res.mach_2 is not None and res.schlieren_2 is not None
    assert float(np.max(res.mach_2)) > 3.2
    assert float(np.max(res.schlieren_2)) > float(np.median(res.schlieren_2)) * 2.5


def test_viscous_turbulent_overexpanded_separation() -> None:
    """High ambient back-pressure drives adverse pressure gradient separating turbulent BL."""
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12, contraction_ratio=3.2, expansion_ratio=12.0, n_points=101
    )
    res_sep = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="METHALOX",
        chamber_pressure_mpa=1.2,
        ambient_pressure_kpa=320.0,
        viscous_mode="turbulent",
        nx1=64,
        nr1=22,
        include_plume=False,
        max_iters=1800,
    )
    assert res_sep.separation_detected is True
    assert res_sep.separation_x_m is not None
    assert res_sep.separation_x_m > geom.throat_x_m
    assert float(np.min(res_sep.u_1[:, -1])) < -1.0


def test_python_vs_cpp20_openmp_2d_parity() -> None:
    """C++20/OpenMP 2D DLL matches pure-NumPy curvilinear solver to machine precision."""
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.05, contraction_ratio=3.0, expansion_ratio=4.0, n_points=61
    )
    res_cpp = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="COLD_GAS_N2",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=101.325,
        viscous_mode="euler",
        nx1=36,
        nr1=12,
        include_plume=False,
        max_iters=250,
        use_cpp=True,
    )
    if res_cpp.solver_backend != "cpp20_openmp_dll":
        pytest.skip("C++20 2D OpenMP shared library not built in this environment")

    res_py = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="COLD_GAS_N2",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=101.325,
        viscous_mode="euler",
        nx1=36,
        nr1=12,
        include_plume=False,
        max_iters=250,
        use_cpp=False,
    )
    max_dm = float(np.max(np.abs(res_cpp.mach_1 - res_py.mach_1)))
    assert max_dm < 1e-6


def test_vts_paraview_export_schlieren_plot_and_mcp_2d_tool(tmp_path: Path) -> None:
    """ParaView XML .vts export, 2x2 Schlieren plot, and MCP run_2d_navier_stokes all succeed."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.08, contraction_ratio=3.0, expansion_ratio=6.0, n_points=81
    )
    res_plume = solve_axisymmetric_navier_stokes_2d(
        geom,
        propellant_key="METHALOX",
        chamber_pressure_mpa=5.0,
        ambient_pressure_kpa=101.325,
        viscous_mode="euler",
        nx1=40,
        nr1=14,
        include_plume=True,
        nx2=50,
        max_iters=600,
    )
    vts_int, vts_plume = export_axisymmetric_vts(res_plume, tmp_path, stem="unit_ns2d")
    assert vts_int.exists() and '<VTKFile type="StructuredGrid"' in vts_int.read_text("utf-8")
    assert vts_plume is not None and vts_plume.exists()

    png_path = generate_2d_ns_schlieren_plot(
        res_plume, res_plume, res_plume, tmp_path / "unit_schlieren.png"
    )
    assert png_path.exists() and png_path.stat().st_size > 10_000

    resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "run_2d_navier_stokes",
                "arguments": {
                    "nozzle_type": "jpl_45_15",
                    "propellant": "COLD_GAS_N2",
                    "chamber_pressure_mpa": 1.0,
                    "ambient_pressure_kpa": 2.0,
                    "viscous_mode": "euler",
                },
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert 0.94 < payload["discharge_coefficient_cd"] < 1.01
    assert payload["sonic_line_offset_mm"] > 4.0
    assert payload["mass_conservation_error_pct"] < 0.5
