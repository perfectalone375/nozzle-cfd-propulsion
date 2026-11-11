"""Tests for MacCormack shock-capturing CFD, 2D MOC plume, Bartz cooling, VTK, CLI & MCP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nozzle_cfd.cfd_solver import solve_quasi1d_maccormack
from nozzle_cfd.cli import main as cli_main
from nozzle_cfd.cooling import analyze_bartz_cooling
from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.mcp_server import handle_jsonrpc_request
from nozzle_cfd.moc_2d import simulate_2d_nozzle_and_plume
from nozzle_cfd.visualize import export_plume_vtk


def test_maccormack_supersonic_design_accuracy() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12, contraction_ratio=3.2, expansion_ratio=12.0, n_points=121
    )
    cfd = solve_quasi1d_maccormack(
        geom,
        propellant_key="METHALOX",
        chamber_pressure_mpa=10.0,
        ambient_pressure_kpa=101.325,
    )
    assert cfd.converged is True
    assert cfd.shock_detected is False
    assert cfd.mach_l2_error_pct < 1.0
    assert cfd.mass_conservation_error_pct < 0.5
    throat_idx = int(np.argmin(geom.area_ratio))
    assert abs(float(cfd.mach[throat_idx]) - 1.0) < 0.06


def test_maccormack_captures_standing_normal_shock() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.10, contraction_ratio=3.0, expansion_ratio=8.0, n_points=121
    )
    cfd = solve_quasi1d_maccormack(
        geom,
        propellant_key="METHALOX",
        chamber_pressure_mpa=1.0,
        ambient_pressure_kpa=380.0,
    )
    assert cfd.converged is True
    assert cfd.flow_regime == "NORMAL_SHOCK_INSIDE_NOZZLE"
    assert cfd.shock_detected is True
    assert cfd.shock_upstream_mach is not None and cfd.shock_upstream_mach > 1.8
    assert cfd.shock_downstream_mach is not None and cfd.shock_downstream_mach < 0.8
    assert abs(float(cfd.pressure_kpa[-1]) - 380.0) < 2.0
    assert cfd.mach_l2_error_pct < 5.0


def test_bartz_cooling_material_survival_tradeoff() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12, contraction_ratio=3.2, expansion_ratio=12.0, n_points=121
    )
    cfd = solve_quasi1d_maccormack(geom, "METHALOX", 10.0, 101.325)
    grcop = analyze_bartz_cooling(geom, cfd, material_key="GRCOP_42")
    inconel = analyze_bartz_cooling(geom, cfd, material_key="INCONEL_718")

    # Peak heat flux must occur right near the throat where mass flux * density is maximal
    assert abs(grcop.peak_heat_flux_x_m - geom.throat_x_m) < 0.06
    assert grcop.peak_heat_flux_mw_m2 > 20.0
    assert grcop.wall_survives is True
    assert inconel.wall_survives is False
    assert grcop.peak_hot_wall_temp_k < inconel.peak_hot_wall_temp_k


def test_2d_plume_and_paraview_vtk_export(tmp_path: Path) -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12, contraction_ratio=3.2, expansion_ratio=12.0, n_points=81
    )
    cfd = solve_quasi1d_maccormack(geom, "METHALOX", 10.0, 101.325)
    plume = simulate_2d_nozzle_and_plume(geom, cfd, nx_plume=60, nr_half=25)
    assert plume.shock_diamond_wavelength_m > 0.1
    assert len(plume.moc_rays) >= 6

    vtk_path = export_plume_vtk(plume, tmp_path / "test_plume.vtk")
    content = vtk_path.read_text(encoding="utf-8")
    assert "DATASET STRUCTURED_GRID" in content
    assert "SCALARS Mach_Number float 1" in content
    assert "SCALARS Static_Pressure_kPa float 1" in content


def test_cli_export_all_and_mcp_jsonrpc(tmp_path: Path, capsys: object) -> None:
    _ = capsys
    rc = cli_main(
        [
            "--propellant",
            "KEROLOX",
            "--nx",
            "81",
            "--export-all",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0
    assert (tmp_path / "nozzle_plume_field.vtk").exists()
    assert (tmp_path / "nozzle_shock_cfd_analysis.png").exists()
    assert (tmp_path / "fvm_riemann_validation.png").exists()
    assert (tmp_path / "multi_propellant_trade_study.png").exists()
    assert (tmp_path / "plume_mach_diamonds_2d.png").exists()
    assert (tmp_path / "nozzle_cfd_benchmark_report.json").exists()

    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "nozzle-cfd-mcp"

    tools_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    assert len(tools_resp["result"]["tools"]) == 6

    call_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "compare_propellants", "arguments": {"expansion_ratio": 16.0}},
        }
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    parsed = json.loads(call_resp["result"]["content"][0]["text"])
    assert len(parsed["results"]) == 4
