"""Command-line interface for multi-propellant rocket propulsion and CFD digital twin."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nozzle_cfd.cfd_solver import solve_quasi1d_maccormack
from nozzle_cfd.cooling import analyze_bartz_cooling
from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.moc_2d import simulate_2d_nozzle_and_plume
from nozzle_cfd.propellants import (
    PROPELLANTS,
    evaluate_propellant_performance,
    run_propellant_trade_study,
)
from nozzle_cfd.visualize import (
    export_plume_vtk,
    generate_2d_plume_plot,
    generate_propellant_trade_plot,
    generate_shock_cfd_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nozzle-cfd",
        description="Multi-propellant rocket propulsion & supersonic nozzle CFD digital twin.",
    )
    parser.add_argument(
        "--propellant",
        default="METHALOX",
        choices=sorted(PROPELLANTS.keys()),
        help="Rocket propellant combination (default: METHALOX).",
    )
    parser.add_argument(
        "--chamber-pressure-mpa",
        type=float,
        default=10.0,
        help="Chamber stagnation pressure Pc in MPa (default: 10.0).",
    )
    parser.add_argument(
        "--ambient-pressure-kpa",
        type=float,
        default=101.325,
        help="Ambient back-pressure Pa in kPa (default: 101.325 sea-level).",
    )
    parser.add_argument(
        "--expansion-ratio",
        type=float,
        default=12.0,
        help="Nozzle supersonic area ratio Ae/At (default: 12.0).",
    )
    parser.add_argument(
        "--throat-radius-m",
        type=float,
        default=0.12,
        help="Nozzle throat radius Rt in meters (default: 0.12).",
    )
    parser.add_argument(
        "--material",
        default="GRCOP_42",
        choices=["GRCOP_42", "CUC_ZR", "INCONEL_718"],
        help="Chamber wall alloy for Bartz regenerative cooling analysis.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=141,
        help="Axial CFD grid points (default: 141).",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Generate VTK mesh, 3 Matplotlib figures, and JSON report in --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="deliverables",
        help="Directory for generated plots, VTK files, and reports (default: deliverables).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON summary to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    perf = evaluate_propellant_performance(
        args.propellant,
        chamber_pressure_mpa=args.chamber_pressure_mpa,
        ambient_pressure_kpa=args.ambient_pressure_kpa,
        expansion_ratio=args.expansion_ratio,
        throat_radius_m=args.throat_radius_m,
    )
    geom = build_rao_bell_nozzle(
        throat_radius_m=args.throat_radius_m,
        contraction_ratio=3.2,
        expansion_ratio=args.expansion_ratio,
        n_points=args.nx,
    )
    cfd = solve_quasi1d_maccormack(
        geom,
        propellant_key=args.propellant,
        chamber_pressure_mpa=args.chamber_pressure_mpa,
        ambient_pressure_kpa=args.ambient_pressure_kpa,
    )
    cooling = analyze_bartz_cooling(geom, cfd, material_key=args.material)
    plume = simulate_2d_nozzle_and_plume(geom, cfd)

    artifacts: list[str] = []
    if args.export_all:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Also run off-design in-nozzle normal shock validation benchmark case
        geom_shock = build_rao_bell_nozzle(
            throat_radius_m=0.10,
            contraction_ratio=3.0,
            expansion_ratio=8.0,
            n_points=121,
        )
        cfd_shock = solve_quasi1d_maccormack(
            geom_shock,
            propellant_key=args.propellant,
            chamber_pressure_mpa=1.0,
            ambient_pressure_kpa=380.0,
        )
        trade = run_propellant_trade_study(
            chamber_pressure_mpa=args.chamber_pressure_mpa,
            ambient_pressure_kpa=args.ambient_pressure_kpa,
            expansion_ratio=args.expansion_ratio,
            throat_radius_m=args.throat_radius_m,
        )

        vtk_p = export_plume_vtk(plume, out_dir / "nozzle_plume_field.vtk")
        shock_png = generate_shock_cfd_plot(
            geom_shock, cfd, cfd_shock, cooling, out_dir / "nozzle_shock_cfd_analysis.png"
        )
        trade_png = generate_propellant_trade_plot(
            trade, out_dir / "multi_propellant_trade_study.png"
        )
        plume_png = generate_2d_plume_plot(geom, plume, out_dir / "plume_mach_diamonds_2d.png")

        report = {
            "performance": perf.to_dict(),
            "cfd_design": {
                "flow_regime": cfd.flow_regime,
                "iterations": cfd.iterations,
                "converged": cfd.converged,
                "exit_mach": round(float(cfd.mach[-1]), 4),
                "analytical_exit_mach": round(float(cfd.analytical_mach[-1]), 4),
                "mach_l2_error_pct": cfd.mach_l2_error_pct,
                "mass_conservation_error_pct": cfd.mass_conservation_error_pct,
            },
            "cfd_in_nozzle_shock_benchmark": {
                "flow_regime": cfd_shock.flow_regime,
                "iterations": cfd_shock.iterations,
                "converged": cfd_shock.converged,
                "shock_detected": cfd_shock.shock_detected,
                "shock_x_m": cfd_shock.shock_x_m,
                "shock_area_ratio": cfd_shock.shock_area_ratio,
                "shock_upstream_mach": cfd_shock.shock_upstream_mach,
                "shock_downstream_mach": cfd_shock.shock_downstream_mach,
                "mach_l2_error_pct": cfd_shock.mach_l2_error_pct,
            },
            "cooling": {
                "material": cooling.wall_material,
                "peak_heat_flux_mw_m2": cooling.peak_heat_flux_mw_m2,
                "peak_hot_wall_temp_k": cooling.peak_hot_wall_temp_k,
                "max_allowable_temp_k": cooling.max_allowable_temp_k,
                "thermal_margin_k": cooling.thermal_margin_k,
                "wall_survives": cooling.wall_survives,
                "total_heat_load_mw": cooling.total_heat_load_mw,
            },
            "plume_2d": {
                "grid_shape": list(plume.mach_2d.shape),
                "shock_diamond_wavelength_m": plume.shock_diamond_wavelength_m,
                "fully_expanded_mach": plume.fully_expanded_mach,
                "prandtl_meyer_exit_deg": plume.prandtl_meyer_exit_deg,
            },
            "trade_study": [item.to_dict() for item in trade],
        }
        report_path = out_dir / "nozzle_cfd_benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        artifacts = [str(p) for p in (vtk_p, shock_png, trade_png, plume_png, report_path)]

    summary = {
        "performance": perf.to_dict(),
        "cfd_summary": {
            "flow_regime": cfd.flow_regime,
            "iterations": cfd.iterations,
            "converged": cfd.converged,
            "exit_mach": round(float(cfd.mach[-1]), 4),
            "mach_l2_error_pct": cfd.mach_l2_error_pct,
            "mass_conservation_error_pct": cfd.mass_conservation_error_pct,
            "shock_detected": cfd.shock_detected,
        },
        "cooling_summary": {
            "material": cooling.wall_material,
            "peak_heat_flux_mw_m2": cooling.peak_heat_flux_mw_m2,
            "peak_hot_wall_temp_k": cooling.peak_hot_wall_temp_k,
            "wall_survives": cooling.wall_survives,
        },
        "plume_summary": {
            "shock_diamond_wavelength_m": plume.shock_diamond_wavelength_m,
            "fully_expanded_mach": plume.fully_expanded_mach,
        },
        "artifacts": artifacts,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"[{perf.propellant}] Tc={perf.tc_k:.0f} K | c*={perf.c_star_actual_m_s:.1f} m/s | "
            f"Isp_SL={perf.isp_s:.1f} s | Isp_Vac={perf.isp_vacuum_s:.1f} s | "
            f"Thrust={perf.thrust_kn:.1f} kN"
        )
        print(
            f"[MacCormack CFD] regime={cfd.flow_regime} | iters={cfd.iterations} | "
            f"M_exit={cfd.mach[-1]:.3f} | L2_err={cfd.mach_l2_error_pct:.3f}%"
        )
        print(
            f"[Bartz Cooling] {cooling.wall_material}: "
            f"peak_q={cooling.peak_heat_flux_mw_m2:.2f} MW/m2 | "
            f"peak_Twg={cooling.peak_hot_wall_temp_k:.1f} K | survives={cooling.wall_survives}"
        )
        if artifacts:
            print("Generated artifacts:", ", ".join(artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
