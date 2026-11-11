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
from nozzle_cfd.navier_stokes_2d import (
    build_conical_nozzle_geometry,
    export_axisymmetric_vts,
    solve_axisymmetric_navier_stokes_2d,
)
from nozzle_cfd.propellants import (
    PROPELLANTS,
    evaluate_propellant_performance,
    run_propellant_trade_study,
)
from nozzle_cfd.riemann_fvm import (
    run_grid_convergence_study,
    solve_quasi1d_fvm,
    solve_sod_shock_tube,
)
from nozzle_cfd.visualize import (
    export_plume_vtk,
    generate_2d_ns_schlieren_plot,
    generate_2d_plume_plot,
    generate_fvm_riemann_plot,
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
        "--solver",
        default="fvm_hllc",
        choices=["fvm_hllc", "fvm_ausm", "maccormack"],
        help="Primary 1D nozzle CFD scheme (default: fvm_hllc).",
    )
    parser.add_argument(
        "--limiter",
        default="van_leer",
        choices=["first_order", "minmod", "van_leer", "superbee", "mc"],
        help="MUSCL TVD slope limiter for Finite-Volume Riemann solvers (default: van_leer).",
    )
    parser.add_argument(
        "--viscous-mode",
        default="euler",
        choices=["euler", "laminar", "turbulent"],
        help="2D Axisymmetric Navier-Stokes viscosity & turbulence closure (default: euler).",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Generate VTK/VTS meshes, Matplotlib figures, and JSON report in --output-dir.",
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
    cfd_mac = solve_quasi1d_maccormack(
        geom,
        propellant_key=args.propellant,
        chamber_pressure_mpa=args.chamber_pressure_mpa,
        ambient_pressure_kpa=args.ambient_pressure_kpa,
    )
    if args.solver in ("fvm_hllc", "fvm_ausm"):
        scheme = "ausm_plus" if args.solver == "fvm_ausm" else "hllc"
        cfd = solve_quasi1d_fvm(
            geom,
            propellant_key=args.propellant,
            chamber_pressure_mpa=args.chamber_pressure_mpa,
            ambient_pressure_kpa=args.ambient_pressure_kpa,
            flux_scheme=scheme,
            limiter=args.limiter,
        )
    else:
        cfd = cfd_mac

    cooling = analyze_bartz_cooling(geom, cfd_mac, material_key=args.material)
    plume = simulate_2d_nozzle_and_plume(geom, cfd_mac, viscous_mode=args.viscous_mode)

    artifacts: list[str] = []
    ns2d_jpl = None
    ns2d_sep = None
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
        fvm_shock = solve_quasi1d_fvm(
            geom_shock,
            propellant_key=args.propellant,
            chamber_pressure_mpa=1.0,
            ambient_pressure_kpa=380.0,
            flux_scheme="hllc",
            limiter=args.limiter,
        )
        sod_first = solve_sod_shock_tube(nx=200, flux_scheme="rusanov", limiter="first_order")
        sod_hllc = solve_sod_shock_tube(nx=200, flux_scheme="hllc", limiter=args.limiter)
        gci = run_grid_convergence_study(
            propellant_key=args.propellant,
            chamber_pressure_mpa=args.chamber_pressure_mpa,
            ambient_pressure_kpa=args.ambient_pressure_kpa,
            expansion_ratio=args.expansion_ratio,
            throat_radius_m=args.throat_radius_m,
            flux_scheme="hllc",
            limiter=args.limiter,
            grid_sizes=(81, 161, 321),
        )
        trade = run_propellant_trade_study(
            chamber_pressure_mpa=args.chamber_pressure_mpa,
            ambient_pressure_kpa=args.ambient_pressure_kpa,
            expansion_ratio=args.expansion_ratio,
            throat_radius_m=args.throat_radius_m,
        )

        # Step 2 2D Axisymmetric Navier-Stokes validations (JPL 45-15 & overexpanded separation)
        jpl_geom = build_conical_nozzle_geometry(
            throat_radius_m=0.05,
            contraction_ratio=3.0,
            expansion_ratio=4.0,
            conv_half_angle_deg=45.0,
            div_half_angle_deg=15.0,
            throat_rc_ratio=0.625,
            n_points=121,
        )
        ns2d_jpl = solve_axisymmetric_navier_stokes_2d(
            jpl_geom,
            propellant_key="COLD_GAS_N2",
            chamber_pressure_mpa=1.0,
            ambient_pressure_kpa=2.0,
            viscous_mode="euler",
            limiter=args.limiter,
            nx1=84,
            nr1=30,
            include_plume=False,
            max_iters=2400,
        )
        ns2d_sep = solve_axisymmetric_navier_stokes_2d(
            geom,
            propellant_key=args.propellant,
            chamber_pressure_mpa=1.2,
            ambient_pressure_kpa=320.0,
            viscous_mode="turbulent",
            limiter=args.limiter,
            nx1=84,
            nr1=28,
            include_plume=False,
            max_iters=2400,
        )

        vtk_p = export_plume_vtk(plume, out_dir / "nozzle_plume_field.vtk")
        vts_paths: list[Path] = []
        if plume.ns2d_result is not None:
            vts_int, vts_plume = export_axisymmetric_vts(
                plume.ns2d_result, out_dir, stem="nozzle_ns2d"
            )
            vts_paths.append(vts_int)
            if vts_plume is not None:
                vts_paths.append(vts_plume)
        shock_png = generate_shock_cfd_plot(
            geom_shock, cfd_mac, cfd_shock, cooling, out_dir / "nozzle_shock_cfd_analysis.png"
        )
        fvm_png = generate_fvm_riemann_plot(
            sod_first, sod_hllc, cfd_shock, fvm_shock, gci, out_dir / "fvm_riemann_validation.png"
        )
        trade_png = generate_propellant_trade_plot(
            trade, out_dir / "multi_propellant_trade_study.png"
        )
        plume_png = generate_2d_plume_plot(geom, plume, out_dir / "plume_mach_diamonds_2d.png")
        schlieren_png = generate_2d_ns_schlieren_plot(
            plume.ns2d_result, ns2d_jpl, ns2d_sep, out_dir / "ns2d_schlieren_and_jpl_validation.png"
        )

        report = {
            "performance": perf.to_dict(),
            "cfd_design": {
                "solver": getattr(cfd, "flux_scheme", "maccormack"),
                "backend": getattr(cfd, "solver_backend", "python_numpy"),
                "solve_ms": getattr(cfd, "solve_ms", None),
                "flow_regime": cfd.flow_regime,
                "iterations": cfd.iterations,
                "converged": cfd.converged,
                "exit_mach": round(float(cfd.mach[-1]), 4),
                "analytical_exit_mach": round(float(cfd.analytical_mach[-1]), 4),
                "mach_l2_error_pct": cfd.mach_l2_error_pct,
                "mass_conservation_error_pct": cfd.mass_conservation_error_pct,
            },
            "cfd_in_nozzle_shock_benchmark": {
                "maccormack": {
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
                "fvm_hllc_muscl": {
                    "backend": fvm_shock.solver_backend,
                    "solve_ms": fvm_shock.solve_ms,
                    "flow_regime": fvm_shock.flow_regime,
                    "iterations": fvm_shock.iterations,
                    "converged": fvm_shock.converged,
                    "shock_detected": fvm_shock.shock_detected,
                    "shock_x_m": fvm_shock.shock_x_m,
                    "shock_upstream_mach": fvm_shock.shock_upstream_mach,
                    "shock_downstream_mach": fvm_shock.shock_downstream_mach,
                    "mach_l2_error_pct": fvm_shock.mach_l2_error_pct,
                    "mass_conservation_error_pct": fvm_shock.mass_conservation_error_pct,
                },
            },
            "sod_shock_tube_benchmark": {
                "first_order_rusanov_rho_l1": sod_first.rho_l1_error,
                "muscl_hllc_rho_l1": sod_hllc.rho_l1_error,
                "muscl_hllc_rho_l2": sod_hllc.rho_l2_error,
                "muscl_hllc_p_l2": sod_hllc.p_l2_error,
            },
            "grid_convergence_gci": gci.to_dict(),
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
                "solver_backend": plume.solver_backend,
                "solve_ms": round(plume.solve_ms, 2),
                "discharge_coefficient_cd": plume.discharge_coefficient_cd,
                "mach_disc_detected": plume.mach_disc_detected,
                "separation_detected": plume.separation_detected,
                "shock_diamond_wavelength_m": plume.shock_diamond_wavelength_m,
                "fully_expanded_mach": plume.fully_expanded_mach,
                "prandtl_meyer_exit_deg": plume.prandtl_meyer_exit_deg,
            },
            "2d_navier_stokes_benchmark": {
                "coupled_plume": {
                    "backend": plume.solver_backend,
                    "n_threads": plume.ns2d_result.n_threads if plume.ns2d_result else 1,
                    "solve_ms": round(plume.solve_ms, 2),
                    "mass_conservation_error_pct": (
                        round(plume.ns2d_result.mass_conservation_error_pct, 5)
                        if plume.ns2d_result
                        else None
                    ),
                    "discharge_coefficient_cd": plume.discharge_coefficient_cd,
                    "sonic_line_offset_mm": (
                        round(plume.ns2d_result.sonic_line_offset_mm, 3)
                        if plume.ns2d_result
                        else None
                    ),
                },
                "jpl_45_15_validation": {
                    "backend": ns2d_jpl.solver_backend,
                    "converged": ns2d_jpl.converged,
                    "solve_ms": round(ns2d_jpl.solve_ms, 2),
                    "discharge_coefficient_cd": round(ns2d_jpl.discharge_coefficient_cd, 5),
                    "sonic_line_offset_mm": round(ns2d_jpl.sonic_line_offset_mm, 3),
                    "mass_conservation_error_pct": round(ns2d_jpl.mass_conservation_error_pct, 5),
                },
                "viscous_turbulent_separation": {
                    "backend": ns2d_sep.solver_backend,
                    "viscous_mode": ns2d_sep.viscous_mode,
                    "separation_detected": ns2d_sep.separation_detected,
                    "separation_x_m": (
                        round(ns2d_sep.separation_x_m, 4)
                        if ns2d_sep.separation_x_m is not None
                        else None
                    ),
                    "min_wall_velocity_m_s": round(float(ns2d_sep.u_1[:, -1].min()), 2),
                },
            },
            "trade_study": [item.to_dict() for item in trade],
        }
        report_path = out_dir / "nozzle_cfd_benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        artifacts = [
            str(p)
            for p in (
                vtk_p,
                *vts_paths,
                shock_png,
                fvm_png,
                trade_png,
                plume_png,
                schlieren_png,
                report_path,
            )
        ]

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
            "solver_backend": plume.solver_backend,
            "discharge_coefficient_cd": plume.discharge_coefficient_cd,
            "shock_diamond_wavelength_m": plume.shock_diamond_wavelength_m,
            "fully_expanded_mach": plume.fully_expanded_mach,
            "mach_disc_detected": plume.mach_disc_detected,
            "separation_detected": plume.separation_detected,
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
            f"[2D Axisymmetric NS/Plume] backend={plume.solver_backend} | "
            f"Cd={plume.discharge_coefficient_cd:.4f} | "
            f"L_cell={plume.shock_diamond_wavelength_m:.3f} m | time={plume.solve_ms:.0f} ms"
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
