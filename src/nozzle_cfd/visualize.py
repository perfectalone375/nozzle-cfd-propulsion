"""Visualization suite: high-DPI Matplotlib engineering charts & ParaView VTK export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nozzle_cfd.cfd_solver import NozzleCFDResult
from nozzle_cfd.cooling import CoolingAnalysisResult
from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.moc_2d import Plume2DResult
from nozzle_cfd.propellants import PropellantPerformance
from nozzle_cfd.riemann_fvm import GridConvergenceResult, NozzleFVMResult, SodShockTubeResult


def export_plume_vtk(plume: Plume2DResult, output_path: str | Path) -> Path:
    """Export 2D (x, r) CFD + exhaust plume flowfield as a ParaView Legacy ASCII VTK mesh."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nr, nx = plume.mach_2d.shape
    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "Nozzle CFD + Supersonic Exhaust Plume Shock Diamond Digital Twin",
        "ASCII",
        "DATASET STRUCTURED_GRID",
        f"DIMENSIONS {nx} {nr} 1",
        f"POINTS {nx * nr} float",
    ]

    for ir in range(nr):
        r_val = float(plume.r_m[ir])
        for ix in range(nx):
            x_val = float(plume.x_m[ix])
            lines.append(f"{x_val:.6f} {r_val:.6f} 0.000000")

    def append_scalar(name: str, field: np.ndarray, fill_val: float) -> None:
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        for ir in range(nr):
            for ix in range(nx):
                val = float(field[ir, ix])
                if not np.isfinite(val):
                    val = fill_val
                lines.append(f"{val:.5f}")

    lines.append(f"POINT_DATA {nx * nr}")
    append_scalar("Mach_Number", plume.mach_2d, 0.0)
    append_scalar("Static_Pressure_kPa", plume.pressure_kpa_2d, 101.325)
    append_scalar("Static_Temperature_K", plume.temperature_k_2d, 288.15)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def generate_shock_cfd_plot(
    geometry: NozzleGeometry,
    cfd_design: NozzleCFDResult,
    cfd_shock: NozzleCFDResult,
    cooling: CoolingAnalysisResult,
    output_path: str | Path,
) -> Path:
    """Create a 2x2 publication-grade Matplotlib CFD & thermal validation figure."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=180)
    fig.patch.set_facecolor("#0b1120")

    # 1. Rao Parabolic Bell Nozzle Contour & Expansion Envelope
    ax1 = axes[0, 0]
    ax1.set_facecolor("#111827")
    ax1.plot(
        geometry.x_m, geometry.radius_m * 1000.0, color="#38bdf8", lw=2.5, label="Rao 80% Bell Wall"
    )
    ax1.plot(geometry.x_m, -geometry.radius_m * 1000.0, color="#38bdf8", lw=2.5)
    ax1.fill_between(
        geometry.x_m,
        -geometry.radius_m * 1000.0,
        geometry.radius_m * 1000.0,
        color="#1e3a8a",
        alpha=0.28,
    )
    ax1.axvline(
        geometry.throat_x_m,
        color="#f59e0b",
        ls="--",
        lw=1.4,
        label=f"Throat (r={geometry.throat_radius_m * 1000:.0f} mm)",
    )
    if cfd_shock.shock_detected and cfd_shock.shock_x_m is not None:
        ax1.axvline(
            cfd_shock.shock_x_m,
            color="#ef4444",
            ls=":",
            lw=1.8,
            label=f"Normal Shock @ x={cfd_shock.shock_x_m:.2f} m",
        )
    ax1.set_title(
        "1. Axisymmetric Rao Parabolic Bell Contour",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax1.set_xlabel("Axial Position x [m]")
    ax1.set_ylabel("Radius r [mm]")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.2)

    # 2. MacCormack CFD vs Exact Rankine-Hugoniot Shock Solution (Mach Number)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#111827")
    ax2.plot(
        cfd_design.x_m,
        cfd_design.mach,
        color="#22d3ee",
        lw=2.2,
        label=f"CFD Design Supersonic (L2={cfd_design.mach_l2_error_pct:.2f}%)",
    )
    ax2.plot(
        cfd_design.x_m,
        cfd_design.analytical_mach,
        color="#a5f3fc",
        ls="--",
        lw=1.3,
        label="Exact Isentropic Analytical",
    )
    ax2.plot(
        cfd_shock.x_m,
        cfd_shock.mach,
        color="#f43f5e",
        lw=2.3,
        label=(
            f"CFD Normal Shock ({cfd_shock.shock_upstream_mach}->{cfd_shock.shock_downstream_mach})"
        ),
    )
    ax2.plot(
        cfd_shock.x_m,
        cfd_shock.analytical_mach,
        color="#fda4af",
        ls=":",
        lw=1.6,
        label="Exact Rankine-Hugoniot Shock",
    )
    ax2.axhline(1.0, color="#fbbf24", ls="-.", lw=1.0, alpha=0.7)
    ax2.set_title(
        "2. MacCormack Shock-Capturing vs Analytical Truth",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax2.set_xlabel("Axial Position x [m]")
    ax2.set_ylabel("Mach Number M [-]")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.2)

    # 3. Bartz Convective Heat Flux & Hot-Gas Wall Temperature
    ax3 = axes[1, 0]
    ax3.set_facecolor("#111827")
    l1 = ax3.plot(
        cooling.x_m,
        cooling.heat_flux_mw_m2,
        color="#fb923c",
        lw=2.2,
        label=f"Bartz Heat Flux (Peak={cooling.peak_heat_flux_mw_m2:.1f} MW/m2)",
    )
    ax3.set_xlabel("Axial Position x [m]")
    ax3.set_ylabel("Heat Flux q'' [MW/m2]", color="#fb923c")
    ax3.tick_params(axis="y", labelcolor="#fb923c")
    ax3.grid(alpha=0.2)

    ax3_r = ax3.twinx()
    l2 = ax3_r.plot(
        cooling.x_m,
        cooling.hot_wall_temp_k,
        color="#34d399",
        lw=2.0,
        label=f"Inner Wall Twg (Max={cooling.peak_hot_wall_temp_k:.0f} K)",
    )
    l3 = ax3_r.axhline(
        cooling.max_allowable_temp_k,
        color="#ef4444",
        ls="--",
        lw=1.3,
        label=f"GRCop-42 Limit ({cooling.max_allowable_temp_k:.0f} K)",
    )
    ax3_r.set_ylabel("Wall Temperature [K]", color="#34d399")
    ax3_r.tick_params(axis="y", labelcolor="#34d399")
    ax3.set_ylim(0.0, float(np.max(cooling.heat_flux_mw_m2)) * 1.32)
    ax3_r.set_ylim(150.0, cooling.max_allowable_temp_k * 1.25)
    lines = l1 + l2 + [l3]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc="upper right", fontsize=8, framealpha=0.88)
    ax3.set_title(
        "3. Bartz Nozzle Thermal Load & Regenerative Cooling",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )

    # 4. Pressure Ratio Expansion & Shock Jump Profile
    ax4 = axes[1, 1]
    ax4.set_facecolor("#111827")
    ax4.semilogy(
        cfd_design.x_m,
        cfd_design.pressure_kpa,
        color="#38bdf8",
        lw=2.2,
        label="Design Supersonic P(x) [10 MPa]",
    )
    ax4.semilogy(
        cfd_shock.x_m,
        cfd_shock.pressure_kpa,
        color="#fb7185",
        lw=2.2,
        label="Shocked Nozzle P(x) [1 MPa -> 380 kPa]",
    )
    ax4.semilogy(
        cfd_shock.x_m,
        cfd_shock.analytical_pressure_kpa,
        color="#fecdd3",
        ls=":",
        lw=1.4,
        label="Analytical Shock Pressure",
    )
    ax4.set_title(
        "4. Static Pressure Expansion & Normal Shock Compression",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax4.set_xlabel("Axial Position x [m]")
    ax4.set_ylabel("Static Pressure [kPa] (log scale)")
    ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.88)
    ax4.grid(alpha=0.2, which="both")

    fig.suptitle(
        "Multi-Propellant Rocket Nozzle CFD Twin — Shock Capturing & Bartz Thermal Suite",
        fontsize=13.5,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def generate_propellant_trade_plot(
    trade_results: list[PropellantPerformance],
    output_path: str | Path,
) -> Path:
    """Plot Sea-Level vs Vacuum Specific Impulse and C* across all rocket propellants."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    fig.patch.set_facecolor("#0b1120")

    names = [item.propellant for item in trade_results]
    isp_sl = [item.isp_s for item in trade_results]
    isp_vac = [item.isp_vacuum_s for item in trade_results]
    c_star = [item.c_star_actual_m_s for item in trade_results]
    tc = [item.tc_k for item in trade_results]

    idx = np.arange(len(names))
    w = 0.36
    ax1.set_facecolor("#111827")
    b1 = ax1.bar(idx - w / 2, isp_sl, width=w, color="#38bdf8", label="Sea-Level Isp [s]")
    b2 = ax1.bar(idx + w / 2, isp_vac, width=w, color="#818cf8", label="Vacuum Isp [s]")
    ax1.bar_label(b1, fmt="%.1f", padding=3, fontsize=8.5, color="#e2e8f0")
    ax1.bar_label(b2, fmt="%.1f", padding=3, fontsize=8.5, color="#e2e8f0")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(names, fontweight="bold")
    ax1.set_ylabel("Specific Impulse Isp [s]")
    ax1.set_title("Specific Impulse (Pc = 10 MPa, eps = 16)", fontweight="bold", color="#f8fafc")
    ax1.set_ylim(0, max(isp_vac) * 1.18)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.2)

    ax2.set_facecolor("#111827")
    b3 = ax2.bar(idx, c_star, width=0.5, color="#34d399", label="Characteristic Velocity c* [m/s]")
    ax2.bar_label(b3, fmt="%.0f", padding=3, fontsize=8.5, color="#e2e8f0")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(names, fontweight="bold")
    ax2.set_ylabel("Characteristic Velocity c* [m/s]", color="#34d399")
    ax2.set_ylim(0, max(c_star) * 1.18)
    ax2.set_title("Combustion Efficiency & Chamber Flame Temp", fontweight="bold", color="#f8fafc")
    ax2.grid(axis="y", alpha=0.2)

    ax2_r = ax2.twinx()
    ax2_r.plot(idx, tc, color="#f59e0b", marker="o", lw=2.2, label="Chamber Tc [K]")
    ax2_r.set_ylabel("Chamber Flame Temperature Tc [K]", color="#f59e0b")
    ax2_r.set_ylim(2800, max(tc) * 1.1)

    fig.suptitle(
        "Multi-Propellant Thermochemical Trade Study (Hydrolox vs Methalox vs Kerolox vs APCP)",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def generate_2d_plume_plot(
    geometry: NozzleGeometry,
    plume: Plume2DResult,
    output_path: str | Path,
) -> Path:
    """Render 2D axisymmetric supersonic nozzle + shock diamond Mach contour heatmap."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=180)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#020617")

    masked_mach = np.ma.masked_invalid(plume.mach_2d)
    x_grid, r_grid = np.meshgrid(plume.x_m, plume.r_m)
    pcm = ax.pcolormesh(x_grid, r_grid * 1000.0, masked_mach, cmap="inferno", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Local Mach Number M [-]")

    # Overlay physical titanium/copper nozzle hardware walls
    ax.plot(geometry.x_m, geometry.radius_m * 1000.0, color="#38bdf8", lw=2.6, label="Nozzle Wall")
    ax.plot(geometry.x_m, -geometry.radius_m * 1000.0, color="#38bdf8", lw=2.6)
    ax.axvline(
        geometry.total_length_m,
        color="#f8fafc",
        ls="--",
        lw=1.2,
        alpha=0.8,
        label="Nozzle Exit Plane",
    )

    # Overlay 2D MOC internal wave rays cleanly
    for idx_r, (rx, rr) in enumerate(plume.moc_rays[:6]):
        lbl = "2D MOC Characteristic Rays" if idx_r == 0 else None
        ax.plot(rx, np.asarray(rr) * 1000.0, color="#67e8f9", lw=0.8, alpha=0.55, label=lbl)
        ax.plot(rx, -np.asarray(rr) * 1000.0, color="#67e8f9", lw=0.8, alpha=0.55)

    ax.set_title(
        "2D Axisymmetric Supersonic Nozzle & Exhaust Plume Shock Diamonds "
        f"(L_cell={plume.shock_diamond_wavelength_m:.2f} m | "
        f"M_j={plume.fully_expanded_mach:.2f})",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
    )
    ax.set_xlabel("Axial Position x [m]")
    ax.set_ylabel("Radial Position r [mm]")
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def generate_fvm_riemann_plot(
    sod_first: SodShockTubeResult,
    sod_hllc: SodShockTubeResult,
    mac_shock: NozzleCFDResult,
    fvm_shock: NozzleFVMResult,
    gci: GridConvergenceResult,
    output_path: str | Path,
) -> Path:
    """Create a 2x2 beginner-friendly FVM Riemann + GCI verification comparison figure."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=180)
    fig.patch.set_facecolor("#0b1120")

    # 1. Sod Shock Tube: Why HLLC + Van Leer MUSCL beats 1st-Order Rusanov
    ax1 = axes[0, 0]
    ax1.set_facecolor("#111827")
    ax1.plot(sod_hllc.x, sod_hllc.rho_exact, color="#f8fafc", lw=2.2, label="Toro Exact Analytical")
    ax1.plot(
        sod_first.x,
        sod_first.rho_num,
        color="#f59e0b",
        ls="--",
        lw=1.7,
        label=f"1st-Order Rusanov (L1={sod_first.rho_l1_error:.4f})",
    )
    ax1.plot(
        sod_hllc.x,
        sod_hllc.rho_num,
        color="#22d3ee",
        lw=2.2,
        label=f"MUSCL HLLC+ ({sod_hllc.limiter}, L1={sod_hllc.rho_l1_error:.4f})",
    )
    ax1.annotate(
        "Contact Wave\n(HLLC keeps crisp jump)",
        xy=(0.685, 0.42),
        xytext=(0.71, 0.72),
        arrowprops={"arrowstyle": "->", "color": "#22d3ee", "lw": 1.2},
        fontsize=8,
        color="#a5f3fc",
    )
    ax1.set_title(
        "1. NASA Sod Shock Tube Riemann Benchmark (Density)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax1.set_xlabel("Tube Position x [-]")
    ax1.set_ylabel("Nondimensional Density rho [-]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2)

    # 2. Nozzle Standing Normal Shock: MacCormack vs HLLC+ MUSCL-TVD
    ax2 = axes[0, 1]
    ax2.set_facecolor("#111827")
    ax2.plot(
        fvm_shock.x_m,
        fvm_shock.analytical_mach,
        color="#f8fafc",
        ls=":",
        lw=1.8,
        label="Exact Rankine-Hugoniot Truth",
    )
    ax2.plot(
        mac_shock.x_m,
        mac_shock.mach,
        color="#fb7185",
        lw=1.8,
        alpha=0.9,
        label=f"1969 MacCormack FD (L2={mac_shock.mach_l2_error_pct:.2f}%)",
    )
    ax2.plot(
        fvm_shock.x_m,
        fvm_shock.mach,
        color="#34d399",
        lw=2.3,
        label=f"HLLC+ MUSCL-TVD FVM (L2={fvm_shock.mach_l2_error_pct:.2f}%)",
    )
    ax2.axhline(1.0, color="#fbbf24", ls="-.", lw=1.0, alpha=0.65)
    ax2.set_title(
        "2. Standing Normal Shock Crispness (FD vs Finite-Volume)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax2.set_xlabel("Nozzle Axial Position x [m]")
    ax2.set_ylabel("Mach Number M [-]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    # 3. ASME V&V 20 Grid Convergence Index (Log-Log Slope)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#111827")
    nx_arr = np.asarray(gci.grid_sizes, dtype=float)
    h_rel = nx_arr[0] / nx_arr
    ax3.loglog(
        h_rel,
        gci.mach_l2_errors_pct,
        marker="o",
        color="#38bdf8",
        lw=2.3,
        label=f"Mach L2 Error (p_obs={gci.observed_order_error:.2f})",
    )
    ax3.loglog(
        h_rel,
        gci.pressure_l2_errors_pct,
        marker="s",
        color="#a78bfa",
        lw=2.0,
        label="Pressure L2 Error",
    )
    ref_2nd = gci.mach_l2_errors_pct[0] * (h_rel**2)
    ax3.loglog(h_rel, ref_2nd, color="#94a3b8", ls="--", lw=1.3, label="Ideal 2nd-Order Slope (-2)")
    for idx_pt, (hx, err, nx_lbl) in enumerate(
        zip(h_rel, gci.mach_l2_errors_pct, gci.grid_sizes, strict=True)
    ):
        if idx_pt == 0:
            x_off, y_off = 8, -18
        elif idx_pt == len(h_rel) - 1:
            x_off, y_off = -46, 12
        else:
            x_off, y_off = 8, -14
        ax3.annotate(
            f"N={nx_lbl}\n{err:.3f}%",
            (hx, err),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=8,
        )
    ax3.invert_xaxis()
    ax3.set_xlim(h_rel[0] * 1.15, h_rel[-1] * 0.82)
    ax3.set_title(
        f"3. Roache GCI Mesh Refinement (GCI_fine={gci.gci_fine_pct:.3f}%)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax3.set_xlabel("Normalized Cell Size h/h_coarse (finer ->)")
    ax3.set_ylabel("Discretization L2 Error [%]")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(alpha=0.2, which="both")

    # 4. Iterative Residual Convergence History (MacCormack vs HLLC+ MUSCL-TVD)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#111827")
    if mac_shock.residual_history:
        it_mac = np.arange(len(mac_shock.residual_history)) * 50 + 1
        ax4.semilogy(
            it_mac,
            mac_shock.residual_history,
            color="#fb7185",
            lw=1.8,
            label=f"MacCormack FD (conv={mac_shock.converged}, iters={mac_shock.iterations})",
        )
    if fvm_shock.residual_history:
        it_fvm = np.arange(len(fvm_shock.residual_history)) * 50 + 1
        ax4.semilogy(
            it_fvm,
            fvm_shock.residual_history,
            color="#34d399",
            lw=2.1,
            label=f"HLLC+ MUSCL-TVD (conv={fvm_shock.converged}, iters={fvm_shock.iterations})",
        )
    ax4.axhline(1e-6, color="#fbbf24", ls="--", lw=1.1, label="Convergence Tolerance (1e-6)")
    ax4.set_title(
        f"4. Residual Convergence ({fvm_shock.solver_backend}: {fvm_shock.solve_ms:.1f} ms)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
    )
    ax4.set_xlabel("Iteration Count [-]")
    ax4.set_ylabel("Max Normalized Momentum Residual [-]")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(alpha=0.2, which="both")

    fig.suptitle(
        "Step 1 Validation: Finite-Volume HLLC+ Riemann Solver, MUSCL-TVD & ASME V&V 20 GCI",
        fontsize=13.5,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def generate_2d_ns_schlieren_plot(
    ns2d_plume: Any,
    ns2d_jpl: Any,
    ns2d_sep: Any,
    output_path: str | Path,
) -> Path:
    """Create a 2x2 showcase validating true 2D Axisymmetric Navier-Stokes & Numerical Schlieren."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), dpi=180)
    fig.patch.set_facecolor("#0b1120")

    def _plot_mirrored_blocks(
        ax: Any, res: Any, field1: np.ndarray, field2: np.ndarray | None, **kw: Any
    ) -> Any:
        mesh_top_1 = ax.pcolormesh(res.x1_v, res.r1_v, field1, shading="flat", **kw)
        ax.pcolormesh(res.x1_v, -res.r1_v, field1, shading="flat", **kw)
        if res.x2_v is not None and field2 is not None:
            ax.pcolormesh(res.x2_v, res.r2_v, field2, shading="flat", **kw)
            ax.pcolormesh(res.x2_v, -res.r2_v, field2, shading="flat", **kw)
        return mesh_top_1

    # 1. 2D Axisymmetric Mach Contours + Curved Sonic Line (Rao Nozzle + Coupled Plume)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#020617")
    vmax_m = float(
        max(
            3.5,
            np.nanmax(ns2d_plume.mach_1),
            np.nanmax(ns2d_plume.mach_2) if ns2d_plume.mach_2 is not None else 3.5,
        )
    )
    mappable1 = _plot_mirrored_blocks(
        ax1,
        ns2d_plume,
        ns2d_plume.mach_1,
        ns2d_plume.mach_2,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax_m,
    )
    # Nozzle physical walls & curved M=1 sonic line
    wx = ns2d_plume.x1_v[:, -1]
    wr = ns2d_plume.r1_v[:, -1]
    ax1.plot(wx, wr, color="#f8fafc", lw=2.2)
    ax1.plot(wx, -wr, color="#f8fafc", lw=2.2)
    sx = np.concatenate([ns2d_plume.sonic_line_x_m[::-1], ns2d_plume.sonic_line_x_m])
    sr = np.concatenate([-ns2d_plume.sonic_line_r_m[::-1], ns2d_plume.sonic_line_r_m])
    ax1.plot(sx, sr, color="#ffffff", ls="--", lw=1.6, label="Curved Sonic Line (M=1.0)")
    cb1 = fig.colorbar(mappable1, ax=ax1, pad=0.02, aspect=26)
    cb1.set_label("Local Mach Number M [-]", fontsize=8.5)
    ax1.set_title(
        f"1. 2D Axisymmetric Mach Field ({ns2d_plume.solver_backend}, "
        f"{ns2d_plume.solve_ms:.0f} ms, err={ns2d_plume.mass_conservation_error_pct:.3f}%)",
        fontsize=10.5,
        fontweight="bold",
        color="#f8fafc",
    )
    ax1.set_xlabel("Axial Position x [m]")
    ax1.set_ylabel("Radial Position r [m]")
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Optical Numerical Schlieren |grad(rho)|
    ax2 = axes[0, 1]
    ax2.set_facecolor("#e2e8f0")
    s1 = ns2d_plume.schlieren_1
    s2 = ns2d_plume.schlieren_2
    s_ref = float(np.percentile(s2 if s2 is not None else s1, 97) + 1e-6)
    sch_1_img = np.log1p(5.0 * s1 / s_ref)
    sch_2_img = np.log1p(5.0 * s2 / s_ref) if s2 is not None else None
    vmax_s = float(np.percentile(sch_2_img if sch_2_img is not None else sch_1_img, 99.5))
    mappable2 = _plot_mirrored_blocks(
        ax2,
        ns2d_plume,
        sch_1_img,
        sch_2_img,
        cmap="gray_r",
        vmin=0.0,
        vmax=max(0.5, vmax_s),
    )
    ax2.plot(wx, wr, color="#0f172a", lw=2.5)
    ax2.plot(wx, -wr, color="#0f172a", lw=2.5)
    cb2 = fig.colorbar(mappable2, ax=ax2, pad=0.02, aspect=26)
    cb2.set_label("Log Density Gradient ln(1 + 5|grad rho|/ref)", fontsize=8.5)
    ax2.set_title(
        "2. Numerical Schlieren: Oblique Nozzle Wave & Plume Shock Diamonds",
        fontsize=10.5,
        fontweight="bold",
        color="#f8fafc",
    )
    ax2.set_xlabel("Axial Position x [m]")
    ax2.set_ylabel("Radial Position r [m]")

    # 3. JPL 45-15 Transonic Throat Validation: Wall vs Centerline Acceleration
    ax3 = axes[1, 0]
    ax3.set_facecolor("#111827")
    xj = ns2d_jpl.x1_c[:, 0]
    m_cl = ns2d_jpl.mach_1[:, 0]
    m_wl = ns2d_jpl.mach_1[:, -1]
    i_th_j = int(np.argmin(ns2d_jpl.r1_v[:, -1]))
    x_th_j = float(ns2d_jpl.x1_v[i_th_j, -1])
    ax3.plot(xj, m_cl, color="#38bdf8", lw=2.2, label="Centerline Mach M(x, r=0)")
    ax3.plot(xj, m_wl, color="#fb923c", lw=2.2, label="Wall Mach M(x, r=R_wall)")
    ax3.axvline(
        x_th_j,
        color="#e2e8f0",
        ls=":",
        lw=1.2,
        label=f"Geometric Throat (Cd={ns2d_jpl.discharge_coefficient_cd:.4f})",
    )
    ax3.axhline(1.0, color="#fbbf24", ls="--", lw=1.1, alpha=0.8)
    ax3.annotate(
        f"Curved Sonic Line Shift\n(CL lags wall by {ns2d_jpl.sonic_line_offset_mm:.1f} mm)",
        xy=(x_th_j, float(m_wl[min(len(m_wl) - 1, i_th_j)])),
        xytext=(max(0.01, x_th_j - 0.055), 2.15),
        arrowprops={"arrowstyle": "->", "color": "#fb923c", "lw": 1.3},
        fontsize=8,
        color="#fed7aa",
    )
    ax3.set_title(
        "3. JPL 45-15 (rc/rt=0.625) 2D Transonic Throat Overshoot",
        fontsize=10.5,
        fontweight="bold",
        color="#f8fafc",
    )
    ax3.set_xlabel("Axial Position x [m]")
    ax3.set_ylabel("Mach Number M [-]")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.2)

    # 4. Viscous Navier-Stokes Boundary-Layer Shock-Induced Separation
    ax4 = axes[1, 1]
    ax4.set_facecolor("#111827")
    xs = ns2d_sep.x1_c[:, -1]
    u_wall = ns2d_sep.u_1[:, -1]
    u_core = ns2d_sep.u_1[:, 0]
    ax4.plot(xs, u_core, color="#34d399", lw=2.0, label="Supersonic Core Velocity u(x, r=0)")
    ax4.plot(
        xs, u_wall, color="#fb7185", lw=2.2, label="Near-Wall Turbulent BL Velocity u(x, wall)"
    )
    ax4.axhline(0.0, color="#fbbf24", ls="--", lw=1.1)
    ax4.fill_between(
        xs,
        u_wall,
        0.0,
        where=(u_wall < 0.0),
        color="#f43f5e",
        alpha=0.35,
        label="Reverse Flow Separation Bubble (u < 0)",
    )
    if ns2d_sep.separation_x_m is not None:
        ax4.axvline(
            ns2d_sep.separation_x_m,
            color="#fda4af",
            ls=":",
            lw=1.5,
            label=f"Separation Point x_sep={ns2d_sep.separation_x_m:.3f} m",
        )
    ax4.set_title(
        f"4. Viscous Turbulent BL & Separation (u_min={np.min(u_wall):.1f} m/s)",
        fontsize=10.2,
        fontweight="bold",
        color="#f8fafc",
    )
    ax4.set_xlabel("Nozzle Axial Position x [m]")
    ax4.set_ylabel("Axial Velocity u [m/s]")
    ax4.legend(loc="lower left", fontsize=8)
    ax4.grid(alpha=0.2)

    fig.suptitle(
        "Step 2 Validation: 2D Axisymmetric Navier-Stokes, Schlieren & Transonic Separation",
        fontsize=13.2,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
