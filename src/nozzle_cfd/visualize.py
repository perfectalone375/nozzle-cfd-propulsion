"""Visualization suite: high-DPI Matplotlib engineering charts & ParaView VTK export."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nozzle_cfd.cfd_solver import NozzleCFDResult
from nozzle_cfd.cooling import CoolingAnalysisResult
from nozzle_cfd.geometry import NozzleGeometry
from nozzle_cfd.moc_2d import Plume2DResult
from nozzle_cfd.propellants import PropellantPerformance


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
