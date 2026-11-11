"""Model Context Protocol (MCP) JSON-RPC 2.0 server for nozzle-cfd-propulsion."""

from __future__ import annotations

import json
import sys
from typing import Any

from nozzle_cfd import __version__
from nozzle_cfd.cfd_solver import solve_quasi1d_maccormack
from nozzle_cfd.cooling import analyze_bartz_cooling
from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.moc_2d import simulate_2d_nozzle_and_plume
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

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "nozzle-cfd-mcp", "version": __version__}

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "simulate_nozzle_cfd",
        "description": (
            "Run Conservative Quasi-1D MacCormack CFD + Rao Bell geometry + regime analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "propellant": {"type": "string", "enum": sorted(PROPELLANTS.keys())},
                "chamber_pressure_mpa": {"type": "number", "default": 10.0},
                "ambient_pressure_kpa": {"type": "number", "default": 101.325},
                "expansion_ratio": {"type": "number", "default": 12.0},
                "throat_radius_m": {"type": "number", "default": 0.12},
            },
        },
    },
    {
        "name": "compare_propellants",
        "description": "Compare Hydrolox, Methalox, Kerolox, and Solid APCP thrust, c*, and Isp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chamber_pressure_mpa": {"type": "number", "default": 10.0},
                "ambient_pressure_kpa": {"type": "number", "default": 101.325},
                "expansion_ratio": {"type": "number", "default": 16.0},
            },
        },
    },
    {
        "name": "analyze_nozzle_cooling",
        "description": "Compute Bartz convective heat flux and regenerative liner survival margin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "propellant": {"type": "string", "default": "METHALOX"},
                "chamber_pressure_mpa": {"type": "number", "default": 10.0},
                "material": {"type": "string", "default": "GRCOP_42"},
            },
        },
    },
    {
        "name": "simulate_2d_plume",
        "description": (
            "Compute 2D Method of Characteristics rays and plume shock-diamond wavelength."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "propellant": {"type": "string", "default": "METHALOX"},
                "chamber_pressure_mpa": {"type": "number", "default": 10.0},
                "ambient_pressure_kpa": {"type": "number", "default": 101.325},
                "expansion_ratio": {"type": "number", "default": 12.0},
            },
        },
    },
    {
        "name": "run_fvm_riemann_benchmark",
        "description": (
            "Run Finite-Volume HLLC+/AUSM+ MUSCL-TVD CFD benchmark on Toro's Sod Shock Tube "
            "or an overexpanded standing-shock rocket nozzle (auto-uses C++20 DLL if present)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "benchmark": {"type": "string", "enum": ["sod", "nozzle_shock"], "default": "sod"},
                "flux_scheme": {
                    "type": "string",
                    "enum": ["hllc", "ausm_plus", "rusanov"],
                    "default": "hllc",
                },
                "limiter": {
                    "type": "string",
                    "enum": ["first_order", "minmod", "van_leer", "superbee", "mc"],
                    "default": "van_leer",
                },
                "nx": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "run_grid_convergence_study",
        "description": (
            "Perform formal 3-grid ASME V&V 20 / Roache Grid Convergence Index (GCI) verification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "propellant": {"type": "string", "default": "METHALOX"},
                "expansion_ratio": {"type": "number", "default": 12.0},
                "flux_scheme": {"type": "string", "enum": ["hllc", "ausm_plus"], "default": "hllc"},
                "limiter": {
                    "type": "string",
                    "enum": ["minmod", "van_leer"],
                    "default": "van_leer",
                },
            },
        },
    },
]

MCP_RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "nozzle://propellants/database",
        "name": "Rocket Propellant Thermochemistry Catalog",
        "mimeType": "application/json",
    }
]


def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "simulate_nozzle_cfd":
        prop = str(args.get("propellant", "METHALOX"))
        pc = float(args.get("chamber_pressure_mpa", 10.0))
        pa = float(args.get("ambient_pressure_kpa", 101.325))
        eps = float(args.get("expansion_ratio", 12.0))
        rt = float(args.get("throat_radius_m", 0.12))
        perf = evaluate_propellant_performance(prop, pc, pa, eps, rt)
        geom = build_rao_bell_nozzle(throat_radius_m=rt, expansion_ratio=eps, n_points=101)
        cfd = solve_quasi1d_maccormack(geom, prop, pc, pa)
        return {
            "performance": perf.to_dict(),
            "cfd": {
                "flow_regime": cfd.flow_regime,
                "converged": cfd.converged,
                "iterations": cfd.iterations,
                "exit_mach": round(float(cfd.mach[-1]), 4),
                "shock_detected": cfd.shock_detected,
                "shock_x_m": cfd.shock_x_m,
                "mach_l2_error_pct": cfd.mach_l2_error_pct,
            },
        }
    if name == "compare_propellants":
        pc = float(args.get("chamber_pressure_mpa", 10.0))
        pa = float(args.get("ambient_pressure_kpa", 101.325))
        eps = float(args.get("expansion_ratio", 16.0))
        return {"results": [item.to_dict() for item in run_propellant_trade_study(pc, pa, eps)]}
    if name == "analyze_nozzle_cooling":
        prop = str(args.get("propellant", "METHALOX"))
        pc = float(args.get("chamber_pressure_mpa", 10.0))
        mat = str(args.get("material", "GRCOP_42"))
        geom = build_rao_bell_nozzle(n_points=101)
        cfd = solve_quasi1d_maccormack(geom, prop, pc, 101.325)
        cool = analyze_bartz_cooling(geom, cfd, material_key=mat)
        return {
            "propellant": cool.propellant,
            "material": cool.wall_material,
            "peak_heat_flux_mw_m2": cool.peak_heat_flux_mw_m2,
            "peak_hot_wall_temp_k": cool.peak_hot_wall_temp_k,
            "thermal_margin_k": cool.thermal_margin_k,
            "wall_survives": cool.wall_survives,
            "total_heat_load_mw": cool.total_heat_load_mw,
        }
    if name == "simulate_2d_plume":
        prop = str(args.get("propellant", "METHALOX"))
        pc = float(args.get("chamber_pressure_mpa", 10.0))
        pa = float(args.get("ambient_pressure_kpa", 101.325))
        eps = float(args.get("expansion_ratio", 12.0))
        geom = build_rao_bell_nozzle(expansion_ratio=eps, n_points=101)
        cfd = solve_quasi1d_maccormack(geom, prop, pc, pa)
        plume = simulate_2d_nozzle_and_plume(geom, cfd)
        return {
            "grid_shape": list(plume.mach_2d.shape),
            "shock_diamond_wavelength_m": plume.shock_diamond_wavelength_m,
            "fully_expanded_mach": plume.fully_expanded_mach,
            "prandtl_meyer_exit_deg": plume.prandtl_meyer_exit_deg,
        }
    if name == "run_fvm_riemann_benchmark":
        bench = str(args.get("benchmark", "sod")).lower()
        flux = str(args.get("flux_scheme", "hllc")).lower()
        lim = str(args.get("limiter", "van_leer")).lower()
        nx = int(args.get("nx", 200))
        if bench == "sod":
            res = solve_sod_shock_tube(nx=nx, flux_scheme=flux, limiter=lim)
            return {
                "benchmark": "sod",
                "flux_scheme": res.flux_scheme,
                "limiter": res.limiter,
                "nx": res.nx,
                "rho_l1_error": res.rho_l1_error,
                "rho_l2_error": res.rho_l2_error,
                "p_l2_error": res.p_l2_error,
                "solve_ms": res.solve_ms,
            }
        geom = build_rao_bell_nozzle(
            throat_radius_m=0.10,
            contraction_ratio=3.0,
            expansion_ratio=8.0,
            n_points=min(max(nx, 81), 321),
        )
        fvm = solve_quasi1d_fvm(geom, "METHALOX", 1.0, 380.0, flux_scheme=flux, limiter=lim)
        return {
            "benchmark": "nozzle_shock",
            "solver_backend": fvm.solver_backend,
            "flux_scheme": fvm.flux_scheme,
            "limiter": fvm.limiter,
            "converged": fvm.converged,
            "iterations": fvm.iterations,
            "shock_detected": fvm.shock_detected,
            "shock_upstream_mach": fvm.shock_upstream_mach,
            "shock_downstream_mach": fvm.shock_downstream_mach,
            "mach_l2_error_pct": fvm.mach_l2_error_pct,
            "solve_ms": fvm.solve_ms,
        }
    if name == "run_grid_convergence_study":
        prop = str(args.get("propellant", "METHALOX"))
        eps = float(args.get("expansion_ratio", 12.0))
        flux = str(args.get("flux_scheme", "hllc"))
        lim = str(args.get("limiter", "van_leer"))
        gci = run_grid_convergence_study(
            propellant_key=prop,
            expansion_ratio=eps,
            flux_scheme=flux,
            limiter=lim,
            grid_sizes=(81, 161, 321),
        )
        return gci.to_dict()
    raise ValueError(f"Unknown MCP tool: {name}")


def handle_jsonrpc_request(req: dict[str, Any]) -> dict[str, Any] | None:
    req_id = req.get("id")
    method = str(req.get("method", ""))
    params = req.get("params") or {}

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": MCP_RESOURCES}}
    if method == "resources/read":
        uri = str(params.get("uri", ""))
        if uri == "nozzle://propellants/database":
            payload = {k: v.display_name for k, v in PROPELLANTS.items()}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {"uri": uri, "mimeType": "application/json", "text": json.dumps(payload)}
                    ]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32002, "message": "Resource not found"},
        }
    if method == "tools/call":
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        try:
            res = _call_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True},
            }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue
        reply = handle_jsonrpc_request(payload)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
