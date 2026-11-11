"""Tests for multi-propellant rocket thermochemistry and Rao bell nozzle geometry."""

from __future__ import annotations

import numpy as np
import pytest

from nozzle_cfd.geometry import build_rao_bell_nozzle
from nozzle_cfd.propellants import (
    area_mach_function,
    evaluate_propellant_performance,
    mach_from_area_ratio,
    run_propellant_trade_study,
)


def test_area_mach_roundtrip_supersonic_and_subsonic() -> None:
    gamma = 1.21
    for target_eps in (1.5, 4.0, 12.0, 45.0):
        m_sup = mach_from_area_ratio(target_eps, gamma, supersonic=True)
        m_sub = mach_from_area_ratio(target_eps, gamma, supersonic=False)
        assert m_sup > 1.0
        assert 0.0 < m_sub < 1.0
        assert area_mach_function(m_sup, gamma) == pytest.approx(target_eps, rel=1e-6)
        assert area_mach_function(m_sub, gamma) == pytest.approx(target_eps, rel=1e-6)


def test_multi_propellant_trade_study_ranking() -> None:
    results = run_propellant_trade_study(
        chamber_pressure_mpa=10.0,
        ambient_pressure_kpa=101.325,
        expansion_ratio=16.0,
    )
    keys = [r.propellant for r in results]
    assert keys == ["HYDROLOX", "METHALOX", "KEROLOX", "SOLID_APCP"]
    # Vacuum Isp order must strictly reflect molecular weight / flame temperature physics
    isps = [r.isp_vacuum_s for r in results]
    assert isps[0] > isps[1] > isps[2] > isps[3]
    assert 390.0 <= results[0].isp_vacuum_s <= 430.0
    assert 310.0 <= results[1].isp_vacuum_s <= 350.0


def test_summerfield_flow_separation_detection() -> None:
    # Massive expansion at sea-level low chamber pressure triggers separation risk
    overexpanded = evaluate_propellant_performance(
        "METHALOX",
        chamber_pressure_mpa=2.0,
        ambient_pressure_kpa=101.325,
        expansion_ratio=35.0,
    )
    assert overexpanded.separation_risk is True
    assert overexpanded.expansion_regime == "OVEREXPANDED_SEPARATED_RISK"

    # Vacuum operation is safely underexpanded
    vacuum = evaluate_propellant_performance(
        "METHALOX",
        chamber_pressure_mpa=10.0,
        ambient_pressure_kpa=0.0,
        expansion_ratio=35.0,
    )
    assert vacuum.separation_risk is False
    assert vacuum.expansion_regime == "VACUUM_UNDEREXPANDED"


def test_rao_bell_nozzle_geometry_continuity_and_throat() -> None:
    geom = build_rao_bell_nozzle(
        throat_radius_m=0.12,
        contraction_ratio=3.2,
        expansion_ratio=16.0,
        bell_percent=80.0,
        n_points=141,
    )
    assert geom.x_m.shape == (141,)
    assert np.all(np.diff(geom.x_m) > 0.0)
    throat_idx = int(np.argmin(geom.area_ratio))
    assert geom.area_ratio[throat_idx] == pytest.approx(1.0, abs=1e-3)
    assert geom.area_ratio[-1] == pytest.approx(16.0, rel=1e-2)
    assert 0.97 <= geom.divergence_efficiency <= 0.996
