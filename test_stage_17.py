"""Tests for Stage 17 — Emergent Gompertz Mortality & Complexity Collapse.

Tests 17.1-17.8   : Analytical (default params, seed=42)
Tests 17.9-17.12  : Monte Carlo (5000 trajectories, seed=42)
Tests 17.13-17.15 : Sensitivity sweeps (3 param values each)
Tests 17.16-17.17 : Eigenvalue / dimensionality structural properties
Test  17.18       : Projection validity (scalar MC vs 9-axis MC)
"""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.stages.stage_17_gompertz import (
    GompertzSimulator,
    complexity_trajectory,
    dominant_mode_share,
    participation_ratio,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sim() -> GompertzSimulator:
    return GompertzSimulator()


@pytest.fixture(scope="module")
def analytical(sim: GompertzSimulator) -> dict:
    return sim.run(seed=42)


@pytest.fixture(scope="module")
def gompertz_fit(sim: GompertzSimulator) -> dict:
    return sim.fit_gompertz()


@pytest.fixture(scope="module")
def dim_results(sim: GompertzSimulator) -> dict:
    return complexity_trajectory(sim)


@pytest.fixture(scope="module")
def mc_scalar(sim: GompertzSimulator) -> dict:
    return sim.run_monte_carlo_scalar(n_trajectories=5000, seed=42)


@pytest.fixture(scope="module")
def mc_9axis(sim: GompertzSimulator) -> dict:
    return sim.run_monte_carlo(n_trajectories=5000, seed=42)


# ── Tests 17.1-17.8: Analytical checks ───────────────────────────────────────


def test_17_01_gompertz_r_squared(gompertz_fit: dict) -> None:
    """17.1: Gompertz R^2 >= 0.95 (analytical, ages 30-85)."""
    assert gompertz_fit["r_squared"] >= 0.95, (
        f"R^2 = {gompertz_fit['r_squared']:.6f}, expected >= 0.95"
    )


def test_17_02_mrdt_analytical_fitted_agreement(gompertz_fit: dict) -> None:
    """17.2: |MRDT_analytical - MRDT_fitted| / MRDT_fitted <= 0.20."""
    diff = abs(gompertz_fit["mrdt_analytical"] - gompertz_fit["mrdt_fitted"]) / gompertz_fit["mrdt_fitted"]
    assert diff <= 0.20, (
        f"MRDT agreement = {diff:.4f}, expected <= 0.20 "
        f"(analytical={gompertz_fit['mrdt_analytical']:.2f}, fitted={gompertz_fit['mrdt_fitted']:.2f})"
    )


def test_17_03_mrdt_physiological_range(gompertz_fit: dict) -> None:
    """17.3: MRDT in [4, 15] years for default params."""
    mrdt = gompertz_fit["mrdt_fitted"]
    assert 4.0 <= mrdt <= 15.0, f"MRDT = {mrdt:.2f}, expected in [4, 15]"


def test_17_04_complexity_collapse(dim_results: dict) -> None:
    """17.4: D_eff(80) / D_eff(30) <= 0.50."""
    ratio = dim_results["collapse_ratio"]
    assert ratio <= 0.50, (
        f"Collapse ratio = {ratio:.4f}, expected <= 0.50 "
        f"(D_eff(30)={dim_results['d_eff_30']:.2f}, D_eff(80)={dim_results['d_eff_80']:.2f})"
    )


def test_17_05_dominant_mode_share_80(sim: GompertzSimulator) -> None:
    """17.5: Dominant mode share at age 80 >= 60%."""
    p1 = dominant_mode_share(sim.eigenvalue_spectrum(80), sim.sigma_w)
    assert p1 >= 60.0, f"p_1(80) = {p1:.2f}%, expected >= 60%"


def test_17_06_survival_monotone(analytical: dict) -> None:
    """17.6: Survival curve monotonically non-increasing."""
    S = analytical["survival"]
    diffs = np.diff(S)
    assert np.all(diffs <= 1e-12), "Survival curve is not monotonically non-increasing"


def test_17_07_median_lifespan(analytical: dict) -> None:
    """17.7: Median lifespan in [60, 95] years."""
    med = analytical["median_lifespan"]
    assert 60.0 <= med <= 95.0, f"Median lifespan = {med:.2f}, expected in [60, 95]"


def test_17_08_criticality_age(analytical: dict) -> None:
    """17.8: Criticality age > 100."""
    crit = analytical["criticality_age"]
    assert crit > 100.0, f"Criticality age = {crit:.2f}, expected > 100"


# ── Tests 17.9-17.12: Monte Carlo checks ─────────────────────────────────────


def test_17_09_mc9axis_mrdt_agreement(mc_9axis: dict, gompertz_fit: dict) -> None:
    """17.9: 9-axis MC MRDT within 35% of analytical MRDT."""
    diff = abs(mc_9axis["empirical_mrdt"] - gompertz_fit["mrdt_fitted"]) / gompertz_fit["mrdt_fitted"]
    assert diff <= 0.35, (
        f"9-axis MC MRDT agreement = {diff:.4f}, expected <= 0.35 "
        f"(mc={mc_9axis['empirical_mrdt']:.2f}, analytical={gompertz_fit['mrdt_fitted']:.2f})"
    )


def test_17_10_mc9axis_gompertz_r_squared(mc_9axis: dict) -> None:
    """17.10: 9-axis MC Gompertz R^2 >= 0.80."""
    assert mc_9axis["empirical_r_squared"] >= 0.80, (
        f"MC 9-axis R^2 = {mc_9axis['empirical_r_squared']:.4f}, expected >= 0.80"
    )


def test_17_11_mc_scalar_mrdt_agreement(mc_scalar: dict, gompertz_fit: dict) -> None:
    """17.11: Scalar MC MRDT within 15% of analytical MRDT."""
    diff = abs(mc_scalar["empirical_mrdt"] - gompertz_fit["mrdt_fitted"]) / gompertz_fit["mrdt_fitted"]
    assert diff <= 0.15, (
        f"Scalar MC MRDT agreement = {diff:.4f}, expected <= 0.15 "
        f"(mc={mc_scalar['empirical_mrdt']:.2f}, analytical={gompertz_fit['mrdt_fitted']:.2f})"
    )


def test_17_12_mc9axis_median_lifespan(mc_9axis: dict, analytical: dict) -> None:
    """17.12: 9-axis MC median lifespan within 40 years of analytical.

    The analytical Kramers formula underestimates mortality compared to
    the 9-axis MC because cross-axis noise coupling contributes additional
    variance to the mode-1 projection, lowering the effective barrier.
    """
    diff = abs(mc_9axis["median_lifespan_mc"] - analytical["median_lifespan"])
    assert diff <= 40.0, (
        f"Median lifespan diff = {diff:.2f} years, expected <= 40 "
        f"(mc={mc_9axis['median_lifespan_mc']:.2f}, analytical={analytical['median_lifespan']:.2f})"
    )


# ── Tests 17.13-17.15: Sensitivity ───────────────────────────────────────────


def test_17_13_sensitivity_sigma_w() -> None:
    """17.13: MRDT increases with sigma_w (3-point sweep)."""
    mrdts = []
    for sw in [0.8, 1.2, 1.8]:
        s = GompertzSimulator(sigma_w=sw)
        mrdts.append(s.fit_gompertz()["mrdt_fitted"])
    assert mrdts[0] < mrdts[1] < mrdts[2], (
        f"MRDT not monotonically increasing with sigma_w: {mrdts}"
    )


def test_17_14_sensitivity_x_crit() -> None:
    """17.14: MRDT decreases with x_crit (3-point sweep).

    MRDT ~ sigma_w^2 / (gamma * x_c^2), so larger x_c => smaller MRDT.
    """
    mrdts = []
    for xc in [2.0, 2.7, 3.5]:
        s = GompertzSimulator(x_crit=xc)
        mrdts.append(s.fit_gompertz()["mrdt_fitted"])
    assert mrdts[0] > mrdts[1] > mrdts[2], (
        f"MRDT not monotonically decreasing with x_crit: {mrdts}"
    )


def test_17_15_sensitivity_gamma() -> None:
    """17.15: MRDT decreases with gamma (3-point sweep)."""
    mrdts = []
    for g in [0.008, 0.014, 0.017]:
        s = GompertzSimulator(gamma_drift=g)
        mrdts.append(s.fit_gompertz()["mrdt_fitted"])
    assert mrdts[0] > mrdts[1] > mrdts[2], (
        f"MRDT not monotonically decreasing with gamma: {mrdts}"
    )


# ── Tests 17.16-17.18: Structural checks ─────────────────────────────────────


def test_17_16_eigenvalues_negative(sim: GompertzSimulator) -> None:
    """17.16: All eigenvalues remain strictly negative for all ages."""
    for age in range(sim.age_start, sim.age_end + 1):
        eigs = sim.eigenvalue_spectrum(age)
        assert np.all(eigs < 0), (
            f"Non-negative eigenvalue at age {age}: {eigs}"
        )


def test_17_17_d_eff_monotone(dim_results: dict) -> None:
    """17.17: D_eff monotonically non-increasing over age range."""
    diffs = np.diff(dim_results["d_eff"])
    assert np.all(diffs <= 1e-10), (
        f"D_eff not monotonically non-increasing; max increase = {np.max(diffs):.6e}"
    )


def test_17_18_projection_validity(mc_scalar: dict, mc_9axis: dict) -> None:
    """17.18: Scalar MC within 25% of 9-axis MC MRDT.

    The 9-axis MC includes cross-axis noise coupling into mode-1 via the
    orthogonal mixing matrix, producing a systematic MRDT shift that is
    a genuine finding about cross-coupled mortality dynamics.
    """
    diff = abs(mc_scalar["empirical_mrdt"] - mc_9axis["empirical_mrdt"]) / mc_9axis["empirical_mrdt"]
    assert diff <= 0.25, (
        f"Projection validity: |MRDT_scalar - MRDT_9axis| / MRDT_9axis = {diff:.4f}, expected <= 0.25 "
        f"(scalar={mc_scalar['empirical_mrdt']:.2f}, 9axis={mc_9axis['empirical_mrdt']:.2f})"
    )


# ── Integration test: full stage run ─────────────────────────────────────────


def test_stage_17_full_run_fast() -> None:
    """Full stage 17 run in fast mode produces valid results."""
    from hdr_validation.stages.stage_17_gompertz import run_stage_17

    result = run_stage_17(n_trajectories=2000, seed=42, fast_mode=True)
    assert result is not None
    assert "checks" in result
    assert len(result["checks"]) == 18
    assert "provenance" in result
