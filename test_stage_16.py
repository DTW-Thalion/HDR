"""Tests for Stage 16 -- Model-Failure Extension Integration Validation."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.stages.stage_16_extensions import (
    _make_stage16_config,
    _run_subtest_16_01_pwa,
    _run_subtest_16_02_absorbing,
    _run_subtest_16_03_basin_stability,
    _run_subtest_16_04_multisite,
    _run_subtest_16_05_adaptive,
    _run_subtest_16_06_jump,
    _run_subtest_16_07_mimpc,
    _run_subtest_16_08_multirate,
    _run_subtest_16_09_cumulative,
    _run_subtest_16_10_condcoupling,
    _run_subtest_16_11_expansion,
    _run_subtest_16_12_baseline,
    _run_subtest_16_13_dm,
    _run_subtest_16_14_ca,
    _run_subtest_16_15_os,
    _run_subtest_16_16_ad,
    _run_subtest_16_17_crd,
    run_stage_16,
)


def _fast_cfg(n_seeds=2, T=32):
    return _make_stage16_config(n_seeds=n_seeds, T=T)


# ---- Module-scoped fixtures: one run per unique subtest+params ----

@pytest.fixture(scope="module")
def result_16_01():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_01_pwa(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_02():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_02_absorbing(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_03():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_03_basin_stability(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_04():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_04_multisite(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_06():
    cfg = _fast_cfg(n_seeds=2, T=128)
    return _run_subtest_16_06_jump(cfg, n_seeds=2, T=128)


@pytest.fixture(scope="module")
def result_16_07():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_07_mimpc(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_08():
    cfg = _fast_cfg(n_seeds=2, T=64)
    return _run_subtest_16_08_multirate(cfg, n_seeds=2, T=64)


@pytest.fixture(scope="module")
def result_16_09():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_09_cumulative(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_10():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_10_condcoupling(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_11():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_11_expansion(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_13():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_13_dm(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_14():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_14_ca(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_15():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_15_os(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_16():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_16_ad(cfg, n_seeds=2, T=32)


@pytest.fixture(scope="module")
def result_16_17():
    cfg = _fast_cfg(n_seeds=2, T=32)
    return _run_subtest_16_17_crd(cfg, n_seeds=2, T=32)


# ===================================================================
# 16.01: PWA SLDS (existing)
# ===================================================================

def test_stage16_pwa_numerical_stability(result_16_01):
    assert result_16_01["numerical_stability"] is True


def test_stage16_pwa_region_consistency(result_16_01):
    assert result_16_01["region_consistency_rate"] >= 0.90


# ===================================================================
# 16.02: Absorbing-State Partition (M2)
# ===================================================================

def test_stage16_absorbing_monotonicity(result_16_02):
    assert result_16_02["monotonicity_rate"] == 1.0


def test_stage16_absorbing_detection(result_16_02):
    assert result_16_02["absorbing_detected"] is True


def test_stage16_absorbing_backward_compat(result_16_02):
    assert result_16_02["backward_compatible"] is True


# ===================================================================
# 16.03: Basin Stability Classification (M1)
# ===================================================================

def test_stage16_basin_stability_classification(result_16_03):
    assert result_16_03["classification_accuracy"] == 1.0


def test_stage16_basin_stability_mode_b_bypass(result_16_03):
    assert result_16_03["mode_b_bypass_rate"] == 1.0


def test_stage16_basin_stability_projection(result_16_03):
    assert result_16_03["projection_error"] < 1e-10


# ===================================================================
# 16.04: Multi-Site Dynamics (M4)
# ===================================================================

def test_stage16_multisite_stability(result_16_04):
    assert result_16_04["composite_stable"] is True
    assert result_16_04["gershgorin_holds"] is True


def test_stage16_multisite_imm_convergence(result_16_04):
    assert result_16_04["per_site_imm_converged"] is True


def test_stage16_multisite_propagation(result_16_04):
    assert result_16_04["cross_site_response"] > 0.1


# ===================================================================
# 16.05: Adaptive estimation (existing) — single caller, no fixture
# ===================================================================

def test_stage16_adaptive_drift_tracked():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_05_adaptive(cfg, n_seeds=2, T=64)
    assert result["drift_tracked_rate"] >= 0.50


# ===================================================================
# 16.06: Jump-Diffusion (M6)
# ===================================================================

def test_stage16_jump_rate_in_ci(result_16_06):
    assert result_16_06["jump_rate_in_ci"] is True


def test_stage16_jump_committor_shift(result_16_06):
    assert result_16_06["committor_shift_proportional"] is True


def test_stage16_jump_magnitude_distribution(result_16_06):
    assert result_16_06["jump_magnitude_ks_p"] > 0.01


def test_stage16_jump_prophylactic_trigger(result_16_06):
    assert result_16_06["prophylactic_trigger_rate"] == 1.0


# ===================================================================
# 16.07: Mixed-Integer MPC (M7)
# ===================================================================

def test_stage16_mimpc_integrality(result_16_07):
    assert result_16_07["binary_integrality_rate"] == 1.0


def test_stage16_mimpc_onetime_constraint(result_16_07):
    assert result_16_07["onetime_constraint_satisfied"] is True


def test_stage16_mimpc_feasibility(result_16_07):
    assert result_16_07["feasibility_rate"] == 1.0


# ===================================================================
# 16.08: Multi-Rate IMM (M8)
# ===================================================================

def test_stage16_multirate_masking(result_16_08):
    assert result_16_08["masking_correct"] is True


def test_stage16_multirate_covariance(result_16_08):
    assert result_16_08["covariance_growth_monotonic"] is True
    assert result_16_08["observation_epoch_improvement"] is True


def test_stage16_multirate_mode_accuracy(result_16_08):
    assert result_16_08["mode_accuracy_above_threshold"] is True


# ===================================================================
# 16.09: Cumulative-Exposure (M9)
# ===================================================================

def test_stage16_cumulative_monotonicity(result_16_09):
    assert result_16_09["monotonicity_rate"] == 1.0


def test_stage16_cumulative_constraint(result_16_09):
    assert result_16_09["exposure_violations"] == 0


def test_stage16_cumulative_toxicity_coupling(result_16_09):
    assert result_16_09["toxicity_correlation"] > 0.2


# ===================================================================
# 16.10: State-Conditioned Coupling (M10)
# ===================================================================

def test_stage16_condcoupling_sigmoid(result_16_10):
    assert result_16_10["sigmoid_correct"] is True


def test_stage16_condcoupling_stability(result_16_10):
    assert result_16_10["stability_preserved"] is True


def test_stage16_condcoupling_sign_reversal(result_16_10):
    assert result_16_10["sign_reversal_observed"] is True


# ===================================================================
# 16.11: Modular Axis Expansion (M11)
# ===================================================================

def test_stage16_expansion_stability(result_16_11):
    assert result_16_11["expanded_stable"] is True
    assert result_16_11["bound_holds"] is True


def test_stage16_expansion_unperturbed(result_16_11):
    assert result_16_11["original_unperturbed"] is True


def test_stage16_expansion_responsiveness(result_16_11):
    assert result_16_11["new_axis_responsive"] is True


# ===================================================================
# 16.12: PD Baseline (existing) — single caller, no fixture
# ===================================================================

def test_stage16_baseline_equivalence():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_12_baseline(cfg, n_seeds=2, T=32)
    assert result["backward_compatible"] is True


# ===================================================================
# 16.13: DM Profile (M5+M10)
# ===================================================================

def test_stage16_dm_individual_invariants(result_16_13):
    assert result_16_13["drift_tracked"] is True
    assert result_16_13["coupling_correct"] is True


def test_stage16_dm_interaction_bounded(result_16_13):
    assert result_16_13["interaction_bounded"] is True


# ===================================================================
# 16.14: CA Profile (7 extensions)
# ===================================================================

def test_stage16_ca_individual_invariants(result_16_14):
    assert result_16_14["individual_invariants_pass"] is True


def test_stage16_ca_interaction_stress(result_16_14):
    assert result_16_14["jump_near_ceiling_safe"] is True
    assert result_16_14["multisite_supervisor_correct"] is True


# ===================================================================
# 16.15: OS Profile (4 extensions)
# ===================================================================

def test_stage16_os_individual_invariants(result_16_15):
    assert result_16_15["individual_invariants_pass"] is True


def test_stage16_os_reconvergence(result_16_15):
    assert result_16_15["reconvergence_within_bound"] is True


# ===================================================================
# 16.16: AD Profile (M3+M2+M8)
# ===================================================================

def test_stage16_ad_individual_invariants(result_16_16):
    assert result_16_16["individual_invariants_pass"] is True


def test_stage16_ad_threshold_interaction(result_16_16):
    assert result_16_16["threshold_irr_coupling"] is True
    assert result_16_16["region_stability_during_blackout"] is True


# ===================================================================
# 16.17: CRD Profile (M11 only)
# ===================================================================

def test_stage16_crd_expansion_invariants(result_16_17):
    assert result_16_17["expansion_invariants_pass"] is True


def test_stage16_crd_cost_ratio(result_16_17):
    assert result_16_17["cost_ratio"] <= 1.10


# ===================================================================
# Integration / smoke tests (existing)
# ===================================================================

def test_stage16_stub_returns_not_implemented():
    """Verify that no sub-tests are still stubbed."""
    from hdr_validation.stages.stage_16_extensions import STAGE_16_SUBTESTS
    stubs = [k for k, v in STAGE_16_SUBTESTS.items() if v["status"] == "STUB"]
    assert stubs == [], f"Still stubbed: {stubs}"


def test_stage16_full_run_no_crash():
    result = run_stage_16(n_seeds=1, T=16, fast_mode=True)
    assert all(r.get("pass", True) or r.get("status") == "NOT_IMPLEMENTED"
               for r in result.values())
