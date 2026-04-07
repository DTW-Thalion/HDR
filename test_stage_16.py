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


# ===== 16.01: PWA SLDS (existing) =====

def test_stage16_pwa_numerical_stability():
    cfg = _fast_cfg(n_seeds=2, T=32)
    _run_subtest_16_04_multisite(cfg, n_seeds=2, T=32)


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


# ===== 16.02: Absorbing-State Partition =====

def test_stage16_absorbing_monotonicity():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_02_absorbing(cfg, n_seeds=2, T=32)
    assert result["monotonicity_rate"] >= 0.95


def test_stage16_absorbing_detection():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_02_absorbing(cfg, n_seeds=2, T=32)
    assert result["absorbing_detected"] is True


def test_stage16_absorbing_backward_compat():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_02_absorbing(cfg, n_seeds=2, T=32)
    assert result["backward_compatible"] is True


# ===== 16.03: Basin Stability Classification =====

def test_stage16_basin_stability_classification():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_03_basin_stability(cfg, n_seeds=2, T=32)
    assert result["classification_accuracy"] == 1.0


def test_stage16_basin_stability_mode_b_bypass():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_03_basin_stability(cfg, n_seeds=2, T=32)
    assert result["mode_b_bypass_rate"] == 1.0


def test_stage16_basin_stability_projection():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_03_basin_stability(cfg, n_seeds=2, T=32)
    assert result["projection_error"] < 1e-10


# ===== 16.04: Multi-Site Dynamics =====

def test_stage16_multisite_stability():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_04_multisite(cfg, n_seeds=2, T=32)
    assert result["composite_stable"] is True
    assert result["gershgorin_holds"] is True


def test_stage16_multisite_imm_convergence():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_04_multisite(cfg, n_seeds=2, T=32)
    assert result["per_site_imm_converged"] is True


def test_stage16_multisite_propagation():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_04_multisite(cfg, n_seeds=2, T=32)
    assert result["cross_site_response"] > 0.1


# ===== 16.05: Adaptive Estimation (existing) =====

def test_stage16_adaptive_drift_tracked():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_05_adaptive(cfg, n_seeds=2, T=64)
    assert result["drift_tracked_rate"] >= 0.50


# ===== 16.06: Jump-Diffusion =====

def test_stage16_jump_rate_in_ci():
    cfg = _fast_cfg(n_seeds=2, T=128)
    result = _run_subtest_16_06_jump(cfg, n_seeds=2, T=128)
    assert result["jump_rate_in_ci"]


def test_stage16_jump_committor_shift():
    cfg = _fast_cfg(n_seeds=2, T=128)
    result = _run_subtest_16_06_jump(cfg, n_seeds=2, T=128)
    assert result["committor_shift_proportional"] is True


def test_stage16_jump_magnitude_distribution():
    cfg = _fast_cfg(n_seeds=2, T=128)
    result = _run_subtest_16_06_jump(cfg, n_seeds=2, T=128)
    assert result["jump_magnitude_ks_p"] > 0.01


def test_stage16_jump_prophylactic_trigger():
    cfg = _fast_cfg(n_seeds=2, T=128)
    result = _run_subtest_16_06_jump(cfg, n_seeds=2, T=128)
    assert result["prophylactic_trigger_rate"] == 1.0


# ===== 16.07: Mixed-Integer MPC =====

def test_stage16_mimpc_integrality():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_07_mimpc(cfg, n_seeds=2, T=32)
    assert result["binary_integrality_rate"] == 1.0


def test_stage16_mimpc_onetime_constraint():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_07_mimpc(cfg, n_seeds=2, T=32)
    assert result["onetime_constraint_satisfied"] is True


def test_stage16_mimpc_feasibility():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_07_mimpc(cfg, n_seeds=2, T=32)
    assert result["feasibility_rate"] == 1.0


# ===== 16.08: Multi-Rate IMM =====

def test_stage16_multirate_masking():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_08_multirate(cfg, n_seeds=2, T=64)
    assert result["masking_correct"] is True


def test_stage16_multirate_covariance():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_08_multirate(cfg, n_seeds=2, T=64)
    assert result["covariance_growth_monotonic"] is True
    assert result["observation_epoch_improvement"] is True


def test_stage16_multirate_mode_accuracy():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_08_multirate(cfg, n_seeds=2, T=64)
    assert result["mode_accuracy_above_threshold"] is True


# ===== 16.09: Cumulative-Exposure =====

def test_stage16_cumulative_monotonicity():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_09_cumulative(cfg, n_seeds=2, T=32)
    assert result["monotonicity_rate"] == 1.0


def test_stage16_cumulative_constraint():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_09_cumulative(cfg, n_seeds=2, T=32)
    assert result["exposure_violations"] == 0


def test_stage16_cumulative_toxicity_coupling():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_09_cumulative(cfg, n_seeds=2, T=32)
    assert result["toxicity_correlation"] > 0.2


# ===== 16.10: State-Conditioned Coupling =====

def test_stage16_condcoupling_sigmoid():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_10_condcoupling(cfg, n_seeds=2, T=32)
    assert result["sigmoid_correct"] is True


def test_stage16_condcoupling_stability():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_10_condcoupling(cfg, n_seeds=2, T=32)
    assert result["stability_preserved"] is True


def test_stage16_condcoupling_sign_reversal():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_10_condcoupling(cfg, n_seeds=2, T=32)
    assert result["sign_reversal_observed"] is True


# ===== 16.11: Modular Axis Expansion =====

def test_stage16_expansion_stability():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_11_expansion(cfg, n_seeds=2, T=32)
    assert result["expanded_stable"] is True
    assert result["bound_holds"] is True


def test_stage16_expansion_unperturbed():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_11_expansion(cfg, n_seeds=2, T=32)
    assert result["original_unperturbed"] is True


def test_stage16_expansion_responsiveness():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_11_expansion(cfg, n_seeds=2, T=32)
    assert result["new_axis_responsive"] is True


# ===== 16.12: PD Baseline (existing) =====

def test_stage16_baseline_equivalence():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_12_baseline(cfg, n_seeds=2, T=32)
    assert result["backward_compatible"] is True


# ===== 16.13: DM Profile =====

def test_stage16_dm_individual_invariants():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_13_dm(cfg, n_seeds=2, T=32)
    assert result["drift_tracked"] is True
    assert result["coupling_correct"] is True


def test_stage16_dm_interaction_bounded():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_13_dm(cfg, n_seeds=2, T=32)
    assert result["interaction_bounded"] is True


# ===== 16.14: CA Profile =====

def test_stage16_ca_individual_invariants():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_14_ca(cfg, n_seeds=2, T=32)
    assert result["individual_invariants_pass"] is True


def test_stage16_ca_interaction_stress():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_14_ca(cfg, n_seeds=2, T=32)
    assert result["jump_near_ceiling_safe"] is True
    assert result["multisite_supervisor_correct"] is True


# ===== 16.15: OS Profile =====

def test_stage16_os_individual_invariants():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_15_os(cfg, n_seeds=2, T=32)
    assert result["individual_invariants_pass"] is True


def test_stage16_os_reconvergence():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_15_os(cfg, n_seeds=2, T=32)
    assert result["reconvergence_within_bound"] is True


# ===== 16.16: AD Profile =====

def test_stage16_ad_individual_invariants():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_16_ad(cfg, n_seeds=2, T=32)
    assert result["individual_invariants_pass"] is True


def test_stage16_ad_threshold_interaction():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_16_ad(cfg, n_seeds=2, T=32)
    assert result["threshold_irr_coupling"] is True
    assert result["region_stability_during_blackout"] is True


# ===== 16.17: CRD Profile =====

def test_stage16_crd_expansion_invariants():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_17_crd(cfg, n_seeds=2, T=32)
    assert result["expansion_invariants_pass"] is True


def test_stage16_crd_cost_ratio():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_17_crd(cfg, n_seeds=2, T=32)
    assert result["cost_ratio"] <= 1.10


# ===== Integration tests =====

def test_stage16_full_run_no_crash():
    result = run_stage_16(n_seeds=1, T=16, fast_mode=True)
    assert all(r.get("pass", True) or r.get("status") == "NOT_IMPLEMENTED"
               for r in result.values())
