"""Tests for Stage 09 — MJLS-SMPC and Belief-MPC Baselines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_09_baselines import (
    belief_mpc_policy,
    mjls_smpc_policy,
    run_stage_09,
)


def test_mjls_smpc_uses_correct_basin_gain():
    """mjls_smpc_policy should use K_banks[z_true]."""
    n = 4
    K_banks = {
        0: np.eye(n) * 0.1,
        1: np.eye(n) * 0.5,
        2: np.eye(n) * 0.9,
    }
    x_hat = np.ones(n)
    x_ref = np.zeros(n)
    u = mjls_smpc_policy(x_hat, z_true=1, K_banks=K_banks, x_ref=x_ref)
    expected = -K_banks[1] @ x_hat
    expected = np.clip(expected, -0.6, 0.6)
    np.testing.assert_allclose(u, expected, atol=1e-9)


def test_mjls_smpc_clipped_to_bounds():
    """Output should be clipped to [-0.6, 0.6]."""
    n = 4
    K_banks = {0: np.eye(n) * 10.0}
    x_hat = np.ones(n) * 2.0
    x_ref = np.zeros(n)
    u = mjls_smpc_policy(x_hat, z_true=0, K_banks=K_banks, x_ref=x_ref)
    assert np.all(np.abs(u) <= 0.6 + 1e-9)


def test_belief_mpc_weighted_combination():
    """belief_mpc_policy should produce weighted combination of gains."""
    n = 4
    K_banks = {
        0: np.zeros((n, n)),
        1: np.eye(n),
    }
    # Equal weights → K_mixture = 0.5 * I
    mode_posteriors = {0: 0.5, 1: 0.5}
    x_hat = np.array([0.1, 0.1, 0.1, 0.1])
    x_ref = np.zeros(n)
    u = belief_mpc_policy(x_hat, mode_posteriors, K_banks, x_ref)
    expected = -0.5 * x_hat  # K_mixture = 0.5 * I
    np.testing.assert_allclose(u, expected, atol=1e-9)


def test_belief_mpc_clipped_to_bounds():
    """Output should be within [-0.6, 0.6]."""
    n = 4
    K_banks = {0: np.eye(n) * 10.0}
    mode_posteriors = {0: 1.0}
    x_hat = np.ones(n) * 2.0
    x_ref = np.zeros(n)
    u = belief_mpc_policy(x_hat, mode_posteriors, K_banks, x_ref)
    assert np.all(np.abs(u) <= 0.6 + 1e-9)


# ---- Module-scoped fixture for (n_seeds=2, n_ep=3, T=32) group ----

@pytest.fixture(scope="module")
def stage_09_with_path(tmp_path_factory):
    """Run Stage 09 once with standard fast parameters; return (result, output_dir)."""
    tmp = tmp_path_factory.mktemp("stage_09")
    result = run_stage_09(n_seeds=2, n_ep=3, T=32, output_dir=tmp, fast_mode=False)
    return result, tmp


def test_run_stage_09_fast(stage_09_with_path):
    """Smoke test: run with n_seeds=2, n_ep=3. Check output structure."""
    result, tmp = stage_09_with_path

    assert "policies" in result
    required = {"open_loop", "pooled_lqr_estimated", "mjls_smpc", "belief_mpc", "hdr_mode_a"}
    assert set(result["policies"].keys()) == required

    # MJLS-SMPC has oracle note
    assert "note" in result["policies"]["mjls_smpc"]
    assert "oracle" in result["policies"]["mjls_smpc"]["note"].lower()

    # JSON saved
    out_file = tmp / "baseline_comparison.json"
    assert out_file.exists()


def test_belief_mpc_gain_finite(stage_09_with_path):
    """Belief-MPC gain should be finite and within a reasonable range."""
    result, tmp = stage_09_with_path
    gain = result["policies"]["belief_mpc"]["mean_gain_vs_pooled_lqr"]
    assert gain is not None
    assert np.isfinite(gain)
    assert -0.5 <= gain <= 0.5
