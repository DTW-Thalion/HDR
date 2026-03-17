"""Tests for Stage 10 — Mode B FP/FN Sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_10_mode_b_sweep import (
    inject_miscalibration,
    run_stage_10,
)


def test_inject_miscalibration_output_in_range():
    """inject_miscalibration should return a probability in (0, 1)."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        p = inject_miscalibration(p_true=0.7, R_Brier_target=0.05, rng=rng)
        assert 0.0 < p < 1.0


def test_inject_miscalibration_zero_noise():
    """With R_Brier_target=0, output should be close to p_true (no noise)."""
    rng = np.random.default_rng(0)
    p_true = 0.6
    # With zero noise std, output = sigmoid(logit(p_true)) = p_true exactly
    p = inject_miscalibration(p_true=p_true, R_Brier_target=0.0, rng=rng)
    assert abs(p - p_true) < 1e-6


# ---- Module-scoped fixture for (N_sim=100, T=20) group ----

@pytest.fixture(scope="module")
def stage_10_result_standard(tmp_path_factory):
    """Run Stage 10 once with standard fast parameters; return (result, output_dir)."""
    tmp = tmp_path_factory.mktemp("stage_10")
    result = run_stage_10(N_sim=100, T=20, output_dir=tmp, fast_mode=False)
    return result, tmp


def test_run_stage_10_output_structure(stage_10_result_standard):
    """Stage 10 should produce JSON with correct structure."""
    result, tmp = stage_10_result_standard

    assert "sweep" in result
    assert len(result["sweep"]) == 5  # 5 R_Brier levels

    out_file = tmp / "mode_b_fp_fn_sweep.json"
    assert out_file.exists()


def test_run_stage_10_thresholds_equal_at_zero(stage_10_result_standard):
    """At R_Brier_target=0, fixed and robust thresholds should be equal."""
    result, tmp = stage_10_result_standard
    row_zero = result["sweep"][0]
    assert row_zero["R_Brier_target"] == 0.0
    assert abs(row_zero["threshold_fixed"] - row_zero["threshold_robust"]) < 1e-9


def test_run_stage_10_fp_rates_in_range(stage_10_result_standard):
    """All FP and FN rates should be in [0, 1]."""
    result, tmp = stage_10_result_standard
    for row in result["sweep"]:
        assert 0.0 <= row["FP_rate_fixed_threshold"] <= 1.0
        assert 0.0 <= row["FP_rate_robust_threshold"] <= 1.0
        assert 0.0 <= row["FN_rate_fixed_threshold"] <= 1.0
        assert 0.0 <= row["FN_rate_robust_threshold"] <= 1.0


def test_run_stage_10_all_5_levels_present(stage_10_result_standard):
    """All 5 R_Brier sweep levels should be present."""
    result, tmp = stage_10_result_standard
    r_brier_values = [row["R_Brier_target"] for row in result["sweep"]]
    assert len(r_brier_values) == 5
    expected = [0.00, 0.05, 0.10, 0.15, 0.20]
    for exp, got in zip(expected, r_brier_values):
        assert abs(exp - got) < 1e-9


# ---- Single-caller test with different parameters ----

def test_run_stage_10_robust_fp_le_fixed_fp(tmp_path: Path):
    """FP rate with robust threshold should be <= FP rate with fixed threshold (all levels)."""
    result = run_stage_10(N_sim=500, T=30, output_dir=tmp_path, fast_mode=False)
    for row in result["sweep"]:
        assert row["FP_rate_robust_threshold"] <= row["FP_rate_fixed_threshold"] + 1e-9, \
            f"At R_Brier={row['R_Brier_target']}: robust FP={row['FP_rate_robust_threshold']} " \
            f"> fixed FP={row['FP_rate_fixed_threshold']}"
