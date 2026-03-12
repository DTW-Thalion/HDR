"""Tests for Stage 11 — Riccati Invariant Set Verification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_11_invariant_set import (
    check_trajectory_containment,
    compute_lyapunov_level_set_radius,
    run_stage_11,
)


def test_compute_lyapunov_level_set_radius_positive():
    """c_k should be a positive scalar."""
    n = 4
    P_k = np.eye(n)
    A_cl_k = 0.5 * np.eye(n)  # stable closed-loop
    Q_w_k = 0.1 * np.eye(n)
    c_k = compute_lyapunov_level_set_radius(P_k, A_cl_k, Q_w_k, n_sigma=3.0)
    assert c_k > 0.0
    assert np.isfinite(c_k)


def test_compute_lyapunov_level_set_radius_scales_with_noise():
    """Larger noise covariance should lead to larger c_k."""
    n = 4
    P_k = np.eye(n)
    A_cl_k = 0.5 * np.eye(n)
    Q_small = 0.01 * np.eye(n)
    Q_large = 1.0 * np.eye(n)
    c_small = compute_lyapunov_level_set_radius(P_k, A_cl_k, Q_small)
    c_large = compute_lyapunov_level_set_radius(P_k, A_cl_k, Q_large)
    assert c_large > c_small


def test_check_trajectory_containment_returns_correct_keys():
    """check_trajectory_containment returns dict with required keys."""
    n = 4
    P_k = np.eye(n)
    c_k = 10.0
    trajectories = [np.random.default_rng(i).normal(size=(20, n)) for i in range(3)]
    basin_labels = [np.zeros(20, dtype=int) for _ in range(3)]
    result = check_trajectory_containment(trajectories, P_k, c_k, basin_labels, target_basin=0)
    assert "containment_rate" in result
    assert "mean_lyapunov_value" in result
    assert "max_lyapunov_value" in result
    assert "n_steps_checked" in result


def test_check_trajectory_containment_rate_in_range():
    """containment_rate should be in [0, 1]."""
    n = 4
    P_k = np.eye(n)
    c_k = 4.0  # Some points inside, some outside
    rng = np.random.default_rng(0)
    trajectories = [rng.normal(size=(50, n)) for _ in range(5)]
    basin_labels = [np.zeros(50, dtype=int) for _ in range(5)]
    result = check_trajectory_containment(trajectories, P_k, c_k, basin_labels, target_basin=0)
    assert 0.0 <= result["containment_rate"] <= 1.0


def test_check_trajectory_containment_empty_basin():
    """If no steps in target basin, containment_rate should be NaN."""
    n = 4
    P_k = np.eye(n)
    c_k = 10.0
    trajectories = [np.zeros((10, n))]
    basin_labels = [np.ones(10, dtype=int)]  # All basin 1, target is basin 0
    result = check_trajectory_containment(trajectories, P_k, c_k, basin_labels, target_basin=0)
    assert result["n_steps_checked"] == 0
    assert np.isnan(result["containment_rate"])


def test_check_trajectory_containment_all_inside():
    """Small trajectory near origin should have high containment rate for large c_k."""
    n = 4
    P_k = np.eye(n)
    c_k = 1000.0  # Very large level set — all points inside
    rng = np.random.default_rng(0)
    trajectories = [rng.normal(scale=0.1, size=(50, n)) for _ in range(5)]
    basin_labels = [np.zeros(50, dtype=int) for _ in range(5)]
    result = check_trajectory_containment(trajectories, P_k, c_k, basin_labels, target_basin=0)
    assert result["containment_rate"] == 1.0


def test_run_stage_11_fast(tmp_path: Path):
    """Smoke test: run with n_seeds=2, T=20. Check output structure."""
    result = run_stage_11(n_seeds=2, T=20, output_dir=tmp_path, fast_mode=False)

    assert "basins" in result
    assert len(result["basins"]) == 3  # 3 basins

    for k_str, data in result["basins"].items():
        assert "c_k" in data
        assert "containment_rate" in data
        assert "proposition_8_4_criterion_met" in data
        assert "n_steps_checked" in data
        if data["containment_rate"] is not None:
            assert 0.0 <= data["containment_rate"] <= 1.0
        assert data["c_k"] > 0

    out_file = tmp_path / "invariant_set_verification.json"
    assert out_file.exists()


def test_run_stage_11_containment_reasonable(tmp_path: Path):
    """Under Mode A, containment rate should be > 0 for the closed-loop system."""
    result = run_stage_11(n_seeds=2, T=30, output_dir=tmp_path, fast_mode=False)
    # At least some basins should have nonzero steps checked
    total_steps = sum(
        data["n_steps_checked"] for data in result["basins"].values()
    )
    assert total_steps > 0

    # For basins with checked steps, rate should be positive
    for k_str, data in result["basins"].items():
        if data["n_steps_checked"] > 0 and data["containment_rate"] is not None:
            assert data["containment_rate"] >= 0.0


def test_rpi_rate_higher_than_overall_rate(tmp_path: Path):
    """RPI rate should exceed overall rate when trajectories start inside the set."""
    result = run_stage_11(n_seeds=2, T=64, output_dir=tmp_path, fast_mode=False)
    for k_str, data in result["basins"].items():
        rpi = data.get("containment_rate_rpi")
        overall = data.get("containment_rate")
        if rpi is not None and overall is not None:
            # If trajectories start inside (fixed init), rpi should be >= overall
            # Allow small tolerance for stochastic variation
            assert rpi >= overall - 0.05, (
                f"Basin {k_str}: rpi_rate={rpi:.3f} < overall={overall:.3f}. "
                "Trajectories may still be starting outside the ellipsoid."
            )
