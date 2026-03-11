"""Tests for hdr_validation.inference.ici — including mu_erg."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hdr_validation.inference.ici import (
    apply_calibration,
    brier_reliability,
    check_mu_erg_vs_mu_hat,
    compute_ici_state,
    compute_mu_erg,
    compute_p_A_robust,
    compute_T_k_eff,
    isotonic_calibrate,
)


# ── compute_mu_erg ────────────────────────────────────────────────────────────

def test_mu_erg_all_correct():
    """All-correct history → mu_erg = 0.0."""
    history = [False] * 100
    assert compute_mu_erg(history) == 0.0


def test_mu_erg_all_incorrect():
    """All-incorrect history → mu_erg = 1.0."""
    history = [True] * 100
    assert compute_mu_erg(history) == 1.0


def test_mu_erg_alternating():
    """Alternating True/False → mu_erg ≈ 0.5."""
    history = [True, False] * 50
    result = compute_mu_erg(history)
    assert abs(result - 0.5) < 1e-9


def test_mu_erg_empty_history():
    """Empty history → mu_erg = 0.0."""
    assert compute_mu_erg([]) == 0.0


def test_mu_erg_in_range():
    """mu_erg should always be in [0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(1, 200)
        history = [bool(x) for x in rng.integers(0, 2, size=n)]
        result = compute_mu_erg(history)
        assert 0.0 <= result <= 1.0


# ── check_mu_erg_vs_mu_hat ────────────────────────────────────────────────────

def test_mu_erg_le_mu_hat_no_warning():
    """When mu_erg <= mu_hat, no warning is raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_mu_erg_vs_mu_hat(mu_erg=0.1, mu_hat=0.3)
    assert len(w) == 0


def test_mu_erg_gt_mu_hat_warns():
    """When mu_erg > mu_hat + 1e-9, a warning is issued."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_mu_erg_vs_mu_hat(mu_erg=0.5, mu_hat=0.2)
    assert len(w) == 1
    assert "underestimated" in str(w[0].message).lower()


def test_mu_erg_equal_mu_hat_no_warning():
    """When mu_erg == mu_hat exactly (within tolerance), no warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_mu_erg_vs_mu_hat(mu_erg=0.3, mu_hat=0.3)
    assert len(w) == 0


# ── compute_ici_state ─────────────────────────────────────────────────────────

def test_ici_state_contains_mu_erg_when_history_provided():
    """compute_ici_state includes mu_erg when classification_history given."""
    history = [True, False, True, False]  # mu_erg = 0.5
    state = compute_ici_state(
        mu_hat=0.8,
        mu_bar_required=0.6,
        R_brier=0.01,
        R_brier_max=0.05,
        T_k_eff_per_basin=[10.0, 20.0],
        omega_min=5.0,
        classification_history=history,
    )
    assert "mu_erg" in state
    assert abs(state["mu_erg"] - 0.5) < 1e-9


def test_ici_state_mu_erg_nan_when_no_history():
    """compute_ici_state has mu_erg=NaN when no history provided."""
    state = compute_ici_state(
        mu_hat=0.5,
        mu_bar_required=0.6,
        R_brier=0.01,
        R_brier_max=0.05,
        T_k_eff_per_basin=[10.0],
        omega_min=5.0,
    )
    assert math.isnan(state["mu_erg"])


def test_ici_state_mu_erg_le_mu_hat_valid():
    """mu_erg assertion does not trigger for valid inputs (mu_erg <= mu_hat)."""
    history = [False] * 80 + [True] * 20  # mu_erg = 0.2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        state = compute_ici_state(
            mu_hat=0.5,
            mu_bar_required=0.6,
            R_brier=0.01,
            R_brier_max=0.05,
            T_k_eff_per_basin=[10.0],
            omega_min=5.0,
            classification_history=history,
        )
    assert len(w) == 0
    assert state["mu_erg"] <= state["mu_hat"] + 1e-9


# ── Other ICI tests ───────────────────────────────────────────────────────────

def test_compute_T_k_eff_basic():
    """Proposition 9.2: T_k_eff = T * pi_k * (1-p_miss) * (1-rho_k)."""
    result = compute_T_k_eff(T=100, pi_k=0.16, p_miss=0.5, rho_k=0.96)
    expected = 100 * 0.16 * 0.5 * 0.04
    assert abs(result - expected) < 1e-9


def test_compute_p_A_robust_increases_with_R_brier():
    """p_A^robust = p_A + k_calib * R_Brier should increase with R_Brier."""
    p0 = compute_p_A_robust(0.70, 1.0, 0.0)
    p1 = compute_p_A_robust(0.70, 1.0, 0.05)
    assert p1 > p0


def test_brier_reliability_perfect_calibration():
    """Perfect calibration (predicted = observed) → reliability ≈ 0."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    result = brier_reliability(y_true, y_prob)
    assert result["reliability"] < 1e-9


def test_isotonic_calibrate_and_apply():
    """Calibration map should be monotone and apply_calibration returns array."""
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) > 0.5).astype(float)
    y_prob = np.clip(rng.random(200), 0, 1)
    cal_map = isotonic_calibrate(y_true, y_prob, n_bins=10)
    assert cal_map.shape == (10,)
    # Monotone non-decreasing
    for i in range(len(cal_map) - 1):
        assert cal_map[i] <= cal_map[i + 1] + 1e-9
    # apply_calibration returns same-length array
    calibrated = apply_calibration(y_prob, cal_map)
    assert calibrated.shape == y_prob.shape
