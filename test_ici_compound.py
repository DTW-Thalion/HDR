"""
Unit tests for the Inference–Control Interface (ICI).
Tests cover Propositions 9.1, 9.2, Definitions 8.1/8.2, and Theorem H.10.
"""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.inference.ici import (
    apply_calibration,
    brier_reliability,
    compute_degradation_factor,
    compute_epsilon_H,
    compute_ici_state,
    compute_iss_residual,
    compute_mode_b_suboptimality_bound,
    compute_mu_bar_required,
    compute_omega_min,
    compute_p_A_robust,
    compute_T_k_eff,
    isotonic_calibrate,
)


class TestTKeff:
    """Proposition 9.2 — Compound bound."""

    def test_formula_factorisation(self):
        T = 1000.0
        pi_k, p_miss, rho_k = 0.16, 0.516, 0.96
        T_k_eff = compute_T_k_eff(T, pi_k, p_miss, rho_k)
        expected = T * pi_k * (1 - p_miss) * (1 - rho_k)
        assert abs(T_k_eff - expected) < 1e-9

    def test_hdv_params_high_degradation(self):
        """HDR validation params yield >100× degradation."""
        T = 1000.0
        T_k_eff = compute_T_k_eff(T, 0.16, 0.516, 0.96)
        assert T / max(T_k_eff, 1e-9) > 100

    def test_degradation_factor_bounds(self):
        """Degradation factor is in [0, 1]."""
        for pi in [0.05, 0.16, 0.50]:
            for pm in [0.0, 0.3, 0.8]:
                for rho in [0.5, 0.72, 0.96]:
                    d = compute_degradation_factor(pi, pm, rho)
                    assert 0.0 <= d <= 1.0

    def test_monotone_in_rho(self):
        """Higher rho → lower T_k_eff."""
        T_low_rho = compute_T_k_eff(1000, 0.16, 0.5, 0.72)
        T_high_rho = compute_T_k_eff(1000, 0.16, 0.5, 0.96)
        assert T_high_rho < T_low_rho

    def test_monotone_in_p_miss(self):
        """Higher missingness → lower T_k_eff."""
        T_low_miss = compute_T_k_eff(1000, 0.16, 0.20, 0.96)
        T_high_miss = compute_T_k_eff(1000, 0.16, 0.80, 0.96)
        assert T_high_miss < T_low_miss

    def test_zero_pi_gives_zero(self):
        assert compute_T_k_eff(1000, 0.0, 0.5, 0.9) == pytest.approx(0.0)

    def test_omega_min_positive(self):
        omega = compute_omega_min(n_theta=72)
        assert omega > 0.0


class TestMuBarRequired:
    """Proposition 9.1 — Quantified ISS bound."""

    def test_returns_valid_probability(self):
        mu_bar = compute_mu_bar_required(0.5, 0.04, 0.24, 0.1, 1.5)
        assert 0.0 <= mu_bar <= 1.0

    def test_larger_epsilon_gives_larger_mu_bar(self):
        mu1 = compute_mu_bar_required(0.2, 0.04, 0.24, 0.1, 1.5)
        mu2 = compute_mu_bar_required(0.5, 0.04, 0.24, 0.1, 1.5)
        assert mu2 > mu1

    def test_zero_mismatch_returns_one(self):
        mu_bar = compute_mu_bar_required(0.5, 0.1, 0.0, 0.0, 0.0)
        assert mu_bar == pytest.approx(1.0)

    def test_iss_residual_monotone_in_mu_bar(self):
        r1 = compute_iss_residual(0.1, 0.04, 0.24, 0.1, 1.5)
        r2 = compute_iss_residual(0.5, 0.04, 0.24, 0.1, 1.5)
        assert r2 > r1 > 0


class TestBrierReliability:
    """Definition 8.2 — Brier reliability decomposition."""

    def test_identity_check(self):
        rng = np.random.default_rng(42)
        y_true = (rng.uniform(size=500) < 0.16).astype(float)
        y_prob = np.clip(rng.beta(1, 5, 500), 1e-4, 1 - 1e-4)
        decomp = brier_reliability(y_true, y_prob, n_bins=10)
        # Brier ≈ Reliability - Resolution + Uncertainty (up to finite-bin error)
        lhs = decomp["brier_score"]
        rhs = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert abs(lhs - rhs) < 0.05

    def test_perfect_calibration_has_zero_reliability(self):
        """A perfectly calibrated predictor has Reliability = 0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        # Predict exactly the bin frequency
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        decomp = brier_reliability(y_true, y_prob, n_bins=2)
        assert decomp["reliability"] < 0.05

    def test_all_fields_present(self):
        rng = np.random.default_rng(0)
        y_true = (rng.uniform(size=200) < 0.3).astype(float)
        y_prob = rng.uniform(size=200)
        decomp = brier_reliability(y_true, y_prob)
        for key in ["brier_score", "reliability", "resolution", "uncertainty", "n_samples"]:
            assert key in decomp


class TestPARobust:
    """Definition 8.1 — Calibration-adjusted Mode B threshold."""

    def test_inflates_threshold(self):
        p_A_robust = compute_p_A_robust(0.70, 1.0, 0.08)
        assert p_A_robust > 0.70

    def test_zero_reliability_unchanged(self):
        assert compute_p_A_robust(0.70, 1.0, 0.0) == pytest.approx(0.70)

    def test_clipped_to_0_99(self):
        assert compute_p_A_robust(0.70, 10.0, 0.50) <= 0.99

    def test_monotone_in_R_brier(self):
        p1 = compute_p_A_robust(0.70, 1.0, 0.02)
        p2 = compute_p_A_robust(0.70, 1.0, 0.10)
        assert p2 > p1


class TestEpsilonH:
    """Theorem H.10 — Horizon truncation error."""

    def test_decreasing_in_H(self):
        eps6 = compute_epsilon_H(0.412, 6)
        eps12 = compute_epsilon_H(0.412, 12)
        assert eps12 < eps6

    def test_decreasing_in_rho_star(self):
        eps_low = compute_epsilon_H(0.2, 6)
        eps_high = compute_epsilon_H(0.8, 6)
        assert eps_high > eps_low

    def test_benchmark_b_parameters(self):
        """For Benchmark B (rho*=0.412, H=6): eps_H ≈ 0.005."""
        eps = compute_epsilon_H(0.412, 6)
        assert eps < 0.01

    def test_suboptimality_bound_includes_eps_H(self):
        bound_full = compute_mode_b_suboptimality_bound(0.016, 0.02, 6, 0.412)
        bound_no_eps = 2 * 0.016 + 0.02 * 6
        assert bound_full > bound_no_eps - 1e-9


class TestICIState:
    """Full ICI state vector computation."""

    def test_all_conditions_off_when_clean(self):
        state = compute_ici_state(
            mu_hat=0.01,
            mu_bar_required=0.50,
            R_brier=0.001,
            R_brier_max=0.05,
            T_k_eff_per_basin=[50.0, 10.0, 5.0],
            omega_min=1.0,
        )
        assert not state["condition_i"]
        assert not state["condition_ii"]
        assert not state["condition_iii"]
        assert not state["mode_c_recommended"]

    def test_condition_i_triggers(self):
        state = compute_ici_state(
            mu_hat=0.70,
            mu_bar_required=0.30,
            R_brier=0.001,
            R_brier_max=0.05,
            T_k_eff_per_basin=[50.0, 10.0],
            omega_min=1.0,
        )
        assert state["condition_i"]
        assert state["mode_c_recommended"]

    def test_condition_iii_triggers(self):
        state = compute_ici_state(
            mu_hat=0.01,
            mu_bar_required=0.50,
            R_brier=0.001,
            R_brier_max=0.05,
            T_k_eff_per_basin=[50.0, 0.1, 5.0],  # basin 1 below ω_min
            omega_min=1.0,
        )
        assert state["condition_iii"]
        assert state["worst_basin_idx"] == 1
        assert state["mode_c_recommended"]


class TestCalibration:
    """Posterior calibration via isotonic regression."""

    def test_calibration_map_monotone(self):
        rng = np.random.default_rng(0)
        y_true = (rng.uniform(size=400) < 0.2).astype(float)
        y_prob = rng.uniform(size=400)
        cal_map = isotonic_calibrate(y_true, y_prob, n_bins=10)
        # Monotone non-decreasing
        for i in range(len(cal_map) - 1):
            assert cal_map[i] <= cal_map[i + 1] + 1e-9

    def test_apply_calibration_shape(self):
        cal_map = np.linspace(0.0, 1.0, 10)
        y_raw = np.array([0.05, 0.25, 0.55, 0.85, 0.99])
        y_cal = apply_calibration(y_raw, cal_map)
        assert y_cal.shape == y_raw.shape

    def test_calibration_reduces_reliability(self):
        """Calibration should reduce Brier reliability component."""
        rng = np.random.default_rng(7)
        n = 600
        y_true = (rng.uniform(size=n) < 0.16).astype(float)
        # Deliberately miscalibrated (overconfident)
        raw_logit = rng.normal(scale=2.0, size=n)
        raw_logit[y_true == 1] += 2.0
        y_prob = 1.0 / (1.0 + np.exp(-raw_logit))
        split = n // 2
        cal_map = isotonic_calibrate(y_true[:split], y_prob[:split])
        y_cal = apply_calibration(y_prob[split:], cal_map)
        r_before = brier_reliability(y_true[split:], y_prob[split:])["reliability"]
        r_after = brier_reliability(y_true[split:], y_cal)["reliability"]
        # Calibration should reduce (or maintain) reliability
        assert r_after <= r_before + 0.05  # allow small tolerance
