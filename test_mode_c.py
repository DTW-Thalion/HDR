"""
Unit tests for Mode C (Identification Mode) — HDR v7.3.
Covers Proposition 7.1 (well-posedness), supervisor logic, and ModeCTracker.
"""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.control.mode_c import (
    ModeCTracker,
    fisher_information_proxy,
    mode_c_action,
    mode_c_entry_conditions,
    supervisor_mode_select,
)


class TestModeCAction:
    """Proposition 7.1 — Mode C action well-posedness."""

    def test_solution_exists(self):
        rng = np.random.default_rng(0)
        x_hat = rng.normal(size=8)
        u = mode_c_action(x_hat, control_dim=8, sigma_dither=0.08, rng=rng, used_burden=0.0, budget=28.0)
        assert u.shape == (8,)
        assert np.any(np.abs(u) > 1e-8)

    def test_clipped_to_u_max(self):
        rng = np.random.default_rng(1)
        x_hat = rng.normal(size=8) * 10
        u = mode_c_action(x_hat, control_dim=8, sigma_dither=0.5, rng=rng, used_burden=0.0, budget=28.0)
        assert np.all(np.abs(u) <= 0.35 + 1e-8)

    def test_budget_exhaustion_reduces_action(self):
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        x_hat = np.zeros(8)
        u_full = mode_c_action(x_hat, 8, 0.1, rng1, used_burden=0.0, budget=28.0)
        u_empty = mode_c_action(x_hat, 8, 0.1, rng2, used_burden=27.9, budget=28.0)
        assert np.linalg.norm(u_empty) < np.linalg.norm(u_full) + 1e-6


class TestFisherProxy:
    """Fisher information proxy (persistent excitation criterion)."""

    def test_more_diverse_inputs_higher_proxy(self):
        rng = np.random.default_rng(10)
        # Diverse regressors
        R_diverse = rng.normal(size=(50, 8))
        # Repetitive regressors
        r_vec = rng.normal(size=8)
        R_repetitive = np.tile(r_vec, (50, 1)) + rng.normal(size=(50, 8)) * 0.01
        assert fisher_information_proxy(R_diverse) >= fisher_information_proxy(R_repetitive)

    def test_empty_returns_zero(self):
        assert fisher_information_proxy(np.zeros((1, 8))) == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        R = rng.normal(size=(20, 5))
        assert fisher_information_proxy(R) >= 0.0


class TestModeCEntryConditions:
    """Entry condition logic."""

    def test_no_conditions_triggered(self):
        result = mode_c_entry_conditions(
            mu_hat=0.10, mu_bar_required=0.50,
            R_brier=0.01, R_brier_max=0.05,
            T_k_eff_per_basin=[100.0, 50.0], omega_min=1.0,
        )
        assert not result["any_triggered"]
        assert not result["condition_i"]
        assert not result["condition_iii"]

    def test_condition_i_triggers(self):
        result = mode_c_entry_conditions(
            mu_hat=0.70, mu_bar_required=0.30,
            R_brier=0.01, R_brier_max=0.05,
            T_k_eff_per_basin=[100.0, 50.0], omega_min=1.0,
        )
        assert result["condition_i"]
        assert result["any_triggered"]

    def test_condition_ii_triggers(self):
        result = mode_c_entry_conditions(
            mu_hat=0.10, mu_bar_required=0.50,
            R_brier=0.10, R_brier_max=0.05,
            T_k_eff_per_basin=[100.0, 50.0], omega_min=1.0,
        )
        assert result["condition_ii"]
        assert result["any_triggered"]

    def test_condition_iii_triggers(self):
        result = mode_c_entry_conditions(
            mu_hat=0.10, mu_bar_required=0.50,
            R_brier=0.01, R_brier_max=0.05,
            T_k_eff_per_basin=[100.0, 0.001], omega_min=1.0,
        )
        assert result["condition_iii"]
        assert result["any_triggered"]


class TestSupervisorModeSelect:
    """Triple-mode supervisor priority: Mode C > Mode B > Mode A."""

    def test_mode_c_overrides_mode_b(self):
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": True},
            mode_b_conditions_met=True,
            mode_c_active=False,
            degradation_flag=False,
        )
        assert mode == "mode_c"

    def test_mode_b_proceeds_when_clean(self):
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": False},
            mode_b_conditions_met=True,
            mode_c_active=False,
            degradation_flag=False,
        )
        assert mode == "mode_b"

    def test_mode_a_when_no_conditions(self):
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": False},
            mode_b_conditions_met=False,
            mode_c_active=False,
            degradation_flag=False,
        )
        assert mode == "mode_a"

    def test_degradation_forces_mode_a(self):
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": True},
            mode_b_conditions_met=True,
            mode_c_active=True,
            degradation_flag=True,
        )
        assert mode == "mode_a"

    def test_mode_c_active_continues(self):
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": False},
            mode_b_conditions_met=True,
            mode_c_active=True,  # already running
            degradation_flag=False,
        )
        assert mode == "mode_c"

    def test_t_k_eff_below_threshold_preempts_mode_b(self):
        """When T_k_eff < omega_min, Mode C preempts Mode B even if ICI conditions
        appear clean (mode_c_recommended=False) and mode_c_active=False."""
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": False},
            mode_b_conditions_met=True,
            mode_c_active=False,
            degradation_flag=False,
            t_k_eff_below_threshold=True,
        )
        assert mode == "mode_c"

    def test_t_k_eff_above_threshold_allows_mode_b(self):
        """When T_k_eff >= omega_min, Mode B proceeds normally when conditions met."""
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": False},
            mode_b_conditions_met=True,
            mode_c_active=False,
            degradation_flag=False,
            t_k_eff_below_threshold=False,
        )
        assert mode == "mode_b"

    def test_degradation_still_overrides_threshold(self):
        """Degradation flag forces Mode A even when t_k_eff_below_threshold is True."""
        mode = supervisor_mode_select(
            ici_state={"mode_c_recommended": True},
            mode_b_conditions_met=True,
            mode_c_active=True,
            degradation_flag=True,
            t_k_eff_below_threshold=True,
        )
        assert mode == "mode_a"


class TestModeCTracker:
    """ModeCTracker state management."""

    def test_enters_and_tracks_steps(self):
        tracker = ModeCTracker(T_C_max=20)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[5.0, 0.1])
        assert tracker.active
        rng = np.random.default_rng(0)
        for _ in range(5):
            tracker.tick(u=rng.normal(size=8), x_hat=rng.normal(size=8))
        assert tracker.steps_in_mode_c == 5

    def test_degradation_flag_at_T_C_max(self):
        tracker = ModeCTracker(T_C_max=5)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[0.001])
        rng = np.random.default_rng(0)
        for _ in range(6):
            tracker.tick(u=rng.normal(size=8), x_hat=rng.normal(size=8))
        assert tracker.degradation_flag

    def test_fisher_proxy_increases_with_data(self):
        tracker = ModeCTracker(T_C_max=50)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[0.001])
        rng = np.random.default_rng(42)
        proxy_before = tracker.fisher_proxy
        for _ in range(20):
            tracker.tick(u=rng.normal(size=8), x_hat=rng.normal(size=8))
        proxy_after = tracker.fisher_proxy
        assert proxy_after >= proxy_before

    def test_exit_clears_state(self):
        tracker = ModeCTracker(T_C_max=50)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[0.001])
        tracker.exit()
        assert not tracker.active
        assert tracker.steps_in_mode_c == 0
        assert len(tracker.regressors) == 0

    def test_should_exit_when_all_conditions_met(self):
        tracker = ModeCTracker(T_C_max=50)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[0.001])
        assert tracker.should_exit(
            mu_hat=0.01,
            mu_bar_required=0.50,
            R_brier=0.01,
            R_brier_max=0.05,
            T_k_eff_per_basin=[100.0, 50.0],
            omega_min=1.0,
        )

    def test_should_not_exit_when_conditions_not_met(self):
        tracker = ModeCTracker(T_C_max=50)
        tracker.enter(step=0, R_brier=0.09, T_k_eff_per_basin=[0.001])
        assert not tracker.should_exit(
            mu_hat=0.70,
            mu_bar_required=0.30,
            R_brier=0.10,
            R_brier_max=0.05,
            T_k_eff_per_basin=[0.001],
            omega_min=10.0,
        )
