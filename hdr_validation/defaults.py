"""
HDR Validation Suite — Canonical Default Parameters
====================================================
Single source of truth for all shared configuration values.

Profile runners import from here and override only profile-specific
parameters (seeds, episodes, steps_per_episode, mc_rollouts).

When a parameter appears here AND in a profile runner, the profile
runner's value takes precedence (via dict.update). But the parameter
must ALSO appear here so that there is one place to change it.
"""
from __future__ import annotations

from typing import Any

# ── Package version (single source of truth) ──────────────────────────────
# Also declared in pyproject.toml; keep them in sync via CI gate.
HDR_VERSION = "7.4.0"

# ── Shared defaults ───────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    # Dimensions
    "state_dim": 8,
    "obs_dim": 16,
    "control_dim": 8,
    "disturbance_dim": 8,
    "K": 3,
    # Control
    "H": 6,
    "w1": 1.0,
    "w2": 0.5,
    "w3": 0.3,
    "lambda_u": 0.1,
    "alpha_i": 0.05,
    "eps_safe": 0.01,
    # Dynamics
    "rho_reference": [0.72, 0.96, 0.55],
    "max_dwell_len": 128,
    "model_mismatch_bound": 0.347,
    # Target set
    "kappa_lo": 0.55,
    "kappa_hi": 0.75,
    "pA": 0.70,
    "qmin": 0.15,
    # Safety / time
    "steps_per_day": 48,
    "dt_minutes": 30,
    "coherence_window": 24,
    "default_burden_budget": 28.0,
    "circadian_locked_controls": [5, 6],
    # ICI
    "R_brier_max": 0.05,
    "omega_min_factor": 0.005,
    "T_C_max": 50,
    "k_calib": 1.0,
    "sigma_dither": 0.08,
    "epsilon_control": 0.50,
    "missing_fraction_target": 0.516,
    "mode1_base_rate": 0.16,
    "observer_mode_accuracy_approx": 0.55,
    "w3_sweep_values": [0.05, 0.10, 0.20, 0.30, 0.50],
    # Disturbance / tube-MPC (Appendix J)
    "disturbance_beta": 0.999,
    # v7.0 extension parameters
    "n_irr": 2,
    "n_sites": 2,
    "epsilon_G": 0.02,
    "R_k_regions": 2,
    "lambda_cat_max": 0.05,
    "drift_rate": 0.001,
    "delay_steps": 10,
    "n_cum_exp": 1,
    "xi_max": 100.0,
    "n_expansion": 2,
    "delta_J_max": 0.05,
    "m_d": 1,
    "n_particles": 100,
    "n_patients": 10,
    "T_p_values": [10, 50],
    "jump_risk_threshold": 0.3,
    "irr_boundary_threshold": 0.9,
    "lambda_irr": 1.0,
}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Return a config dict = DEFAULTS merged with overrides.

    Usage in profile runners::

        from hdr_validation.defaults import make_config

        SMOKE_CONFIG = make_config(
            profile_name="smoke",
            seeds=[101],
            episodes_per_experiment=8,
            steps_per_episode=128,
            mc_rollouts=50,
            selected_trace_cap=5,
        )
    """
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg
