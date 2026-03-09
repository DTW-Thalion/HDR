"""
Stage 03b — ICI Diagnostic Pipeline — HDR v5.0
================================================

Validates the Inference–Control Interface machinery:
  03b.1  Calibration pipeline: isotonic calibration, report p_A^robust
  03b.2  μ̂ estimation vs μ̄_required — Mode C condition (i) flag
  03b.3  T_k_eff computation and regime boundary flags — condition (iii)
  03b.4  Full ICI state vector unit test (Algorithm 1, Step 5)

Depends on stage_03 outputs (mode probabilities and z_true labels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.ici import (
    apply_calibration,
    brier_reliability,
    compute_degradation_factor,
    compute_ici_state,
    compute_mu_bar_required,
    compute_omega_min,
    compute_p_A_robust,
    compute_T_k_eff,
    isotonic_calibrate,
)
from ..model.slds import make_evaluation_model
from ..plotting import save_calibration_plot, save_line_plot
from ..utils import atomic_write_json, ensure_dir, seed_everything
from .common import save_experiment_bundle


def _load_stage03_data(project_root: Path, profile_name: str) -> dict | None:
    """Load IMM outputs from stage_03 npz (if available)."""
    npz_path = project_root / "results" / "stage_03" / profile_name / "identification_validation" / "selected_traces.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_03b" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]) + 3100)
    eval_model = make_evaluation_model(config, rng)
    n = config["state_dim"]
    n_theta = n * n + n * config["control_dim"] + n  # A + B + b parameters per basin

    # ── 03b.1  Calibration pipeline ──────────────────────────────────────────
    stage03_data = _load_stage03_data(project_root, profile_name)
    if stage03_data is not None and "calibration_true" in stage03_data:
        cal_true = stage03_data["calibration_true"].astype(float)
        cal_prob = stage03_data["calibration_prob"].astype(float)
    else:
        # Synthetic stand-in: miscalibrated sigmoid-shifted probabilities
        n_samples = max(int(config.get("steps_per_episode", 256) * 2), 512)
        cal_true_bin = (rng.uniform(size=n_samples) < 0.16).astype(float)
        raw_logit = rng.normal(scale=1.2, size=n_samples)
        raw_logit[cal_true_bin == 1] += 1.0
        cal_prob = 1.0 / (1.0 + np.exp(-raw_logit * 0.6))  # deliberately miscalibrated
        cal_true = cal_true_bin

    # Split into calibration-fit and held-out halves
    n_cal = len(cal_true)
    split = n_cal // 2
    cal_fit_true, cal_fit_prob = cal_true[:split], cal_prob[:split]
    cal_held_true, cal_held_prob = cal_true[split:], cal_prob[split:]

    # Pre-calibration reliability
    brier_pre = brier_reliability(cal_held_true, cal_held_prob, n_bins=10)

    # Isotonic calibration on fit half
    cal_map = isotonic_calibrate(cal_fit_true, cal_fit_prob, n_bins=10)
    cal_held_calibrated = apply_calibration(cal_held_prob, cal_map)

    # Post-calibration reliability
    brier_post = brier_reliability(cal_held_true, cal_held_calibrated, n_bins=10)

    R_brier = float(brier_post["reliability"])
    p_A_nominal = float(config.get("pA", 0.70))
    k_calib = float(config.get("k_calib", 1.0))
    R_brier_max = float(config.get("R_brier_max", 0.05))
    p_A_robust = compute_p_A_robust(p_A_nominal, k_calib, R_brier)

    # Plot: calibration curves before and after
    cal_df_pre = pd.DataFrame({
        "bin": range(10),
        "p_mean": brier_pre["bin_confidences"],
        "empirical": brier_pre["bin_frequencies"],
        "count": brier_pre["bin_counts"],
    })
    cal_df_post = pd.DataFrame({
        "bin": range(10),
        "p_mean": brier_post["bin_confidences"],
        "empirical": brier_post["bin_frequencies"],
        "count": brier_post["bin_counts"],
    })
    save_calibration_plot(plots_dir / "calibration_pre.png", cal_df_pre, title="Pre-calibration reliability")
    save_calibration_plot(plots_dir / "calibration_post.png", cal_df_post, title="Post-calibration reliability")

    # ── 03b.2  μ̂ estimation vs μ̄_required ───────────────────────────────────
    # Estimate μ̂ from stage_03 mode predictions (fraction of timesteps with
    # wrong MAP mode assignment, approximated from F1 and base rate).
    if stage03_data is not None and "calibration_true" in stage03_data:
        # Simple mode-error rate approximation from calibration arrays
        map_threshold = 0.5
        y_pred = (cal_held_calibrated >= map_threshold).astype(int)
        mu_hat = float(np.mean(y_pred != cal_held_true.astype(int)))
    else:
        mu_hat = 1.0 - float(config.get("observer_mode_accuracy_approx", 0.55))

    # ISS mismatch parameters from model
    rho_vals = np.array(config.get("rho_reference", [0.72, 0.96, 0.55]))
    delta_A = float(np.max(np.abs(rho_vals - rho_vals[0])))  # heuristic from spectral radii
    delta_B = 0.1  # default; in deployment computed from identified B matrices
    K_lqr_norm = 1.5  # heuristic; production system computes ‖K_LQR‖₂
    alpha = float(1.0 - np.max(rho_vals))  # stage-cost decrease rate ≈ 1 − ρ̄

    mu_bar_required = compute_mu_bar_required(
        epsilon_control=float(config.get("epsilon_control", 0.50)),
        alpha=max(alpha, 0.01),
        delta_A=delta_A,
        delta_B=delta_B,
        K_lqr_norm=K_lqr_norm,
    )
    condition_i_triggered = bool(mu_hat >= mu_bar_required)

    # ── 03b.3  T_k_eff computation and regime boundary ───────────────────────
    T_total = float(config.get("steps_per_episode", 256) * config.get("episodes_per_experiment", 5))
    pi_vals = [
        1.0 - float(config.get("mode1_base_rate", 0.16)) - 0.05,  # basin 0 (desired)
        float(config.get("mode1_base_rate", 0.16)),                  # basin 1 (maladaptive)
        0.05,                                                          # basin 2 (transient)
    ][:config.get("K", 3)]
    p_miss = float(config.get("missing_fraction_target", 0.516))
    rho_per_basin = list(config.get("rho_reference", [0.72, 0.96, 0.55]))[:config.get("K", 3)]

    T_k_eff_per_basin = [
        compute_T_k_eff(T_total, pi_vals[k], p_miss, rho_per_basin[k])
        for k in range(config.get("K", 3))
    ]
    omega_min_factor = float(config.get("omega_min_factor", 0.005))
    omega_min = omega_min_factor * T_total  # practical threshold as fraction of T

    regime_flags = [t < omega_min for t in T_k_eff_per_basin]
    condition_iii_triggered = any(regime_flags)
    degradation_factors = [
        compute_degradation_factor(pi_vals[k], p_miss, rho_per_basin[k])
        for k in range(config.get("K", 3))
    ]

    # Plot: T_k_eff per basin vs omega_min
    save_line_plot(
        plots_dir / "T_k_eff_per_basin.png",
        np.arange(len(T_k_eff_per_basin)),
        {"T_k_eff": np.array(T_k_eff_per_basin), "omega_min": np.full(len(T_k_eff_per_basin), omega_min)},
        title="Effective sample count per basin vs regime boundary ω_min",
        xlabel="basin index",
        ylabel="T_k_eff",
    )

    # ── 03b.4  Full ICI state vector unit test ────────────────────────────────
    ici_state = compute_ici_state(
        mu_hat=mu_hat,
        mu_bar_required=mu_bar_required,
        R_brier=R_brier,
        R_brier_max=R_brier_max,
        T_k_eff_per_basin=T_k_eff_per_basin,
        omega_min=omega_min,
    )
    # Unit test: ICI conditions match individually computed flags
    ici_conditions_consistent = bool(
        ici_state["condition_i"] == condition_i_triggered
        and ici_state["condition_iii"] == condition_iii_triggered
        and ici_state["mode_c_recommended"] == (condition_i_triggered or bool(R_brier >= R_brier_max) or condition_iii_triggered)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = []
    for k in range(config.get("K", 3)):
        rows.append({
            "basin": k,
            "pi_k": pi_vals[k],
            "rho_k": rho_per_basin[k],
            "T_k_eff": T_k_eff_per_basin[k],
            "omega_min": omega_min,
            "below_omega_min": regime_flags[k],
            "degradation_factor": degradation_factors[k],
        })
    rows.append({
        "basin": -1,
        "mu_hat": mu_hat,
        "mu_bar_required": mu_bar_required,
        "R_brier_pre": brier_pre["reliability"],
        "R_brier_post": R_brier,
        "p_A_nominal": p_A_nominal,
        "p_A_robust": p_A_robust,
        "condition_i": condition_i_triggered,
        "condition_ii": bool(R_brier >= R_brier_max),
        "condition_iii": condition_iii_triggered,
        "mode_c_recommended": ici_state["mode_c_recommended"],
        "ici_conditions_consistent": ici_conditions_consistent,
    })

    summary = {
        "brier_reliability_pre": float(brier_pre["reliability"]),
        "brier_reliability_post": float(R_brier),
        "calibration_improvement": float(brier_pre["reliability"] - R_brier),
        "p_A_robust": p_A_robust,
        "p_A_nominal": p_A_nominal,
        "mu_hat": mu_hat,
        "mu_bar_required": mu_bar_required,
        "condition_i_triggered": condition_i_triggered,
        "condition_ii_triggered": bool(R_brier >= R_brier_max),
        "condition_iii_triggered": condition_iii_triggered,
        "mode_c_recommended": ici_state["mode_c_recommended"],
        "T_k_eff_maladaptive": T_k_eff_per_basin[1] if len(T_k_eff_per_basin) > 1 else float("nan"),
        "T_k_eff_desired": T_k_eff_per_basin[0],
        "omega_min": omega_min,
        "ici_conditions_consistent": ici_conditions_consistent,
        "degradation_factor_maladaptive": degradation_factors[1] if len(degradation_factors) > 1 else float("nan"),
    }

    save_experiment_bundle(
        stage_root / "ici_diagnostic",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=rows,
        selected_traces={
            "cal_held_prob": cal_held_prob,
            "cal_held_calibrated": cal_held_calibrated,
            "cal_held_true": cal_held_true,
            "T_k_eff_per_basin": np.array(T_k_eff_per_basin),
        },
        log_text="Stage 03b (ICI diagnostic) completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
