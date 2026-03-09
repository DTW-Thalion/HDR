from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..control.mode_b import committor, controlled_value_iteration, heuristic_committor_policy, make_reduced_chain
from ..inference.ici import (
    brier_reliability,
    compute_degradation_factor,
    compute_epsilon_H,
    compute_iss_residual,
    compute_mode_b_suboptimality_bound,
    compute_mu_bar_required,
    compute_omega_min,
    compute_p_A_robust,
    compute_T_k_eff,
)
from ..metrics import fit_linear_relationship
from ..model.coherence import coherence_penalty
from ..model.hsmm import hazard_at
from ..model.recovery import dare_terminal_cost, lyapunov_cost, tau_sandwich, tau_tilde
from ..model.safety import chance_tightening, gaussian_calibration_toy
from ..model.slds import make_evaluation_model
from ..model.target_set import build_target_set
from ..plotting import save_bar_plot, save_calibration_plot, save_line_plot, save_scatter_plot
from ..utils import atomic_write_json, atomic_write_text, ensure_dir, seed_everything
from .common import save_experiment_bundle, summarize_metric_rows


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_01" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng)
    target = build_target_set(0, config)
    x = rng.normal(size=config["state_dim"])
    proj_box = target.project_box(x)
    proj_exact = target.project_exact_slsqp(x)
    tau_rows = []
    xs = rng.normal(size=(64, config["state_dim"]))
    for basin in eval_model.basins[: min(3, len(eval_model.basins))]:
        for xi in xs:
            sand = tau_sandwich(basin.A, np.eye(config["state_dim"]), xi, target, basin.rho)
            tau_rows.append({"rho": basin.rho, **sand})
    # committor and exact value iteration
    P_actions, success_states, failure_states, start_state = make_reduced_chain()
    q = committor(P_actions["conservative"], failure_states, success_states)
    vi = controlled_value_iteration(P_actions, success_states, failure_states)
    heur = heuristic_committor_policy(P_actions, "conservative", success_states, failure_states)
    # chance-constraint toy
    toy = gaussian_calibration_toy(alpha=float(config["alpha_i"]), n_samples=4000, rng=rng)
    # HSMM hazards
    hazards = [dm.hazard()[:32] for dm in eval_model.dwell_models[: min(3, len(eval_model.dwell_models))]]
    # DARE ingredients
    Pterm, Kterm = dare_terminal_cost(eval_model.basins[0].A, eval_model.basins[0].B, np.eye(config["state_dim"]), np.eye(config["control_dim"]))
    # plots
    save_scatter_plot(
        plots_dir / "tau_tilde_vs_tau_L.png",
        np.array([r["tau_tilde"] for r in tau_rows]),
        np.array([r["tau_L"] for r in tau_rows]),
        title="Recovery surrogate vs Lyapunov cost",
        xlabel="tau_tilde",
        ylabel="tau_L",
    )
    save_line_plot(
        plots_dir / "hazards.png",
        np.arange(len(hazards[0])) + 1,
        {f"basin_{i}": hz for i, hz in enumerate(hazards)},
        title="HSMM hazard curves",
        xlabel="dwell length",
        ylabel="hazard",
    )
    save_bar_plot(
        plots_dir / "committor_values.png",
        [str(i) for i in range(len(q))],
        q.tolist(),
        title="Reduced-chain committor",
        ylabel="q",
    )
    cal_rows = []
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        cal_rows.append({"bin": i, "p_mean": (edges[i] + edges[i+1]) / 2.0, "empirical": (edges[i] + edges[i+1]) / 2.0, "count": 1})
    import pandas as pd
    save_calibration_plot(plots_dir / "toy_calibration.png", pd.DataFrame(cal_rows), title="Ideal calibration reference")
    # pytest
    pytest_out = subprocess.run(
        ["python", "-m", "pytest", "tests"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_write_text(stage_root / "pytest_results.txt", pytest_out.stdout + "\n" + pytest_out.stderr)
    # ── ICI formula unit tests (v5.0) ─────────────────────────────────────────
    # Test 1: T_k_eff formula verification — confirm 330× degradation for HDR params
    T_test = 1000.0
    pi_k_mal = float(config.get("mode1_base_rate", 0.16))
    p_miss_test = float(config.get("missing_fraction_target", 0.516))
    rho_mal = 0.96
    T_k_eff_mal = compute_T_k_eff(T_test, pi_k_mal, p_miss_test, rho_mal)
    degrad_factor = compute_degradation_factor(pi_k_mal, p_miss_test, rho_mal)
    # Expected: ~0.003 → 330× degradation
    expected_degrad = pi_k_mal * (1 - p_miss_test) * (1 - rho_mal)
    T_k_eff_formula_ok = bool(abs(degrad_factor - expected_degrad) < 1e-9)
    T_k_eff_high_degradation = bool(T_test / max(T_k_eff_mal, 1e-9) > 100)  # > 100× degradation

    # Test 2: μ̄_required formula unit test
    mu_bar = compute_mu_bar_required(epsilon_control=0.5, alpha=0.04, delta_A=0.24, delta_B=0.1, K_lqr_norm=1.5)
    mu_bar_ok = bool(0.0 < mu_bar < 1.0)

    # Test 3: ISS residual is monotone in μ̄
    res_lo = compute_iss_residual(0.10, 0.04, 0.24, 0.1, 1.5)
    res_hi = compute_iss_residual(0.50, 0.04, 0.24, 0.1, 1.5)
    iss_residual_monotone = bool(res_hi > res_lo > 0)

    # Test 4: Brier reliability decomposition self-consistency
    # Brier = Reliability - Resolution + Uncertainty
    y_toy = (rng.uniform(size=500) < 0.16).astype(float)
    p_toy = np.clip(rng.beta(1, 5, size=500), 1e-4, 1 - 1e-4)
    brier_decomp = brier_reliability(y_toy, p_toy, n_bins=10)
    brier_identity_ok = bool(
        abs(brier_decomp["brier_score"] - (brier_decomp["reliability"] - brier_decomp["resolution"] + brier_decomp["uncertainty"])) < 0.05
    )

    # Test 5: p_A_robust ≥ p_A_nominal
    p_A_test = compute_p_A_robust(0.70, 1.0, 0.08)
    p_A_robust_inflated = bool(p_A_test > 0.70)

    # Test 6: ε_H = ρ*^H is monotone decreasing in H
    eps_H_h6 = compute_epsilon_H(0.412, 6)
    eps_H_h12 = compute_epsilon_H(0.412, 12)
    epsilon_H_monotone = bool(eps_H_h12 < eps_H_h6)

    # Test 7: Full suboptimality bound includes ε_H term and is ≥ 2ε_q + δ_P·H
    bound_full = compute_mode_b_suboptimality_bound(0.016, 0.02, 6, 0.412)
    bound_without_eps_H = 2 * 0.016 + 0.02 * 6
    suboptimality_bound_includes_eps_H = bool(bound_full >= bound_without_eps_H)

    # ── End ICI unit tests ─────────────────────────────────────────────────────

    rows = []
    rows.extend(tau_rows)
    rows.append({"rho": np.nan, "tau_tilde": np.nan, "tau_L": np.nan, "committor_start": float(q[start_state]), "value_start": float(vi["V"][start_state]), "heuristic_start": float(heur["V"][start_state])})
    tau_rank_corr = float(spearmanr([r["tau_tilde"] for r in tau_rows], [r["tau_L"] for r in tau_rows]).correlation)
    summary = summarize_metric_rows(rows)
    summary.update({
        "projection_box_inside": bool(target.contains_box(proj_box)),
        "projection_exact_inside_or_fallback": bool(target.contains_exact(proj_exact) or target.fallback_used),
        "chance_calibration_nominal": toy["nominal"],
        "chance_calibration_empirical": toy["empirical"],
        "chance_calibration_abs_error": toy["abs_error"],
        "committor_bounds_ok": bool(np.all((q >= -1e-8) & (q <= 1 + 1e-8))),
        "bellman_start_value": float(vi["V"][start_state]),
        "bellman_heuristic_gap": float(abs(vi["V"][start_state] - heur["V"][start_state])),
        "coherence_zero_in_band": coherence_penalty(0.6, config["kappa_lo"], config["kappa_hi"]) == 0.0,
        "coherence_monotone_outside": coherence_penalty(0.9, config["kappa_lo"], config["kappa_hi"]) > coherence_penalty(0.8, config["kappa_lo"], config["kappa_hi"]),
        "tau_rank_corr": tau_rank_corr,
        "dare_terminal_trace": float(np.trace(Pterm)),
        "delay_screen_status": "skipped_no_lmi_solver",
        "pytest_return_code": pytest_out.returncode,
        # v5.0 ICI formula checks
        "ici_T_k_eff_formula_ok": T_k_eff_formula_ok,
        "ici_T_k_eff_high_degradation": T_k_eff_high_degradation,
        "ici_T_k_eff_maladaptive": float(T_k_eff_mal),
        "ici_degradation_factor": float(degrad_factor),
        "ici_mu_bar_required_ok": mu_bar_ok,
        "ici_mu_bar_required": float(mu_bar),
        "ici_iss_residual_monotone": iss_residual_monotone,
        "ici_brier_identity_ok": brier_identity_ok,
        "ici_p_A_robust_inflated": p_A_robust_inflated,
        "ici_p_A_robust": float(p_A_test),
        "ici_epsilon_H_h6": float(eps_H_h6),
        "ici_epsilon_H_monotone": epsilon_H_monotone,
        "ici_suboptimality_bound_includes_eps_H": suboptimality_bound_includes_eps_H,
    })
    save_experiment_bundle(
        stage_root / "math_checks",
        config=config,
        seed=config["seeds"][0],
        summary=summary,
        metrics_rows=rows,
        selected_traces={
            "proj_box": proj_box,
            "proj_exact": proj_exact,
            "hazards": np.asarray(hazards, dtype=float),
            "q": q,
        },
        log_text="Stage 01 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
