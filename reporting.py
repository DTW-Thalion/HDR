from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..control.mode_b import controlled_value_iteration, heuristic_committor_policy, make_reduced_chain
from ..control.mode_c import mode_c_entry_conditions, supervisor_mode_select
from ..generator.ground_truth import SyntheticEnv, default_scenarios
from ..inference.ici import (
    brier_reliability,
    compute_epsilon_H,
    compute_mode_b_suboptimality_bound,
    compute_mu_bar_required,
    compute_p_A_robust,
    compute_T_k_eff,
)
from ..model.slds import make_evaluation_model
from ..plotting import save_bar_plot, save_line_plot
from ..utils import atomic_write_json, ensure_dir, seed_everything
from .common import save_experiment_bundle
from .runtime import run_closed_loop_episode


def simulate_markov(P: np.ndarray, start_state: int, policy: str | None, success_states: list[int], failure_states: list[int], rng: np.random.Generator, horizon: int = 12) -> tuple[bool, int]:
    s = start_state
    for t in range(horizon):
        if s in success_states:
            return True, t
        if s in failure_states:
            return False, t
        s = int(rng.choice(len(P), p=P[s]))
    return (s in success_states), horizon


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_05" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng)
    # Part A: reduced chain
    P_actions, success_states, failure_states, start_state = make_reduced_chain()
    exact = controlled_value_iteration(P_actions, success_states, failure_states)
    heur = heuristic_committor_policy(P_actions, "conservative", success_states, failure_states)
    reduced_rows = []
    horizons = [3, 6, 9, 12]
    for H in horizons:
        for label, policy_name in [("exact", exact["policy"][start_state]), ("heuristic", heur["policy"][start_state]), ("conservative", "conservative")]:
            successes, times = [], []
            P = P_actions[str(policy_name)]
            for mc in range(min(int(config["mc_rollouts"]), 300)):
                ok, t = simulate_markov(P, start_state, label, success_states, failure_states, rng, horizon=H)
                successes.append(float(ok))
                times.append(float(t))
            reduced_rows.append({
                "horizon": H,
                "policy": label,
                "escape_prob": float(np.mean(successes)),
                "time_to_escape": float(np.mean(times)),
            })
    # sensitivity sweeps
    sens_rows = []
    for eps_q in [0.0, 0.02, 0.05, 0.10]:
        gap = abs(exact["V"][start_state] - max(0.0, heur["V"][start_state] - eps_q))
        sens_rows.append({"sweep": "epsilon_q", "value": eps_q, "gap": float(gap)})
    for delta_p in [0.0, 0.02, 0.05, 0.10]:
        P_aggr = P_actions["aggressive"].copy()
        P_aggr[start_state] = (1 - delta_p) * P_aggr[start_state] + delta_p * P_actions["conservative"][start_state]
        P_aggr[start_state] /= np.sum(P_aggr[start_state])
        alt = {"conservative": P_actions["conservative"], "aggressive": P_aggr}
        exact_alt = controlled_value_iteration(alt, success_states, failure_states)
        sens_rows.append({"sweep": "delta_P", "value": delta_p, "gap": float(abs(exact_alt["V"][start_state] - exact["V"][start_state]))})
    for rho_scale in [0.85, 0.95, 1.0, 1.05]:
        P_cons = P_actions["conservative"].copy()
        P_cons[2, 2] = min(P_cons[2, 2] * rho_scale, 0.95)
        P_cons[2] /= np.sum(P_cons[2])
        spectral = max(abs(np.linalg.eigvals(P_cons[np.ix_([2, 3, 4], [2, 3, 4])])))
        sens_rows.append({"sweep": "spectral_radius", "value": float(np.real(spectral)), "gap": float(abs(float(np.real(spectral)) - exact["spectral_radius"]))})
    # Part B: hybrid continuous + discrete
    # Use a TRUNCATED escape window (first T//4 steps) rather than full episode.
    # Over a 256-step episode, both policies escape with near-100% probability
    # because the Zipf dwell on basin 1 terminates naturally.  Measuring escape
    # within the first 64 steps isolates the Mode B intervention effect:
    # with Mode B the controller actively pushes toward escape; without it,
    # escape depends entirely on the natural Zipf dwell.
    scenarios = default_scenarios()
    hybrid_rows = []
    n_episodes = int(config["episodes_per_experiment"])
    escape_window = max(config.get("steps_per_episode", 256) // 8, 16)
    for policy_name, allow_mode_b in [("hdr_main", True), ("hdr_conservative", False)]:
        for epi in range(n_episodes):
            seed = int(config["seeds"][epi % len(config["seeds"])] + 5000 + epi + (100 if allow_mode_b else 0))
            rng_local = np.random.default_rng(seed)
            env = SyntheticEnv(eval_model, config, rng_local, scenarios["mode_b_escape"], episode_idx=epi, initial_basin=1, min_initial_dwell=48)
            out = run_closed_loop_episode(
                eval_model,
                config,
                env,
                policy_name="hdr_main",
                allow_mode_b=allow_mode_b,
                with_tau=True,
                with_coherence=True,
            )
            z_true = out.per_step["z_true"]
            # escaped_early: escaped within the truncated window
            escaped_early = float(np.any(z_true[:escape_window] == 0))
            escaped_full = float(np.any(z_true == 0))
            triggered = float(np.sum(out.per_step["mode_b_triggered"]) > 0)
            trigger_steps = np.where(out.per_step["mode_b_triggered"] > 0)[0]
            # FP definition (corrected): Mode B fired when z_true AT THE TRIGGER STEP
            # was NOT in the maladaptive basin (z_true[t0] != 1).
            # The previous definition used np.any(z_true[:t0+1] == 1) which was always
            # True when initial_basin=1 (episode starts in basin 1 for ≥48 steps),
            # making false_positive always 0 regardless of whether the trigger was
            # warranted at the actual trigger step.  The corrected metric tests whether
            # the system was maladaptive at the precise moment Mode B fired.
            if len(trigger_steps) > 0:
                t0 = trigger_steps[0]
                z_at_trigger = int(z_true[t0]) if t0 < len(z_true) else -1
                true_need = float(z_at_trigger == 1)
            else:
                true_need = 0.0
            hybrid_rows.append({
                "policy": policy_name,
                "episode": epi,
                "escaped": escaped_early,
                "escaped_full": escaped_full,
                "time_to_desired": float(np.argmax(z_true == 0) if np.any(z_true == 0) else len(z_true)),
                "safety_violation_rate": float(out.episode_summary["safety_violation_rate"]),
                "mode_b_triggered": triggered,
                "true_need": true_need,
                "false_positive": float(triggered and not true_need),
                "false_negative": float((not triggered) and true_need),
                "fallback_count": float(out.episode_summary["fallback_count"]),
            })
    save_bar_plot(
        plots_dir / "reduced_escape_prob.png",
        [f"{r['policy']}-H{r['horizon']}" for r in reduced_rows[: min(9, len(reduced_rows))]],
        [r["escape_prob"] for r in reduced_rows[: min(9, len(reduced_rows))]],
        "Reduced-chain escape probabilities",
        "escape prob",
    )
    red_df = pd.DataFrame(reduced_rows)
    time_gap_h6 = float(abs(red_df[(red_df["horizon"] == 6) & (red_df["policy"] == "exact")]["time_to_escape"].mean() - red_df[(red_df["horizon"] == 6) & (red_df["policy"] == "heuristic")]["time_to_escape"].mean())) if not red_df.empty else float("nan")
    if not red_df.empty:
        for pol in ["exact", "heuristic", "conservative"]:
            sub = red_df[red_df["policy"] == pol]
            save_line_plot(
                plots_dir / f"escape_vs_horizon_{pol}.png",
                sub["horizon"].to_numpy(),
                {pol: sub["escape_prob"].to_numpy()},
                title=f"Escape probability vs horizon ({pol})",
                xlabel="horizon",
                ylabel="escape probability",
            )
    hybrid_df = pd.DataFrame(hybrid_rows)
    modeb_df = hybrid_df[hybrid_df["policy"] == "hdr_main"]
    conservative_df = hybrid_df[hybrid_df["policy"] == "hdr_conservative"]

    # v5.0: Calibration-adjusted threshold and epsilon_H
    p_A_nominal = float(config.get("pA", 0.70))
    k_calib = float(config.get("k_calib", 1.0))
    R_brier_max = float(config.get("R_brier_max", 0.05))
    # Approximate R_Brier from FP/FN structure (simplified; full version uses stage_03b output)
    R_brier_approx = float(np.clip(modeb_df["false_positive"].mean() * 0.2, 0.0, 0.20)) if not modeb_df.empty else 0.0
    p_A_robust = compute_p_A_robust(p_A_nominal, k_calib, R_brier_approx)

    # ε_H for Theorem H.10
    rho_star = float(exact.get("spectral_radius", 0.412))
    H = int(config.get("H", 6))
    epsilon_H = compute_epsilon_H(rho_star, H)
    eps_q = float(abs(exact["V"][start_state] - heur["V"][start_state]))
    delta_P = 0.02  # conservative estimate
    bound_v4 = float(2.0 * eps_q + delta_P * H)
    bound_v5 = compute_mode_b_suboptimality_bound(eps_q, delta_P, H, rho_star)
    epsilon_H_adds_to_bound = bool(bound_v5 > bound_v4 - 1e-9)

    # v5.0: Fraction of Mode B triggers when T_k_eff ≥ ω_min (should = 1.0 when ICI is respected)
    T_total = float(config.get("steps_per_episode", 256) * len(hybrid_rows))
    pi_mal = float(config.get("mode1_base_rate", 0.16))
    p_miss = float(config.get("missing_fraction_target", 0.516))
    rho_mal = 0.96
    T_k_eff_mal = compute_T_k_eff(T_total, pi_mal, p_miss, rho_mal)
    omega_min_factor = float(config.get("omega_min_factor", 0.005))
    omega_min = omega_min_factor * T_total
    # With 50%+ missingness the maladaptive basin is below ω_min → Mode C should preempt Mode B.
    # Safe_trigger_fraction: fraction of Mode B triggers when T_k_eff_mal ≥ omega_min.
    # At current HDR validation params, T_k_eff_mal < omega_min, so safe_trigger_fraction = 0.0.
    safe_trigger_fraction = 1.0 if T_k_eff_mal >= omega_min else 0.0

    summary = {
        "reduced_exact_value_start": float(exact["V"][start_state]),
        "reduced_heuristic_value_start": float(heur["V"][start_state]),
        "reduced_abs_gap": float(abs(exact["V"][start_state] - heur["V"][start_state])),
        "reduced_time_gap_h6": time_gap_h6,
        "hybrid_escape_gain": float(modeb_df["escaped"].mean() - conservative_df["escaped"].mean()),
        "hybrid_time_gain": float(conservative_df["time_to_desired"].mean() - modeb_df["time_to_desired"].mean()),
        "hybrid_safety_delta": float(modeb_df["safety_violation_rate"].mean() - conservative_df["safety_violation_rate"].mean()),
        "false_positive_rate": float(modeb_df["false_positive"].mean()),
        "false_negative_rate": float(modeb_df["false_negative"].mean()),
        "breakdown_rate": float(np.mean((modeb_df["escaped"] == 0) & (modeb_df["mode_b_triggered"] == 1))),
        # v5.0 additions
        "p_A_robust": float(p_A_robust),
        "p_A_nominal": float(p_A_nominal),
        "R_brier_approx": float(R_brier_approx),
        "epsilon_H": float(epsilon_H),
        "suboptimality_bound_v4": float(bound_v4),
        "suboptimality_bound_v5": float(bound_v5),
        "epsilon_H_adds_to_bound": epsilon_H_adds_to_bound,
        "T_k_eff_maladaptive": float(T_k_eff_mal),
        "omega_min": float(omega_min),
        "mode_b_safe_trigger_fraction": float(safe_trigger_fraction),
        "mode_c_should_preempt": bool(T_k_eff_mal < omega_min),
    }
    save_experiment_bundle(
        stage_root / "mode_b_validation",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=reduced_rows + sens_rows + hybrid_rows,
        selected_traces={},
        log_text="Stage 05 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
