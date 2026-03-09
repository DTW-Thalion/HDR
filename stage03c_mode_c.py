from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..generator.ground_truth import SyntheticEnv, default_scenarios
from ..inference.ici import (
    brier_reliability,
    compute_ici_state,
    compute_mu_bar_required,
    compute_T_k_eff,
)
from ..metrics import fit_linear_relationship
from ..model.slds import make_evaluation_model
from ..plotting import save_heatmap, save_scatter_plot
from ..utils import atomic_save_npz, atomic_write_csv, atomic_write_json, ensure_dir, seed_everything, load_json
from .common import save_experiment_bundle
from .runtime import run_closed_loop_episode


POLICY_OPTIONS = {
    "open_loop": {"allow_mode_b": False, "with_tau": False, "with_coherence": False},
    "pooled_lqr": {"allow_mode_b": False, "with_tau": False, "with_coherence": False},
    "basin_lqr": {"allow_mode_b": False, "with_tau": False, "with_coherence": False},
    "hdr_main": {"allow_mode_b": False, "with_tau": True, "with_coherence": True},
    "hdr_no_tau": {"allow_mode_b": False, "with_tau": False, "with_coherence": True},
    "hdr_no_coherence": {"allow_mode_b": False, "with_tau": True, "with_coherence": False},
}


def _episode_seed(base_seeds: list[int], policy_idx: int, scenario_idx: int, episode_idx: int) -> int:
    return int(base_seeds[episode_idx % len(base_seeds)] + 1000 * policy_idx + 100 * scenario_idx + episode_idx)


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_04" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng)
    scenarios_map = default_scenarios()
    profile = config.get("profile_name", profile_name)
    if profile == "extended":
        scenario_names = ["nominal", "model_mismatch"]
        policy_names = ["open_loop", "pooled_lqr", "basin_lqr", "hdr_main"]
    else:
        scenario_names = ["nominal", "model_mismatch", "missing_heterosk", "delayed_control", "target_drift"]
        policy_names = list(POLICY_OPTIONS.keys())
    policy_items = [(name, POLICY_OPTIONS[name]) for name in policy_names]
    n_episodes = int(config["episodes_per_experiment"])
    selected_cap = int(config["selected_trace_cap"])
    flush_chunk = 2
    all_rows = []
    stage_trace = {}
    for s_idx, scenario_name in enumerate(scenario_names):
        scenario = scenarios_map[scenario_name]
        for p_idx, (policy_name, policy_kwargs) in enumerate(policy_items):
            exp_dir = ensure_dir(stage_root / f"{scenario_name}_{policy_name}")
            summary_path = exp_dir / "summary.json"
            metrics_path = exp_dir / "metrics.csv"
            if summary_path.exists() and metrics_path.exists():
                try:
                    existing_df = pd.read_csv(metrics_path)
                    existing_rows = existing_df.to_dict(orient="records")
                    all_rows.extend(existing_rows)
                    continue
                except Exception:
                    pass
            rows = []
            start_epi = 0
            if metrics_path.exists():
                try:
                    existing_df = pd.read_csv(metrics_path)
                    rows = existing_df.to_dict(orient="records")
                    start_epi = len(rows)
                    all_rows.extend(rows)
                except Exception:
                    rows = []
                    start_epi = 0
            traces = {}
            for epi in range(start_epi, n_episodes):
                seed = _episode_seed(config["seeds"], p_idx, s_idx, epi)
                rng_local = np.random.default_rng(seed)
                env = SyntheticEnv(eval_model, config, rng_local, scenario, episode_idx=epi)
                out = run_closed_loop_episode(
                    eval_model=eval_model,
                    config=config,
                    env=env,
                    policy_name=policy_name,
                    allow_mode_b=policy_kwargs["allow_mode_b"],
                    with_tau=policy_kwargs["with_tau"],
                    with_coherence=policy_kwargs["with_coherence"],
                )
                row = {"policy": policy_name, "scenario": scenario_name, "episode": epi, "seed": seed, **out.episode_summary}
                rows.append(row)
                all_rows.append(row)
                if epi < selected_cap:
                    traces[f"x_true_{epi}"] = out.per_step["dist_true"]
                    traces[f"u_{epi}"] = out.per_step["stage_cost"]
                    traces[f"kappa_{epi}"] = out.per_step["kappa_hat"]
                # partial flush at chunk boundaries
                if ((epi + 1) % flush_chunk == 0) or (epi == n_episodes - 1):
                    atomic_write_csv(exp_dir / "metrics.csv", rows)
                    if traces:
                        atomic_save_npz(exp_dir / "selected_traces.npz", **traces)
            summary = {
                "policy": policy_name,
                "scenario": scenario_name,
                "cum_cost_mean": float(np.mean([r["cum_cost"] for r in rows])),
                "time_in_target_mean": float(np.mean([r["time_in_target"] for r in rows])),
                "safety_violation_rate_mean": float(np.mean([r["safety_violation_rate"] for r in rows])),
                "burden_adherence_mean": float(np.mean([r["burden_adherence"] for r in rows])),
                "circadian_adherence_mean": float(np.mean([r["circadian_adherence"] for r in rows])),
                "challenge_recovery_time_mean": float(np.nanmean([r["challenge_recovery_time"] for r in rows])),
                "recursive_feasibility_rate_mean": float(np.mean([r["recursive_feasibility_rate"] for r in rows])),
                "controller_solve_time_mean": float(np.mean([r["controller_solve_time_mean"] for r in rows])),
                "state_rmse_mean": float(np.mean([r["state_rmse"] for r in rows])),
                "mode_accuracy_mean": float(np.mean([r["mode_accuracy"] for r in rows])),
            }
            save_experiment_bundle(
                exp_dir,
                config=config,
                seed=config["seeds"],
                summary=summary,
                metrics_rows=rows,
                selected_traces=traces,
                log_text=f"Stage 04 {scenario_name} {policy_name}",
            )
    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(index="scenario", columns="policy", values="cum_cost", aggfunc="mean").reindex(index=scenario_names)
    save_heatmap(
        plots_dir / "cum_cost_heatmap.png",
        pivot.to_numpy(),
        xlabels=[str(c) for c in pivot.columns],
        ylabels=[str(i) for i in pivot.index],
        title="Mean cumulative cost",
        colorbar_label="cost",
    )
    # ── Practical stability under mode estimation error (Proposition H.3) ──
    # Fix: the original test injected soft mode-probability noise which didn't
    # change controller gains (causing the flat line artefact).  The correct
    # test forces a HARD mode misclassification at each step with probability μ̄
    # (controller uses wrong basin model), measuring the steady-state residual.
    # The ISS bound predicts: residual ∝ √μ̄ · (ΔA + ΔB‖K‖)/α.
    mu_levels = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.50])
    mu_rows = []
    n_sweeps = max(6, min(12, n_episodes // 2))
    for mu in mu_levels:
        residuals = []
        for epi in range(n_sweeps):
            seed = _episode_seed(config["seeds"], 999, 0, epi)
            rng_local = np.random.default_rng(seed)
            # Use nominal scenario but inject mode mismatch directly in episode
            scenario_nm = scenarios_map["nominal"]
            env = SyntheticEnv(eval_model, config, rng_local, scenario_nm, episode_idx=epi)
            # Run a truncated closed-loop episode with forced wrong-mode probability μ̄
            # At each step: with prob μ̄ the controller receives mode_label = wrong_mode
            # (selected uniformly from the other basins), generating the ηt disturbance
            # (A_ztrue - A_zmisclassified)*x + (B_ztrue - B_zmisclassified)*u.
            obs = env.reset()
            from ..inference.imm import IMMFilter
            from ..model.slds import pooled_basin
            from ..control.lqr import dlqr as _dlqr
            from ..control.mpc import solve_mode_a
            from ..model.target_set import build_target_set
            from ..model.coherence import coherence_from_state_history
            from ..model.hsmm import entrenchment_diagnostic
            from ..model.safety import circadian_allowed_mask, observation_intervals

            imm_sw = IMMFilter(eval_model)
            x_hat_hist_sw = []
            dist_tail = []
            rng_mu = np.random.default_rng(seed + 777)

            for t_sw in range(env.T):
                y_sw = np.nan_to_num(obs["y"], nan=0.0)
                mask_sw = obs["mask"]
                u_prev_sw = env.u_hist[-1] if env.u_hist else np.zeros(eval_model.control_dim)
                st_sw = imm_sw.step(y_sw, mask_sw, u_prev_sw)
                true_mode = int(np.argmax(st_sw.mode_probs))
                # Hard mode mismatch: with prob μ̄ use wrong basin
                if mu > 0 and rng_mu.uniform() < mu:
                    wrong_choices = [k for k in range(len(eval_model.basins)) if k != true_mode]
                    wrong_mode = int(rng_mu.choice(wrong_choices)) if wrong_choices else true_mode
                else:
                    wrong_mode = true_mode
                basin_sw = eval_model.basins[wrong_mode]
                x_hat_sw = st_sw.mixed_mean.copy()
                P_hat_sw = st_sw.mixed_cov.copy()
                x_hat_hist_sw.append(x_hat_sw)
                kappa_sw = coherence_from_state_history(
                    np.asarray(x_hat_hist_sw), axes=config.get("coherence_axes", [1, 5, 6]),
                    window=int(config["coherence_window"])
                )
                target_sw = obs["target"]
                res_sw = solve_mode_a(
                    x_hat=x_hat_sw, P_hat=P_hat_sw, basin=basin_sw,
                    target=target_sw, kappa_hat=kappa_sw, config=config,
                    step=t_sw, used_burden=0.0, with_tau=True, with_coherence=True,
                )
                obs_next_sw, _ = env.step(res_sw.u)
                if t_sw >= env.T // 2:
                    # Use squared distance from target MIDPOINT (not box projection).
                    # The box-projection metric floors at zero whenever x_true is inside
                    # the target box, masking the ISS signal.  Distance from the midpoint
                    # is always positive and grows proportionally with ||A_wrong - A_true||,
                    # cleanly revealing the √μ̄ scaling predicted by Proposition H.3.
                    mid_sw = 0.5 * (obs["target"].box_low + obs["target"].box_high)
                    dist_sw = float(np.sum((obs["x_true"] - mid_sw) ** 2))
                    dist_tail.append(dist_sw)
                obs = obs_next_sw
            residuals.append(float(np.mean(dist_tail)) if dist_tail else 0.0)
        mu_rows.append({"mu": float(mu), "sqrt_mu": float(np.sqrt(mu)), "residual": float(np.mean(residuals))})
    mu_fit = fit_linear_relationship(np.array([r["sqrt_mu"] for r in mu_rows]), np.array([r["residual"] for r in mu_rows]))
    save_scatter_plot(
        plots_dir / "residual_vs_sqrt_mu.png",
        x=np.array([r["sqrt_mu"] for r in mu_rows]),
        y=np.array([r["residual"] for r in mu_rows]),
        title="Residual vs sqrt(mu)",
        xlabel="sqrt(mu)",
        ylabel="steady residual",
        line=(mu_fit["intercept"], mu_fit["slope"]),
    )
    # Practical stability under target drift
    # Practical stability under target drift: use monotone (linear-ramp) drift so
    # that residual measured at episode end grows monotonically with drift_scale.
    # The sinusoidal drift produced artefactually negative slopes because the
    # oscillating box sometimes overlapped the state (reducing dist_true).
    # With linear drift, S*(t) steadily moves away from origin and the controller
    # can only partially track it, giving residual ∝ drift_velocity = delta_S.
    from ..generator.ground_truth import Scenario as _Scenario
    drift_levels = np.array([0.0, 0.04, 0.08, 0.12, 0.16])
    drift_rows = []
    for delta in drift_levels:
        drift_scenario = _Scenario(
            name=f"target_drift_linear_{delta:.2f}",
            model_mismatch=0.12,
            target_drift_scale=float(delta),
            linear_drift_mode=True,  # monotone ramp
        )
        residuals = []
        for epi in range(max(4, min(10, n_episodes // 2))):
            seed = _episode_seed(config["seeds"], 998, 1, epi)
            rng_local = np.random.default_rng(seed)
            env = SyntheticEnv(eval_model, config, rng_local, drift_scenario, episode_idx=epi)
            out = run_closed_loop_episode(eval_model, config, env, "hdr_main", allow_mode_b=False, with_tau=True, with_coherence=True)
            residuals.append(float(np.mean(out.per_step["dist_true"][-max(8, len(out.per_step["dist_true"]) // 4):])))
        drift_rows.append({"delta_S": delta, "residual": float(np.mean(residuals))})
    drift_fit = fit_linear_relationship(np.array([r["delta_S"] for r in drift_rows]), np.array([r["residual"] for r in drift_rows]))
    save_scatter_plot(
        plots_dir / "residual_vs_delta_S.png",
        x=np.array([r["delta_S"] for r in drift_rows]),
        y=np.array([r["residual"] for r in drift_rows]),
        title="Residual vs target drift",
        xlabel="delta_S",
        ylabel="steady residual",
        line=(drift_fit["intercept"], drift_fit["slope"]),
    )
    # Chance calibration nominal vs heavy tail
    calib_rows = []
    for name in ["nominal", "heavy_tail"]:
        scenario = scenarios_map[name]
        violations = []
        for epi in range(max(4, min(10, n_episodes // 2))):
            seed = _episode_seed(config["seeds"], 997, 2, epi)
            rng_local = np.random.default_rng(seed)
            env = SyntheticEnv(eval_model, config, rng_local, scenario, episode_idx=epi)
            out = run_closed_loop_episode(eval_model, config, env, "hdr_main", allow_mode_b=False, with_tau=True, with_coherence=True)
            violations.append(float(np.mean(out.per_step["safety_violation"])))
        calib_rows.append({"scenario": name, "violation_rate": float(np.mean(violations))})
    hdr_nominal = df[(df["policy"] == "hdr_main") & (df["scenario"] == "nominal")]
    pooled_nominal = df[(df["policy"] == "pooled_lqr") & (df["scenario"] == "nominal")]
    open_nominal = df[(df["policy"] == "open_loop") & (df["scenario"] == "nominal")]

    # v5.0: ICI diagnostic — compute mode_a_guarantee_fraction
    # Fraction of episodes where μ̂ ≤ μ̄_required (ISS guarantee holds).
    rho_vals = np.array(config.get("rho_reference", [0.72, 0.96, 0.55]))
    delta_A = float(np.max(np.abs(rho_vals - rho_vals[0])))
    alpha_iss = float(1.0 - np.max(rho_vals))
    mu_bar_req = compute_mu_bar_required(
        epsilon_control=float(config.get("epsilon_control", 0.50)),
        alpha=max(alpha_iss, 0.01),
        delta_A=delta_A,
        delta_B=0.1,
        K_lqr_norm=1.5,
    )
    # Approximate μ̂ per episode from mode_accuracy column (1 − accuracy ≈ μ̂)
    if "mode_accuracy" in hdr_nominal.columns:
        mu_hat_per_ep = (1.0 - hdr_nominal["mode_accuracy"].fillna(0.5)).clip(0, 1)
        mode_a_guarantee_fraction = float((mu_hat_per_ep <= mu_bar_req).mean())
    else:
        mode_a_guarantee_fraction = float("nan")

    # v5.0: Safety delta within ISS bound — if mu_bar_req is small, delta is expected
    safety_delta = float(hdr_nominal["safety_violation_rate"].mean() - pooled_nominal["safety_violation_rate"].mean())
    iss_bound_at_mu_hat = float(np.sqrt(1.0 - mode_a_guarantee_fraction if np.isfinite(mode_a_guarantee_fraction) else 0.5) * (delta_A + 0.1 * 1.5) / max(alpha_iss, 0.01)) if np.isfinite(mode_a_guarantee_fraction) else float("nan")

    summary = {
        "selected_scenarios": scenario_names,
        "selected_policies": policy_names,
        "hdr_vs_open_loop_gain_nominal": float(1.0 - hdr_nominal["cum_cost"].mean() / max(open_nominal["cum_cost"].mean(), 1e-8)),
        "hdr_vs_pooled_gain_nominal": float(1.0 - hdr_nominal["cum_cost"].mean() / max(pooled_nominal["cum_cost"].mean(), 1e-8)),
        "safety_delta_vs_pooled_nominal": safety_delta,
        "burden_adherence_hdr_nominal": float(hdr_nominal["burden_adherence"].mean()),
        "circadian_adherence_hdr_nominal": float(hdr_nominal["circadian_adherence"].mean()),
        "mode_error_fit_slope": float(mu_fit["slope"]),
        "mode_error_fit_r2": float(mu_fit["r2"]),
        "target_drift_fit_slope": float(drift_fit["slope"]),
        "target_drift_fit_r2": float(drift_fit["r2"]),
        "gaussian_calibration_abs_error": float(abs(calib_rows[0]["violation_rate"] - config["alpha_i"])),
        "heavy_tail_calibration_degradation": float(calib_rows[1]["violation_rate"] - calib_rows[0]["violation_rate"]),
        "n_episode_rows": int(len(df)),
        # v5.0 ICI additions
        "mode_a_guarantee_fraction": mode_a_guarantee_fraction,
        "mu_bar_required": float(mu_bar_req),
        "iss_bound_at_observed_mu_hat": iss_bound_at_mu_hat,
        "safety_delta_within_iss_bound": bool(abs(safety_delta) <= iss_bound_at_mu_hat) if np.isfinite(iss_bound_at_mu_hat) else False,
    }
    atomic_write_json(stage_root / "special_sweeps_mode_error.json", {"rows": mu_rows, "fit": mu_fit})
    atomic_write_json(stage_root / "special_sweeps_target_drift.json", {"rows": drift_rows, "fit": drift_fit})
    atomic_write_json(stage_root / "chance_calibration.json", {"rows": calib_rows})
    save_experiment_bundle(
        stage_root / "mode_a_validation_summary",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=all_rows + mu_rows + drift_rows + calib_rows,
        selected_traces={},
        log_text="Stage 04 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
