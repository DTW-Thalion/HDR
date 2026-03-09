from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..generator.ground_truth import SyntheticEnv, default_scenarios
from ..model.coherence import coherence_grad, coherence_penalty, settling_metrics
from ..model.slds import make_evaluation_model
from ..plotting import save_bar_plot, save_line_plot
from ..utils import atomic_write_json, ensure_dir, seed_everything
from .common import save_experiment_bundle
from .runtime import run_closed_loop_episode


def simulate_scalar_coherence(initial_kappa: float, with_penalty: bool, config: dict, T: int = 96, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    kappa = float(initial_kappa)
    traj = []
    lo = float(config["kappa_lo"])
    hi = float(config["kappa_hi"])
    target = 0.5 * (lo + hi)
    for _ in range(T):
        grad = coherence_grad(kappa, lo, hi)
        u = -0.8 * grad if with_penalty else 0.0
        kappa = np.clip(kappa + 0.12 * u - 0.06 * (kappa - target) + rng.normal(scale=0.01), 0.0, 1.0)
        traj.append(kappa)
    arr = np.asarray(traj)
    metrics = settling_metrics(arr, lo, hi)
    metrics["penalty_mean"] = float(np.mean([coherence_penalty(v, lo, hi) for v in arr]))
    metrics["final_kappa"] = float(arr[-1])
    return {"traj": arr, **metrics}


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_06" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng)
    lo, hi = float(config["kappa_lo"]), float(config["kappa_hi"])
    # standalone scalar tests
    standalone_rows = []
    standalone_traces = {}
    init_map = {"under": 0.30, "in_band": 0.65, "over": 0.90}
    for name, init in init_map.items():
        for flag in [False, True]:
            out = simulate_scalar_coherence(init, with_penalty=flag, config=config, T=max(64, config["steps_per_episode"] // 2), seed=int(1000 * init + flag))
            standalone_rows.append({"scenario": name, "with_penalty": flag, **{k: v for k, v in out.items() if k != "traj"}})
            standalone_traces[f"{name}_{int(flag)}"] = out["traj"]
    save_line_plot(
        plots_dir / "standalone_coherence_traces.png",
        np.arange(len(next(iter(standalone_traces.values())))),
        {k: v for k, v in standalone_traces.items()},
        title="Standalone coherence trajectories",
        xlabel="step",
        ylabel="kappa",
    )
    # integrated runs
    scenarios = default_scenarios()
    integrated_rows = []
    n_episodes = int(config["episodes_per_experiment"])
    for scenario_name in ["coherence_under", "coherence_over"]:
        for policy_name, with_coh in [("hdr_no_coherence", False), ("hdr_main", True)]:
            for epi in range(n_episodes):
                seed = int(config["seeds"][epi % len(config["seeds"])] + 7000 + epi + (200 if with_coh else 0))
                rng_local = np.random.default_rng(seed)
                env = SyntheticEnv(eval_model, config, rng_local, scenarios[scenario_name], episode_idx=epi)
                out = run_closed_loop_episode(
                    eval_model,
                    config,
                    env,
                    policy_name="hdr_main",
                    allow_mode_b=False,
                    with_tau=True,
                    with_coherence=with_coh,
                )
                kappa_series = out.per_step["kappa_hat"]
                met = settling_metrics(kappa_series, lo, hi)
                integrated_rows.append({
                    "scenario": scenario_name,
                    "policy": policy_name,
                    "episode": epi,
                    "cum_cost": float(out.episode_summary["cum_cost"]),
                    "time_in_band": float(met["time_in_band"]),
                    "overshoot": float(met["overshoot"]),
                    "settling_time": float(met["settling_time"]),
                })
    int_df = pd.DataFrame(integrated_rows)
    save_bar_plot(
        plots_dir / "integrated_time_in_band.png",
        [f"{r['scenario']}-{r['policy']}" for _, r in int_df.groupby(["scenario", "policy"]).mean(numeric_only=True).reset_index().iterrows()],
        int_df.groupby(["scenario", "policy"])["time_in_band"].mean().tolist(),
        "Integrated time in coherence band",
        "fraction",
    )
    with_df = int_df[int_df["policy"] == "hdr_main"]
    without_df = int_df[int_df["policy"] == "hdr_no_coherence"]
    time_gain = float(with_df["time_in_band"].mean() - without_df["time_in_band"].mean())
    cost_delta = float(with_df["cum_cost"].mean() - without_df["cum_cost"].mean())
    if time_gain > 0.10 and cost_delta <= 0.0:
        help_label = "helps"
    elif time_gain >= -0.02 and cost_delta <= 0.05 * max(without_df["cum_cost"].mean(), 1e-8):
        help_label = "neutral"
    else:
        help_label = "harms"

    # v5.0: w3 calibration sweep — find Pareto-efficient weight
    # Run a sweep over w3 values measuring time_in_band vs cost_delta
    w3_sweep_values = list(config.get("w3_sweep_values", [0.05, 0.10, 0.20, 0.30, 0.50]))
    w3_sweep_rows = []
    n_epi_sweep = max(int(config.get("episodes_per_experiment", 2)), 2)
    for w3_val in w3_sweep_values:
        config_sweep = dict(config)
        config_sweep["w3"] = w3_val
        sweep_rows = []
        for scenario_name in ["coherence_under", "coherence_over"]:
            for epi in range(n_epi_sweep):
                seed = int(config["seeds"][epi % len(config["seeds"])] + 9000 + int(w3_val * 1000) + epi)
                rng_w3 = np.random.default_rng(seed)
                env = SyntheticEnv(eval_model, config_sweep, rng_w3, scenarios[scenario_name], episode_idx=epi)
                out = run_closed_loop_episode(
                    eval_model, config_sweep, env, "hdr_main",
                    allow_mode_b=False, with_tau=True, with_coherence=True,
                )
                kappa_series = out.per_step["kappa_hat"]
                met = settling_metrics(kappa_series, lo, hi)
                sweep_rows.append({
                    "w3": w3_val,
                    "scenario": scenario_name,
                    "cum_cost": float(out.episode_summary["cum_cost"]),
                    "time_in_band": float(met["time_in_band"]),
                })
        if sweep_rows:
            sweep_df_local = pd.DataFrame(sweep_rows)
            time_gain_w3 = float(sweep_df_local["time_in_band"].mean() - without_df["time_in_band"].mean()) if not without_df.empty else float("nan")
            cost_delta_w3 = float(sweep_df_local["cum_cost"].mean() - without_df["cum_cost"].mean()) if not without_df.empty else float("nan")
            pareto_efficient = bool(time_gain_w3 >= 0.10 and cost_delta_w3 <= 0.0)
            w3_sweep_rows.append({
                "w3": w3_val,
                "time_gain": time_gain_w3,
                "cost_delta": cost_delta_w3,
                "pareto_efficient": pareto_efficient,
            })

    w3_df = pd.DataFrame(w3_sweep_rows)
    pareto_w3_values = [r["w3"] for r in w3_sweep_rows if r.get("pareto_efficient", False)]
    best_w3 = float(pareto_w3_values[0]) if pareto_w3_values else float("nan")

    if not w3_df.empty:
        save_line_plot(
            plots_dir / "w3_calibration_sweep.png",
            w3_df["w3"].to_numpy(),
            {
                "time_gain": w3_df["time_gain"].to_numpy(),
                "cost_delta_scaled": (w3_df["cost_delta"] / max(abs(w3_df["cost_delta"].max()), 1e-6)).to_numpy(),
            },
            title="w3 calibration sweep: time-in-band gain vs cost delta",
            xlabel="w3 (coherence weight)",
            ylabel="metric",
        )

    summary = {
        "standalone_in_band_zero_penalty": float(coherence_penalty(0.65, lo, hi) == 0.0),
        "standalone_monotone_outside": float(coherence_penalty(0.90, lo, hi) > coherence_penalty(0.80, lo, hi)),
        "integrated_time_in_band_gain": time_gain,
        "integrated_cost_delta": cost_delta,
        "coherence_help_label": help_label,
        # v5.0 w3 sweep
        "w3_sweep_n_configs": len(w3_sweep_rows),
        "w3_pareto_efficient_count": len(pareto_w3_values),
        "w3_best_pareto": best_w3,
        "w3_nominal_pareto_efficient": bool(float(config.get("w3", 0.3)) in pareto_w3_values),
    }
    save_experiment_bundle(
        stage_root / "coherence_validation",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=standalone_rows + integrated_rows + w3_sweep_rows,
        selected_traces=standalone_traces,
        log_text="Stage 06 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
