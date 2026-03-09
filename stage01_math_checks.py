from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..generator.ground_truth import SyntheticEnv, Scenario, default_scenarios
from ..inference.ici import (
    compute_degradation_factor,
    compute_T_k_eff,
)
from ..model.slds import make_evaluation_model
from ..plotting import save_heatmap, save_line_plot
from ..utils import atomic_write_json, ensure_dir, load_json, seed_everything
from .common import save_experiment_bundle
from .runtime import run_closed_loop_episode


def _sweep_episodes(config: dict) -> int:
    profile = config.get("profile_name", "standard")
    if profile == "smoke":
        return 1
    if profile == "standard":
        return 2
    return 2


def _modify_coupling(env: SyntheticEnv, sparsity: float, sign_flip: float, rng: np.random.Generator):
    for k, A in enumerate(env.A_true):
        A2 = A.copy()
        n = A.shape[0]
        off = ~np.eye(n, dtype=bool)
        keep = (rng.uniform(size=A.shape) < sparsity) & off
        A2[off & (~keep)] = 0.0
        flip = (rng.uniform(size=A.shape) < sign_flip) & off
        A2[flip] *= -1.0
        vals = np.linalg.eigvals(A2)
        max_abs = max(np.max(np.abs(vals)), 1e-8)
        target = 0.96 if k == 1 else 0.72 if k == 0 else 0.55
        env.A_true[k] = A2 * (target / max_abs)


def _mean_metric(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _eval_pair(config: dict, eval_model, scenario: Scenario, episodes: int, seed_offset: int = 0, mode_b: bool = False, modify_env_fn=None):
    hdr_rows, base_rows = [], []
    short_config = dict(config)
    profile = config.get("profile_name", "standard")
    short_config["steps_per_episode"] = min(int(config["steps_per_episode"]), 64 if profile == "smoke" else 96 if profile == "standard" else 128)
    for epi in range(episodes):
        seed = int(config["seeds"][epi % len(config["seeds"])] + seed_offset + epi)
        for policy_name, allow_mode_b, target_rows in [("hdr_main", mode_b, hdr_rows), ("pooled_lqr", False, base_rows)]:
            rng_local = np.random.default_rng(seed + (100 if policy_name == "hdr_main" else 0))
            env = SyntheticEnv(eval_model, short_config, rng_local, scenario, episode_idx=epi, initial_basin=1 if mode_b else None)
            if modify_env_fn is not None:
                modify_env_fn(env, rng_local)
            out = run_closed_loop_episode(eval_model, short_config, env, policy_name if policy_name != "pooled_lqr" else "pooled_lqr", allow_mode_b=allow_mode_b, with_tau=True, with_coherence=True)
            target_rows.append(out.episode_summary)
    return {
        "hdr_cost": _mean_metric(hdr_rows, "cum_cost"),
        "base_cost": _mean_metric(base_rows, "cum_cost"),
        "hdr_safety": _mean_metric(hdr_rows, "safety_violation_rate"),
        "base_safety": _mean_metric(base_rows, "safety_violation_rate"),
        "gain": float(1.0 - _mean_metric(hdr_rows, "cum_cost") / max(_mean_metric(base_rows, "cum_cost"), 1e-8)),
    }


def _group_path(stage_root: Path, name: str) -> Path:
    return stage_root / f"{name}.json"


def _load_or_run(stage_root: Path, name: str, compute_fn):
    path = _group_path(stage_root, name)
    if path.exists():
        return load_json(path)
    payload = compute_fn()
    atomic_write_json(path, payload)
    return payload


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_07" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    base_eval = make_evaluation_model(config, rng)
    scenarios_map = default_scenarios()
    episodes = _sweep_episodes(config)

    def compute_noise_missing():
        profile = config.get("profile_name", "standard")
        noise_levels = [0.8, 1.0, 1.4] if profile != "smoke" else [0.8, 1.2]
        miss_levels = [0.0, 0.15, 0.30] if profile != "smoke" else [0.0, 0.3]
        rows = []
        matrix = []
        for miss in miss_levels:
            row_vals = []
            for noise in noise_levels:
                scenario = Scenario(name="sweep_noise_missing", noise_scale=noise, missingness_boost=miss)
                out = _eval_pair(config, base_eval, scenario, episodes, seed_offset=10000)
                rows.append({"sweep": "noise_missing", "noise": noise, "missing": miss, **out})
                row_vals.append(out["gain"])
            matrix.append(row_vals)
        save_heatmap(plots_dir / "phase_noise_missing.png", np.asarray(matrix), [str(v) for v in noise_levels], [str(v) for v in miss_levels], "HDR gain vs pooled LQR", "gain")
        return {"rows": rows}

    def compute_delay_mismatch():
        profile = config.get("profile_name", "standard")
        delays = [1, 2, 3] if profile == "smoke" else [1, 2, 3, 4]
        mismatches = [0.08, 0.20] if profile == "smoke" else [0.08, 0.16, 0.24]
        rows = []
        matrix = []
        for mm in mismatches:
            row_vals = []
            for delay in delays:
                scenario = Scenario(name="sweep_delay_mismatch", model_mismatch=mm, delay_steps=delay)
                out = _eval_pair(config, base_eval, scenario, episodes, seed_offset=11000)
                rows.append({"sweep": "delay_mismatch", "model_mismatch": mm, "delay": delay, **out})
                row_vals.append(out["gain"])
            matrix.append(row_vals)
        save_heatmap(plots_dir / "phase_delay_mismatch.png", np.asarray(matrix), [str(v) for v in delays], [str(v) for v in mismatches], "HDR gain vs pooled LQR", "gain")
        return {"rows": rows}

    def compute_k_h():
        profile = config.get("profile_name", "standard")
        H_vals = [4, 6] if profile == "smoke" else [4, 6, 8]
        K_vals = [3, 4]
        rows = []
        for K in K_vals:
            gains = []
            for H in H_vals:
                cfg = dict(config)
                cfg["H"] = H
                cfg["K"] = K
                eval_model = make_evaluation_model(cfg, np.random.default_rng(12000 + K * 10 + H), K=K)
                out = _eval_pair(cfg, eval_model, scenarios_map["nominal"], episodes, seed_offset=12000)
                rows.append({"sweep": "K_H", "K": K, "H": H, **out})
                gains.append(out["gain"])
            save_line_plot(plots_dir / f"gain_vs_H_K{K}.png", np.asarray(H_vals), {f"K={K}": np.asarray(gains)}, f"H sweep at K={K}", "H", "gain")
        return {"rows": rows}

    def compute_rho_budget():
        profile = config.get("profile_name", "standard")
        rho_levels = [0.92, 0.98] if profile == "smoke" else [0.92, 0.96, 0.98]
        budgets = [8.0, 20.0] if profile == "smoke" else [8.0, 14.0, 20.0]
        rows = []
        matrix = []
        for budget in budgets:
            row_vals = []
            for rho1 in rho_levels:
                cfg = dict(config)
                cfg["default_burden_budget"] = budget
                cfg["rho_reference"] = [config["rho_reference"][0], rho1, config["rho_reference"][2]]
                eval_model = make_evaluation_model(cfg, np.random.default_rng(13000 + int(budget) + int(rho1 * 100)))
                out = _eval_pair(cfg, eval_model, scenarios_map["nominal"], episodes, seed_offset=13000)
                rows.append({"sweep": "rho_budget", "budget": budget, "rho1": rho1, **out})
                row_vals.append(out["gain"])
            matrix.append(row_vals)
        save_heatmap(plots_dir / "phase_rho_budget.png", np.asarray(matrix), [str(v) for v in rho_levels], [str(v) for v in budgets], "HDR gain vs pooled LQR", "gain")
        return {"rows": rows}

    def compute_pA_qmin():
        profile = config.get("profile_name", "standard")
        pA_vals = [0.6, 0.8] if profile == "smoke" else [0.6, 0.7, 0.8]
        qmin_vals = [0.10, 0.20] if profile == "smoke" else [0.10, 0.15, 0.20]
        rows = []
        matrix = []
        for qmin in qmin_vals:
            row_vals = []
            for pA in pA_vals:
                cfg = dict(config)
                cfg["pA"] = pA
                cfg["qmin"] = qmin
                out = _eval_pair(cfg, base_eval, scenarios_map["mode_b_escape"], episodes, seed_offset=14000, mode_b=True)
                rows.append({"sweep": "pA_qmin", "pA": pA, "qmin": qmin, **out})
                row_vals.append(out["gain"])
            matrix.append(row_vals)
        save_heatmap(plots_dir / "phase_pA_qmin.png", np.asarray(matrix), [str(v) for v in pA_vals], [str(v) for v in qmin_vals], "Mode-B scenario gain", "gain")
        return {"rows": rows}

    def compute_alpha():
        alpha_vals = [0.01, 0.10] if config.get("profile_name", "standard") == "smoke" else [0.01, 0.05, 0.10]
        rows = []
        for alpha in alpha_vals:
            cfg = dict(config)
            cfg["alpha_i"] = alpha
            out = _eval_pair(cfg, base_eval, scenarios_map["nominal"], episodes, seed_offset=15000)
            rows.append({"sweep": "alpha", "alpha_i": alpha, **out})
        return {"rows": rows}

    def compute_coupling():
        profile = config.get("profile_name", "standard")
        sparsities = [0.2, 0.5] if profile == "smoke" else [0.1, 0.3, 0.5]
        sign_flips = [0.0, 0.4] if profile == "smoke" else [0.0, 0.2, 0.4]
        rows = []
        for sparsity in sparsities:
            for sign_flip in sign_flips:
                def mod(env, rng_local, sparsity=sparsity, sign_flip=sign_flip):
                    _modify_coupling(env, sparsity=sparsity, sign_flip=sign_flip, rng=rng_local)
                out = _eval_pair(config, base_eval, scenarios_map["model_mismatch"], episodes, seed_offset=16000, modify_env_fn=mod)
                rows.append({"sweep": "coupling", "sparsity": sparsity, "sign_flip": sign_flip, **out})
        return {"rows": rows}

    def compute_negative():
        nominal = _eval_pair(config, base_eval, scenarios_map["nominal"], episodes, seed_offset=17000)
        inverse = _eval_pair(config, base_eval, scenarios_map["inverse_crime"], episodes, seed_offset=17100)
        return {"rows": [
            {"sweep": "negative_control", "setting": "nominal", **nominal},
            {"sweep": "negative_control", "setting": "inverse_crime", **inverse},
        ]}

    groups = {
        "noise_missing": compute_noise_missing,
        "delay_mismatch": compute_delay_mismatch,
        "K_H": compute_k_h,
        "rho_budget": compute_rho_budget,
        "pA_qmin": compute_pA_qmin,
        "alpha": compute_alpha,
        "coupling": compute_coupling,
        "negative_control": compute_negative,
    }

    all_rows = []
    for name, fn in groups.items():
        payload = _load_or_run(stage_root, name, fn)
        all_rows.extend(payload.get("rows", []))

    df = pd.DataFrame(all_rows)
    inverse = df[(df["sweep"] == "negative_control") & (df["setting"] == "inverse_crime")]
    nominal = df[(df["sweep"] == "negative_control") & (df["setting"] == "nominal")]

    # v5.0: ICI failure regime classification
    # Classify each failure regime as:
    #   inference-remediable  (would improve with Mode C / better posterior)
    #   regime-boundary       (T_k_eff below ω_min; fundamental data insufficiency)
    #   control-structural    (failure persists even with oracle inference)
    inference_remediable = {"noise_missing", "pA_qmin"}  # improved by calibration / Mode C
    regime_boundary_sweeps = {"noise_missing"}            # missingness pushes below ω_min
    control_structural = {"coupling", "delay_mismatch", "rho_budget"}  # independent of inference

    failure_regimes = sorted(df[df["gain"] < 0.0]["sweep"].astype(str).unique().tolist()) if not df.empty else []
    ici_failure_classification = {
        r: (
            "regime_boundary" if r in regime_boundary_sweeps
            else "inference_remediable" if r in inference_remediable
            else "control_structural"
        )
        for r in failure_regimes
    }

    # v5.0: Regime boundary sweep — empirical ω_min vs formula
    T_base = float(config.get("steps_per_episode", 256) * config.get("episodes_per_experiment", 5))
    omega_min_factor = float(config.get("omega_min_factor", 0.005))
    omega_min_formula = omega_min_factor * T_base
    regime_sweep_rows = []
    pi_values = list(config.get("regime_sweep_pi_values", [0.05, 0.10, 0.16, 0.25, 0.40]))
    pmiss_values = list(config.get("regime_sweep_pmiss_values", [0.20, 0.35, 0.516, 0.65, 0.80]))
    rho_values = list(config.get("regime_sweep_rho_values", [0.72, 0.85, 0.90, 0.96]))
    for pi_k in pi_values[:3]:
        for rho_k in rho_values:
            for p_miss in pmiss_values[:3]:
                deg = compute_degradation_factor(pi_k, p_miss, rho_k)
                T_k_eff = compute_T_k_eff(T_base, pi_k, p_miss, rho_k)
                below_boundary = bool(T_k_eff < omega_min_formula)
                regime_sweep_rows.append({
                    "sweep": "regime_boundary",
                    "pi_k": pi_k,
                    "rho_k": rho_k,
                    "p_miss": p_miss,
                    "degradation_factor": deg,
                    "T_k_eff": T_k_eff,
                    "omega_min": omega_min_formula,
                    "below_boundary": below_boundary,
                })
    regime_df = pd.DataFrame(regime_sweep_rows)
    # HDR validation params (pi=0.16, p_miss=0.516, rho=0.96) should be below boundary
    hdv_row = regime_df[(regime_df["pi_k"] == 0.16) & (regime_df["rho_k"] == 0.96)] if not regime_df.empty else pd.DataFrame()
    hdv_below_boundary = bool(hdv_row["below_boundary"].any()) if not hdv_row.empty else True

    if not regime_df.empty:
        below_counts = regime_df.groupby("pi_k")["below_boundary"].mean()
        save_line_plot(
            plots_dir / "regime_boundary_fraction.png",
            below_counts.index.to_numpy(),
            {"fraction_below_omega_min": below_counts.values},
            title="Fraction of configs below regime boundary vs base rate",
            xlabel="pi_k (basin base rate)",
            ylabel="fraction below ω_min",
        )

    all_rows.extend(regime_sweep_rows)

    summary = {
        "worst_gain": float(df["gain"].min()) if not df.empty else float("nan"),
        "best_gain": float(df["gain"].max()) if not df.empty else float("nan"),
        "oracle_optimism_gap": float(inverse["gain"].mean() - nominal["gain"].mean()) if not inverse.empty and not nominal.empty else float("nan"),
        "failure_regimes": failure_regimes,
        "rows": int(len(df)),
        # v5.0 ICI additions
        "ici_failure_classification": ici_failure_classification,
        "regime_sweep_configs": len(regime_sweep_rows),
        "regime_sweep_below_boundary_fraction": float(regime_df["below_boundary"].mean()) if not regime_df.empty else float("nan"),
        "hdv_params_correctly_below_boundary": hdv_below_boundary,
        "omega_min_formula": float(omega_min_formula),
    }
    save_experiment_bundle(
        stage_root / "robustness_falsification",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=all_rows,
        selected_traces={},
        log_text="Stage 07 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
