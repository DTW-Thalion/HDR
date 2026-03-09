from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..generator.ground_truth import SyntheticEnv, default_scenarios
from ..inference.em_updates import fit_linear_dynamics, unpack_dynamics_theta
from ..inference.ici import (
    brier_reliability,
    compute_degradation_factor,
    compute_T_k_eff,
)
from ..metrics import classification_metrics, reliability_bins, rmse
from ..model.hsmm import fit_hazard_from_sequences
from ..model.slds import make_evaluation_model
from ..plotting import save_bar_plot, save_calibration_plot, save_line_plot
from ..utils import atomic_write_json, ensure_dir, seed_everything
from .common import save_experiment_bundle, summarize_metric_rows
from .runtime import run_imm_on_sequence


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _list_npz(dir_path: Path) -> list[Path]:
    return sorted(dir_path.glob("*.npz"))


def _collect_regression_arrays(episodes: list[dict], basin: int, max_len: int | None = None):
    Xs, Us, Xn = [], [], []
    count = 0
    for epi in episodes:
        x = epi["x_true"]
        u = epi["u"]
        z = epi["z_true"]
        for t in range(len(z) - 1):
            if int(z[t]) == basin:
                Xs.append(x[t])
                Us.append(u[t])
                Xn.append(x[t + 1])
                count += 1
                if max_len is not None and count >= max_len:
                    return np.asarray(Xs), np.asarray(Us), np.asarray(Xn)
    return np.asarray(Xs), np.asarray(Us), np.asarray(Xn)


def _relative_error(A_est, B_est, b_est, basin):
    num = np.linalg.norm(A_est - basin.A) + np.linalg.norm(B_est - basin.B) + np.linalg.norm(b_est - basin.b)
    den = np.linalg.norm(basin.A) + np.linalg.norm(basin.B) + np.linalg.norm(basin.b) + 1e-8
    return float(num / den)


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_03" / profile_name)
    plots_dir = ensure_dir(stage_root / "plots")
    rng = seed_everything(int(config["seeds"][0]))
    eval_model = make_evaluation_model(config, rng)
    stage02_root = project_root / "results" / "stage_02" / profile_name
    train_eps = [_load_npz(p) for p in _list_npz(stage02_root / "train")]
    val_eps = [_load_npz(p) for p in _list_npz(stage02_root / "validation")]
    test_eps = [_load_npz(p) for p in _list_npz(stage02_root / "test")]
    observer_rows = []
    axis_rmse_accum = []
    cal_true = []
    cal_prob = []
    for split_name, episodes in [("validation", val_eps), ("test", test_eps)]:
        for epi_idx, epi in enumerate(episodes):
            out = run_imm_on_sequence(epi, eval_model, config)
            axis_rmse = rmse(epi["x_true"], out["x_hat"], axis=0)
            axis_rmse_accum.append(axis_rmse)
            y_true_bin = (epi["z_true"] == 1).astype(int)
            y_pred_bin = (out["map_mode"] == 1).astype(int)
            cls = classification_metrics(y_true_bin, y_pred_bin, y_prob=out["mode_probs"], positive_label=1)
            observer_rows.append({
                "split": split_name,
                "episode": epi_idx,
                "state_rmse": float(np.mean(axis_rmse)),
                "mode_accuracy": float(np.mean(epi["z_true"] == out["map_mode"])),
                **cls,
            })
            cal_true.extend(y_true_bin.tolist())
            cal_prob.extend(out["mode_probs"][:, 1].tolist())
    axis_rmse_mean = np.mean(np.asarray(axis_rmse_accum), axis=0)
    save_bar_plot(plots_dir / "axis_rmse.png", [f"axis_{i}" for i in range(len(axis_rmse_mean))], axis_rmse_mean.tolist(), "State RMSE by axis", "RMSE")
    cal_df = reliability_bins(np.asarray(cal_true), np.asarray(cal_prob), bins=10)
    save_calibration_plot(plots_dir / "mode1_calibration.png", cal_df, title="Maladaptive posterior calibration")
    # hazard estimation
    hazard_hat = fit_hazard_from_sequences([epi["z_true"] for epi in train_eps], K=len(eval_model.basins), max_len=64)
    hazard_rows = []
    for k, dm in enumerate(eval_model.dwell_models):
        hz_true = dm.hazard()[:64]
        hz_hat = hazard_hat[k][:64]
        mse = float(np.mean((hz_true - hz_hat) ** 2))
        hazard_rows.append({"basin": k, "hazard_mse": mse})
    # parameter recovery vs length and priors
    lengths = sorted(set([max(32, config["steps_per_episode"] // 4), max(64, config["steps_per_episode"] // 2), config["steps_per_episode"]]))
    param_rows = []
    for length in lengths:
        for basin_idx, basin in enumerate(eval_model.basins[: min(3, len(eval_model.basins))]):
            X, U, Xn = _collect_regression_arrays(train_eps, basin_idx, max_len=length)
            if len(X) < 8:
                continue
            theta_np = fit_linear_dynamics(X, U, Xn, ridge=1e-3)
            A_np, B_np, b_np = unpack_dynamics_theta(theta_np, eval_model.state_dim, eval_model.control_dim)
            err_np = _relative_error(A_np, B_np, b_np, basin)
            prior_theta = np.vstack([basin.A.T, basin.B.T, basin.b.reshape(1, -1)])
            theta_pr = fit_linear_dynamics(X, U, Xn, ridge=1e-3, prior_theta=prior_theta, prior_strength=0.5)
            A_pr, B_pr, b_pr = unpack_dynamics_theta(theta_pr, eval_model.state_dim, eval_model.control_dim)
            err_pr = _relative_error(A_pr, B_pr, b_pr, basin)
            param_rows.append({"length": length, "basin": basin_idx, "setting": "no_prior", "param_error": err_np})
            param_rows.append({"length": length, "basin": basin_idx, "setting": "with_prior", "param_error": err_pr})
    # designed perturbation / dither comparison with small fresh episodes
    scenarios = default_scenarios()
    extra_rows = []
    for label, control_scale in [("passive", 0.0), ("dither", 0.08), ("designed", 0.18)]:
        episodes = []
        for epi_idx, seed in enumerate(config["seeds"]):
            rng_local = np.random.default_rng(int(seed) + 777 + epi_idx)
            env = SyntheticEnv(eval_model, config, rng_local, scenarios["nominal"], episode_idx=epi_idx)
            obs = env.reset()
            data = {"x_true": [], "z_true": [], "u": []}
            for _ in range(env.T):
                u = np.clip(rng_local.normal(scale=control_scale, size=eval_model.control_dim), -0.35, 0.35)
                data["x_true"].append(obs["x_true"])
                data["z_true"].append(obs["z_true"])
                data["u"].append(u)
                obs, info = env.step(u)
                if info["done"]:
                    break
            episodes.append({k: np.asarray(v) for k, v in data.items()})
        for basin_idx, basin in enumerate(eval_model.basins[: min(3, len(eval_model.basins))]):
            X, U, Xn = _collect_regression_arrays(episodes, basin_idx, max_len=config["steps_per_episode"])
            if len(X) < 8:
                continue
            theta = fit_linear_dynamics(X, U, Xn, ridge=1e-3)
            A_est, B_est, b_est = unpack_dynamics_theta(theta, eval_model.state_dim, eval_model.control_dim)
            extra_rows.append({"setting": label, "basin": basin_idx, "param_error": _relative_error(A_est, B_est, b_est, basin)})
    # plots
    param_df = pd.DataFrame(param_rows)
    if not param_df.empty:
        for setting in sorted(param_df["setting"].unique()):
            sub = param_df[param_df["setting"] == setting].groupby("length")["param_error"].mean()
            save_line_plot(
                plots_dir / f"param_recovery_{setting}.png",
                sub.index.to_numpy(),
                {setting: sub.values},
                title=f"Parameter recovery vs data length ({setting})",
                xlabel="effective samples",
                ylabel="relative error",
            )
    observer_df = pd.DataFrame(observer_rows)
    observer_summary = observer_df.mean(numeric_only=True).to_dict()
    priors_gain = float(
        param_df[param_df["setting"] == "no_prior"]["param_error"].mean() -
        param_df[param_df["setting"] == "with_prior"]["param_error"].mean()
    ) if not param_df.empty else float("nan")
    extra_df = pd.DataFrame(extra_rows)
    passive_err = float(extra_df[extra_df["setting"] == "passive"]["param_error"].mean()) if not extra_df.empty else float("nan")
    dither_err = float(extra_df[extra_df["setting"] == "dither"]["param_error"].mean()) if not extra_df.empty else float("nan")
    designed_err = float(extra_df[extra_df["setting"] == "designed"]["param_error"].mean()) if not extra_df.empty else float("nan")

    # v5.0: Brier reliability decomposition
    brier_decomp = brier_reliability(np.asarray(cal_true), np.asarray(cal_prob), n_bins=10)

    # v5.0: Regime characterisation — T_k_eff per basin
    T_total = float(config.get("steps_per_episode", 256) * config.get("episodes_per_experiment", 5))
    K = config.get("K", 3)
    rho_per_basin = list(config.get("rho_reference", [0.72, 0.96, 0.55]))[:K]
    pi_vals = [
        1.0 - float(config.get("mode1_base_rate", 0.16)) - 0.05,
        float(config.get("mode1_base_rate", 0.16)),
        0.05,
    ][:K]
    p_miss_obs = float(np.mean([
        float(np.mean(np.isnan(epi["y"]) if "y" in epi else 0.516))
        for epi in (val_eps + test_eps) if epi
    ])) if (val_eps + test_eps) else float(config.get("missing_fraction_target", 0.516))
    T_k_eff_per_basin = [
        compute_T_k_eff(T_total, pi_vals[k], p_miss_obs, rho_per_basin[k])
        for k in range(K)
    ]
    omega_min_factor = float(config.get("omega_min_factor", 0.005))
    omega_min = omega_min_factor * T_total
    regime_flags = [bool(t < omega_min) for t in T_k_eff_per_basin]
    degradation_factors = [
        compute_degradation_factor(pi_vals[k], p_miss_obs, rho_per_basin[k])
        for k in range(K)
    ]
    # Regime boundary prediction: at 50%+ missingness + 16% base rate + ρ=0.96,
    # the maladaptive basin should be flagged as below ω_min.
    regime_boundary_correctly_predicts_maladaptive = bool(regime_flags[1]) if K > 1 else False

    summary = {
        "observer_state_rmse": float(observer_summary.get("state_rmse", np.nan)),
        "observer_mode_accuracy": float(observer_summary.get("mode_accuracy", np.nan)),
        "observer_mode_f1": float(observer_summary.get("f1", np.nan)),
        "observer_brier": float(observer_summary.get("brier", np.nan)),
        "axes_rmse_below_0_9": int(np.sum(axis_rmse_mean < 0.9)),
        "hazard_mse_mean": float(np.mean([r["hazard_mse"] for r in hazard_rows])) if hazard_rows else float("nan"),
        "priors_gain": priors_gain,
        "dither_gain": float(passive_err - dither_err) if np.isfinite(passive_err) and np.isfinite(dither_err) else float("nan"),
        "perturbation_gain": float(passive_err - designed_err) if np.isfinite(passive_err) and np.isfinite(designed_err) else float("nan"),
        "near_unit_root_error_ratio": float(
            extra_df[extra_df["basin"] == 1]["param_error"].mean() / max(extra_df[extra_df["basin"] == 0]["param_error"].mean(), 1e-8)
        ) if not extra_df.empty else float("nan"),
        # v5.0 ICI additions
        "brier_reliability": float(brier_decomp["reliability"]),
        "brier_resolution": float(brier_decomp["resolution"]),
        "brier_uncertainty": float(brier_decomp["uncertainty"]),
        "brier_identity_check": float(brier_decomp["brier_score"]),
        "T_k_eff_desired": float(T_k_eff_per_basin[0]),
        "T_k_eff_maladaptive": float(T_k_eff_per_basin[1]) if K > 1 else float("nan"),
        "omega_min": float(omega_min),
        "maladaptive_below_omega_min": bool(regime_flags[1]) if K > 1 else False,
        "degradation_factor_maladaptive": float(degradation_factors[1]) if K > 1 else float("nan"),
        "regime_boundary_correctly_predicted": regime_boundary_correctly_predicts_maladaptive,
        "p_miss_observed": float(p_miss_obs),
    }
    metrics_rows = observer_rows + hazard_rows + param_rows + extra_rows
    save_experiment_bundle(
        stage_root / "identification_validation",
        config=config,
        seed=config["seeds"],
        summary=summary,
        metrics_rows=metrics_rows,
        selected_traces={
            "axis_rmse_mean": axis_rmse_mean,
            "calibration_true": np.asarray(cal_true),
            "calibration_prob": np.asarray(cal_prob),
        },
        log_text="Stage 03 completed.",
    )
    atomic_write_json(stage_root / "stage_summary.json", summary)
    return summary
