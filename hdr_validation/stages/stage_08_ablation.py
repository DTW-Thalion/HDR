"""
Stage 08 — Ablation Study (HDR v5.2)
======================================

Isolates the contribution of each HDR Mode A component to the +3.7% gain
reported in Benchmark A. Five policy variants are evaluated:

  hdr_full           : full HDR (w2=0.5, w3=0.3, calibrated threshold)
  mpc_only           : pure MPC (w2=0, w3=0, calibrated)
  mpc_plus_surrogate : MPC + tau-tilde surrogate (w2=0.5, w3=0, calibrated)
  mpc_plus_coherence : MPC + coherence (w2=0, w3=0.3, calibrated)
  hdr_no_calib       : full HDR without calibration adjustment

Results saved to results/stage_08/ablation_results.json.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


@dataclasses.dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    w2: float
    w3: float
    use_calibration: bool  # if False, always use p_A; ignore R_Brier adjustment


ABLATION_VARIANTS: list[AblationConfig] = [
    AblationConfig("hdr_full",           w2=0.5, w3=0.3, use_calibration=True),
    AblationConfig("mpc_only",           w2=0.0, w3=0.0, use_calibration=True),
    AblationConfig("mpc_plus_surrogate", w2=0.5, w3=0.0, use_calibration=True),
    AblationConfig("mpc_plus_coherence", w2=0.0, w3=0.3, use_calibration=True),
    AblationConfig("hdr_no_calib",       w2=0.5, w3=0.3, use_calibration=False),
]


def _make_benchmark_config(n_seeds: int = 20, n_ep: int = 30, T: int = 256) -> dict[str, Any]:
    """Create the Benchmark A configuration."""
    return {
        "state_dim": 8,
        "obs_dim": 16,
        "control_dim": 8,
        "disturbance_dim": 8,
        "K": 3,
        "H": 6,
        "w1": 1.0,
        "w2": 0.5,
        "w3": 0.3,
        "lambda_u": 0.1,
        "alpha_i": 0.05,
        "eps_safe": 0.01,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 256,
        "model_mismatch_bound": 0.20,
        "kappa_lo": 0.55,
        "kappa_hi": 0.75,
        "pA": 0.70,
        "qmin": 0.15,
        "steps_per_day": 48,
        "dt_minutes": 30,
        "coherence_window": 24,
        "default_burden_budget": 56.0,  # 28 * T/128 = 56 for T=256
        "circadian_locked_controls": [5, 6],
        "R_brier_max": 0.05,
        "omega_min_factor": 0.005,
        "T_C_max": 50,
        "k_calib": 1.0,
        "sigma_dither": 0.08,
        "epsilon_control": 0.50,
        "missing_fraction_target": 0.516,
        "mode1_base_rate": 0.16,
        # Benchmark A parameters
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "steps_per_episode": T,
        "profile_name": "highpower",
    }


def _run_episode(
    cfg: dict[str, Any],
    basin_idx: int,
    rng: np.random.Generator,
    ablation_cfg: AblationConfig,
) -> dict[str, float]:
    """Run one episode and return cost metrics for HDR vs pooled_lqr_estimated baseline.

    Returns dict with 'hdr_cost', 'baseline_cost', 'gain'.
    """
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.control.lqr import dlqr

    # Use a unique rng per episode to avoid reseeding model
    model_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    eval_model = make_evaluation_model(cfg, model_rng)
    basin = eval_model.basins[basin_idx]
    target = build_target_set(basin_idx, cfg)
    T = cfg["steps_per_episode"]
    n = cfg["state_dim"]
    lambda_u = float(cfg.get("lambda_u", 0.1))

    # Build variant config
    variant_cfg = dict(cfg)
    variant_cfg["w2"] = ablation_cfg.w2
    variant_cfg["w3"] = ablation_cfg.w3
    if not ablation_cfg.use_calibration:
        # Disable calibration: set R_brier_max to infinity so condition never triggers
        variant_cfg["R_brier_max"] = 1.0  # effectively disable
        variant_cfg["k_calib"] = 0.0

    # Pooled LQR baseline (estimated model, no mode switching)
    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * float(cfg.get("lambda_u", 0.1))
    try:
        K_pool, _ = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
    except Exception:
        K_pool = np.zeros((n, n))

    # Initial state
    x = rng.normal(scale=0.5, size=n)
    x_ref = np.zeros(n)
    P_hat = np.eye(n) * 0.2

    hdr_cost = 0.0
    baseline_cost = 0.0

    for t in range(T):
        # State cost (shared)
        state_cost = float(np.dot(x, x))

        # HDR Mode A control
        try:
            res = solve_mode_a(x, P_hat, basin, target, kappa_hat=0.65, config=variant_cfg, step=t)
            u_hdr = res.u
        except Exception:
            u_hdr = np.zeros(cfg["control_dim"])

        # Baseline control (pooled LQR)
        u_base = -K_pool @ (x - x_ref)
        u_base = np.clip(u_base, -0.6, 0.6)

        hdr_cost += state_cost + lambda_u * float(np.dot(u_hdr, u_hdr))
        baseline_cost += state_cost + lambda_u * float(np.dot(u_base, u_base))

        # Advance state (same dynamics, different controls)
        w = rng.multivariate_normal(np.zeros(n), basin.Q)
        x = basin.A @ x + basin.B @ u_hdr + basin.b + w

    gain = (baseline_cost - hdr_cost) / max(baseline_cost, 1e-12)
    return {
        "hdr_cost": hdr_cost,
        "baseline_cost": baseline_cost,
        "gain": gain,
        "basin_idx": basin_idx,
    }


def _bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean."""
    rng = np.random.default_rng(rng_seed)
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return float("nan"), float("nan")
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return lo, hi


def run_stage_08(
    n_seeds: int = 20,
    n_ep: int = 30,
    T: int = 256,
    output_dir: Path | None = None,
    fast_mode: bool = False,
) -> dict:
    """Run the Stage 08 ablation study.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds.
    n_ep : int
        Episodes per seed.
    T : int
        Steps per episode.
    output_dir : Path or None
        Directory for output files. Defaults to results/stage_08/.
    fast_mode : bool
        If True, use reduced n_seeds and n_ep for fast smoke testing.

    Returns
    -------
    dict
        Ablation results with variants, metrics, and JSON-compatible schema.
    """
    if fast_mode:
        n_seeds = min(n_seeds, 2)
        n_ep = min(n_ep, 3)
        T = min(T, 64)

    cfg = _make_benchmark_config(n_seeds=n_seeds, n_ep=n_ep, T=T)

    if output_dir is None:
        output_dir = ROOT / "results" / "stage_08"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [101 + i * 101 for i in range(n_seeds)]

    # Collect per-episode gains for each variant on maladaptive (basin=1) episodes
    variant_gains: dict[str, list[float]] = {v.name: [] for v in ABLATION_VARIANTS}

    total_maladaptive = 0
    for seed_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ep_idx in range(n_ep):
            # In Benchmark A, ~179/600 episodes are maladaptive basin
            # Use probability-based selection to match paper ratio
            is_mal = (rng.random() < 0.30)  # ~30% maladaptive
            basin_idx = 1 if is_mal else rng.choice([0, 2])

            if basin_idx != 1:
                continue  # Only collect maladaptive episodes for primary metric

            total_maladaptive += 1
            ep_rng_base = int(seed * 10000 + ep_idx)

            for abl_cfg in ABLATION_VARIANTS:
                ep_rng = np.random.default_rng(ep_rng_base)
                result = _run_episode(cfg, basin_idx=1, rng=ep_rng, ablation_cfg=abl_cfg)
                variant_gains[abl_cfg.name].append(result["gain"])

    # Compute summary statistics for each variant
    variants_out: dict[str, dict] = {}
    for abl_cfg in ABLATION_VARIANTS:
        gains = np.array(variant_gains[abl_cfg.name])
        if len(gains) == 0:
            gains = np.array([0.0])
        mean_gain = float(np.mean(gains))
        ci_lo, ci_hi = _bootstrap_ci(gains)
        win_rate = float(np.mean(gains > 0))
        n_mal = len(gains)
        variants_out[abl_cfg.name] = {
            "mean_gain": round(mean_gain, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "win_rate": round(win_rate, 4),
            "N_mal": n_mal,
        }

    result_json = {
        "variants": variants_out,
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "T": T,
        "total_maladaptive_episodes": total_maladaptive,
    }

    # Save JSON
    out_path = output_dir / "ablation_results.json"
    out_path.write_text(json.dumps(result_json, indent=2))

    # Print ASCII table
    print("\n" + "┌" + "─" * 22 + "┬" + "─" * 10 + "┬" + "─" * 22 + "┬" + "─" * 10 + "┐")
    print("│ {:20s} │ {:8s} │ {:20s} │ {:8s} │".format("Variant", "Gain", "95% CI", "Win Rate"))
    print("├" + "─" * 22 + "┼" + "─" * 10 + "┼" + "─" * 22 + "┼" + "─" * 10 + "┤")
    for name, v in variants_out.items():
        gain_str = f"{v['mean_gain']:+.1%}"
        ci_str = f"[{v['ci_lo']:+.1%}, {v['ci_hi']:+.1%}]"
        win_str = f"{v['win_rate']:.1%}"
        print("│ {:20s} │ {:8s} │ {:20s} │ {:8s} │".format(name, gain_str, ci_str, win_str))
    print("└" + "─" * 22 + "┴" + "─" * 10 + "┴" + "─" * 22 + "┴" + "─" * 10 + "┘")

    # PASS/FAIL criterion: hdr_full gain >= mpc_only gain
    hdr_full_gain = variants_out["hdr_full"]["mean_gain"]
    mpc_only_gain = variants_out["mpc_only"]["mean_gain"]
    ablation_criterion = hdr_full_gain >= mpc_only_gain
    status = "PASS" if ablation_criterion else "FAIL"
    print(f"\n  [{status}] hdr_full gain ({hdr_full_gain:+.4f}) >= mpc_only gain ({mpc_only_gain:+.4f})")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_08()
