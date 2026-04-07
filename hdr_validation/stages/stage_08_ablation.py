"""
Stage 08 — Ablation Study
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
    from hdr_validation.defaults import DEFAULTS

    cfg = dict(DEFAULTS)
    cfg.update({
        "max_dwell_len": 256,
        "default_burden_budget": 56.0,  # 28 * T/128 = 56 for T=256
        # Benchmark A parameters
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "steps_per_episode": T,
        "profile_name": "highpower",
    })
    return cfg


def _kappa_schedule(t: int, T: int, cfg: dict) -> float:
    """Time-varying kappa_hat: starts below kappa_lo, ramps to kappa_hi.

    Models a maladaptive episode where the system gradually approaches
    the target set, exercising the coherence penalty in the out-of-band
    region during the first portion of each episode.
    """
    kappa_lo = float(cfg.get("kappa_lo", 0.55))
    kappa_hi = float(cfg.get("kappa_hi", 0.75))
    kappa_start = kappa_lo - 0.15  # below target (maladaptive)
    ramp_end = int(T * 2 / 3)
    if ramp_end <= 0:
        return kappa_hi
    if t >= ramp_end:
        return kappa_hi
    return kappa_start + (kappa_hi - kappa_start) * (t / ramp_end)


def _get_kappa_hat(ablation_cfg: AblationConfig, t: int, T: int, cfg: dict) -> float:
    """Compute the effective kappa_hat for this step and variant.

    The base kappa follows _kappa_schedule (time-varying, starting below
    kappa_lo to model a maladaptive episode). The calibration adjustment
    modulates this: calibrated variants (use_calibration=True) use
    p_A_robust to scale kappa, while uncalibrated variants use raw p_A.
    """
    from hdr_validation.inference.ici import compute_p_A_robust

    kappa_lo = float(cfg.get("kappa_lo", 0.55))
    kappa_hi = float(cfg.get("kappa_hi", 0.75))

    # Base kappa from schedule (time-varying, starts below target)
    kappa_base = _kappa_schedule(t, T, cfg)

    # R_brier proxy: realistic value between 0 and R_brier_max=0.05
    R_brier_episode = 0.03

    p_A_base = float(cfg.get("pA", 0.70))

    if ablation_cfg.use_calibration:
        k_calib = float(cfg.get("k_calib", 1.0))
        p_A_robust = compute_p_A_robust(
            p_A=p_A_base,
            k_calib=k_calib,
            R_brier=R_brier_episode,
        )
        # p_A_robust >= p_A_base (miscalibration raises threshold).
        # Map the overshoot to a kappa REDUCTION: when posterior is
        # miscalibrated, use a tighter (lower) kappa to compensate.
        overshoot = (p_A_robust - p_A_base) / max(1.0 - p_A_base, 1e-6)
        kappa_hat = kappa_base * (1.0 - overshoot * 0.15)
    else:
        # No calibration adjustment: use base schedule directly
        kappa_hat = kappa_base

    return float(np.clip(kappa_hat, kappa_lo - 0.20, kappa_hi))


def _run_episode(
    cfg: dict[str, Any],
    basin_idx: int,
    rng: np.random.Generator,
    ablation_cfg: AblationConfig,
) -> dict[str, Any]:
    """Run one episode and return cost metrics for HDR vs pooled_lqr_estimated baseline.

    Returns dict with 'hdr_cost', 'baseline_cost', 'gain', 'basin_idx', 'diagnostics'.
    """
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.control.mpc import solve_mode_a, precompute_mode_a_cache
    from hdr_validation.control.lqr import dlqr
    from hdr_validation.model.coherence import coherence_grad, coherence_penalty

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
        variant_cfg["R_brier_max"] = 1.0
        variant_cfg["k_calib"] = 0.0

    # Pre-compute expensive invariants for this episode
    mpc_cache = precompute_mode_a_cache(basin, variant_cfg)

    # Pooled LQR baseline (estimated model, no mode switching)
    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * float(cfg.get("lambda_u", 0.1))
    try:
        K_pool, _ = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
    except Exception:
        K_pool = np.zeros((n, n))

    # Initial state — shared starting point for both policies
    x0 = rng.normal(scale=0.5, size=n)
    x_hdr = x0.copy()
    x_base = x0.copy()
    x_ref = np.zeros(n)
    P_hat = np.eye(n) * 0.2

    hdr_cost = 0.0
    baseline_cost = 0.0

    # Diagnostic accumulators
    coupling_scales: list[float] = []
    kappa_hats: list[float] = []
    u_hdr_norms: list[float] = []
    u_base_norms: list[float] = []

    for t in range(T):
        # State costs on independent trajectories
        hdr_state_cost = float(np.dot(x_hdr, x_hdr))
        base_state_cost = float(np.dot(x_base, x_base))

        # Compute time-varying kappa_hat with calibration modulation
        kappa_hat_t = _get_kappa_hat(ablation_cfg, t, T, variant_cfg)

        # HDR Mode A control (on HDR trajectory)
        try:
            res = solve_mode_a(
                x_hdr, P_hat, basin, target,
                kappa_hat=kappa_hat_t,
                config=variant_cfg, step=t,
                P_terminal_precomputed=mpc_cache["P_terminal"],
                C_pinv_precomputed=mpc_cache["C_pinv"],
            )
            u_hdr = res.u
        except Exception:
            u_hdr = np.zeros(cfg["control_dim"])

        # Baseline control on baseline trajectory (pooled LQR)
        u_base = -K_pool @ (x_base - x_ref)
        u_base = np.clip(u_base, -0.6, 0.6)

        hdr_cost += hdr_state_cost + lambda_u * float(np.dot(u_hdr, u_hdr))
        baseline_cost += base_state_cost + lambda_u * float(np.dot(u_base, u_base))

        # Measure coherence coupling_scale for diagnostics
        g_pen_t = coherence_penalty(
            kappa_hat_t, float(variant_cfg["kappa_lo"]), float(variant_cfg["kappa_hi"]),
        )
        g_grad_t = coherence_grad(
            kappa_hat_t, float(variant_cfg["kappa_lo"]), float(variant_cfg["kappa_hi"]),
        )
        cs_t = float(variant_cfg["w3"]) * (abs(g_grad_t) * 0.5 + g_pen_t * 0.3)

        coupling_scales.append(cs_t)
        kappa_hats.append(kappa_hat_t)
        u_hdr_norms.append(float(np.linalg.norm(u_hdr)))
        u_base_norms.append(float(np.linalg.norm(u_base)))

        # Shared process noise for paired comparison (pre-computed Cholesky)
        w = basin.Q_cholesky @ rng.standard_normal(n)

        # Advance independent trajectories
        x_hdr = basin.A @ x_hdr + basin.B @ u_hdr + basin.b + w
        x_base = basin.A @ x_base + basin.B @ u_base + basin.b + w

    gain = (baseline_cost - hdr_cost) / max(baseline_cost, 1e-12)

    # Compute diagnostics summary
    diag: dict[str, Any] = {
        "coherence_steps_active": sum(1 for cs in coupling_scales if cs > 1e-6),
        "coherence_mean_coupling": float(np.mean(coupling_scales)) if coupling_scales else 0.0,
        "kappa_hat_mean": float(np.mean(kappa_hats)) if kappa_hats else 0.0,
        "kappa_hat_min": float(np.min(kappa_hats)) if kappa_hats else 0.0,
        "kappa_hat_max": float(np.max(kappa_hats)) if kappa_hats else 0.0,
        "u_hdr_mean_norm": float(np.mean(u_hdr_norms)) if u_hdr_norms else 0.0,
        "u_base_mean_norm": float(np.mean(u_base_norms)) if u_base_norms else 0.0,
        "T": T,
    }

    return {
        "hdr_cost": hdr_cost,
        "baseline_cost": baseline_cost,
        "gain": gain,
        "basin_idx": basin_idx,
        "diagnostics": diag,
    }


def _bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean (vectorized)."""
    rng = np.random.default_rng(rng_seed)
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        raise ValueError(
            "_bootstrap_ci received empty data. "
            "This indicates zero maladaptive episodes were collected. "
            "Check N_MAL_MIN guard in run_stage_08."
        )
    # Vectorized: draw all bootstrap samples at once
    indices = rng.integers(0, len(data), size=(n_boot, len(data)))
    boot_means = data[indices].mean(axis=1)
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

    # Fast-mode: force at least N_MAL_MIN maladaptive episodes to guarantee
    # non-vacuous output. Production mode retains probabilistic selection to
    # match the Benchmark A episode distribution.
    N_MAL_MIN = 6
    forced_mal_set: set[tuple[int, int]] = set()
    if n_seeds * n_ep < 20:   # fast/smoke mode threshold
        # Assign first N_MAL_MIN episodes round-robin to basin 1
        count = 0
        for s_idx in range(n_seeds):
            for e_idx in range(n_ep):
                if count >= N_MAL_MIN:
                    break
                forced_mal_set.add((s_idx, e_idx))
                count += 1

    # Collect per-episode results for each variant on maladaptive (basin=1) episodes
    variant_gains: dict[str, list[float]] = {v.name: [] for v in ABLATION_VARIANTS}
    episode_results: dict[str, list[dict]] = {v.name: [] for v in ABLATION_VARIANTS}

    total_maladaptive = 0
    for seed_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ep_idx in range(n_ep):
            # In Benchmark A, ~179/600 episodes are maladaptive basin
            # Use probability-based selection to match paper ratio
            is_forced = (seed_idx, ep_idx) in forced_mal_set
            is_mal = is_forced or (rng.random() < 0.30)
            basin_idx = 1 if is_mal else rng.choice([0, 2])

            if basin_idx != 1:
                continue  # Only collect maladaptive episodes for primary metric

            total_maladaptive += 1
            ep_rng_base = int(seed * 10000 + ep_idx)

            for abl_cfg in ABLATION_VARIANTS:
                ep_rng = np.random.default_rng(ep_rng_base)
                result = _run_episode(cfg, basin_idx=1, rng=ep_rng, ablation_cfg=abl_cfg)
                variant_gains[abl_cfg.name].append(result["gain"])
                episode_results[abl_cfg.name].append(result)

    # Compute summary statistics for each variant
    variants_out: dict[str, dict] = {}
    for abl_cfg in ABLATION_VARIANTS:
        gains = np.array(variant_gains[abl_cfg.name])
        if len(gains) == 0:
            raise ValueError(
                f"Zero maladaptive episodes collected for variant '{abl_cfg.name}'. "
                "This should not happen with the forced_mal_set guard."
            )
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

        # Aggregate per-variant diagnostics
        ep_list = episode_results[abl_cfg.name]
        variants_out[abl_cfg.name]["diagnostics_mean"] = {
            "coherence_steps_active_pct": float(np.mean(
                [ep.get("diagnostics", {}).get("coherence_steps_active", 0) / max(T, 1)
                 for ep in ep_list]
            )),
            "coherence_mean_coupling": float(np.mean(
                [ep.get("diagnostics", {}).get("coherence_mean_coupling", 0.0)
                 for ep in ep_list]
            )),
            "kappa_hat_mean": float(np.mean(
                [ep.get("diagnostics", {}).get("kappa_hat_mean", 0.0)
                 for ep in ep_list]
            )),
        }

    result_json: dict[str, Any] = {
        "variants": variants_out,
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "T": T,
        "total_maladaptive_episodes": total_maladaptive,
    }

    MIN_MAL_FOR_VALID_RESULT = 5
    if total_maladaptive < MIN_MAL_FOR_VALID_RESULT:
        import warnings
        warnings.warn(
            f"Stage 08: only {total_maladaptive} maladaptive episodes collected "
            f"(minimum {MIN_MAL_FOR_VALID_RESULT} required for valid ablation statistics). "
            f"Results are NOT suitable for manuscript use. "
            f"Run at full scale: n_seeds=20, n_ep=30, T=256.",
            stacklevel=2,
        )
        result_json["results_are_valid"] = False
        result_json["validity_note"] = (
            f"N_mal={total_maladaptive} < {MIN_MAL_FOR_VALID_RESULT}: vacuous output"
        )
    else:
        result_json["results_are_valid"] = True
        result_json["validity_note"] = f"N_mal={total_maladaptive}: valid"

    # Ablation criterion: full HDR should outperform pure MPC
    hdr_full_gain = variants_out["hdr_full"]["mean_gain"]
    mpc_only_gain = variants_out["mpc_only"]["mean_gain"]
    ablation_criterion_met = hdr_full_gain >= mpc_only_gain
    result_json["ablation_criterion_met"] = ablation_criterion_met

    w_tau_rho096 = 0.5 / (1 - 0.96**2)
    if ablation_criterion_met:
        result_json["ablation_criterion_note"] = (
            f"hdr_full ({hdr_full_gain:+.4f}) >= mpc_only ({mpc_only_gain:+.4f})"
        )
    else:
        result_json["ablation_criterion_note"] = (
            f"hdr_full ({hdr_full_gain:+.4f}) < mpc_only ({mpc_only_gain:+.4f}) — "
            f"EXPECTED_AT_SHORT_T: tau-tilde weight w_tau={w_tau_rho096:.2f} "
            f"penalises recovery attempts at T={cfg['steps_per_episode']} "
            f"before escape benefit is realised. "
            f"Criterion expected to pass at T>=128. "
            f"Production run required (T=256, n_seeds=20, n_ep=30)."
        )

    # Gate results_are_valid on ablation criterion at production scale
    T_production = cfg.get("steps_per_episode", 256)
    if T_production >= 128 and not ablation_criterion_met:
        result_json["results_are_valid"] = False
        result_json["validity_note"] = (
            result_json.get("validity_note", "")
            + f" | ablation_criterion_met=False at T={T_production} — investigate."
        )

    # Save JSON
    from hdr_validation.provenance import get_provenance
    result_json["provenance"] = get_provenance()
    out_path = output_dir / "ablation_results.json"
    out_path.write_text(encoding="utf-8", data=json.dumps(result_json, indent=2))

    # Print ASCII table
    print("\n" + "+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")
    print("| {:20s} | {:8s} | {:20s} | {:8s} |".format("Variant", "Gain", "95% CI", "Win Rate"))
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")
    for name, v in variants_out.items():
        gain_str = f"{v['mean_gain']:+.1%}"
        ci_str = f"[{v['ci_lo']:+.1%}, {v['ci_hi']:+.1%}]"
        win_str = f"{v['win_rate']:.1%}"
        print("| {:20s} | {:8s} | {:20s} | {:8s} |".format(name, gain_str, ci_str, win_str))
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")

    # PASS/FAIL criterion
    status = "PASS" if ablation_criterion_met else "FAIL"
    print(f"\n  [{status}] hdr_full gain ({hdr_full_gain:+.4f}) >= mpc_only gain ({mpc_only_gain:+.4f})")
    if not ablation_criterion_met and T_production < 128:
        print(f"  Note: EXPECTED_AT_SHORT_T (T={T_production}, w_tau={w_tau_rho096:.1f})")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_08()
