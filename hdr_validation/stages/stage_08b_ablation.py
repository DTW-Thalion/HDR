"""
Stage 08b — Multi-Axis Asymmetric Ablation (HDR v5.3)
=====================================================

Companion to Stage 08 that uses an asymmetric J coupling matrix to
demonstrate non-zero marginal gains from BOTH the coherence penalty
and the calibration adjustment.

Stage 08 uses a single-axis maladaptive basin with isotropic coupling,
where tau-tilde dominates and coherence/calibration gains are negligible.
Stage 08b designs a scenario where:

  - J has 3 strongly-coupled axes (row norm ~0.7) and 5 weakly-coupled
    axes (row norm ~0.08), ratio >= 5x.
  - Initial displacement is concentrated on WEAKLY-coupled axes.
  - R_Brier is set to 0.04 (near R_Brier_max = 0.05) so calibration
    kappa overshoot is large enough to register.
  - The J-proportional coherence weighting in solve_mode_a routes
    planning effort toward strongly-coupled axes, producing measurable
    marginal gain over bare MPC.

Five ablation variants (same as Stage 08):

  hdr_full           : full HDR (w2=0.5, w3=0.3, calibrated, J_coupling)
  mpc_only           : pure MPC (w2=0, w3=0, calibrated)
  mpc_plus_surrogate : MPC + tau-tilde surrogate (w2=0.5, w3=0, calibrated)
  mpc_plus_coherence : MPC + coherence (w2=0, w3=0.3, calibrated, J_coupling)
  hdr_no_calib       : full HDR without calibration adjustment

Results saved to results/stage_08b/ablation_asymmetric_results.json.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .stage_08_ablation import AblationConfig, ABLATION_VARIANTS, _bootstrap_ci

ROOT = Path(__file__).parent.parent.parent


def _build_asymmetric_J(n: int = 8) -> np.ndarray:
    """Build the asymmetric J coupling matrix.

    3 strongly-coupled axes [0,1,2]: row norm ~0.7
    5 weakly-coupled axes [3..7]:    row norm ~0.08
    Row-norm ratio (strong / weak mean) >= 5.0
    """
    rng_J = np.random.default_rng(8008)
    J = np.zeros((n, n))
    # Strong axes [0,1,2]
    for i in range(3):
        row = rng_J.standard_normal(n)
        row /= np.linalg.norm(row)
        J[i, :] = row * 0.7
    # Weak axes [3,4,5,6,7]
    for i in range(3, n):
        row = rng_J.standard_normal(n)
        row /= np.linalg.norm(row)
        J[i, :] = row * 0.08
    # Verify row-norm ratio
    strong_norms = np.linalg.norm(J[:3, :], axis=1)
    weak_norms = np.linalg.norm(J[3:, :], axis=1)
    assert strong_norms.mean() / weak_norms.mean() >= 5.0, (
        f"Row-norm ratio {strong_norms.mean() / weak_norms.mean():.2f} < 5.0"
    )
    return J


def _make_benchmark_config_8b(
    n_seeds: int = 20, n_ep: int = 30, T: int = 256,
) -> dict[str, Any]:
    """Create the Stage 8b benchmark configuration with asymmetric J."""
    n = 8
    J = _build_asymmetric_J(n)

    cfg: dict[str, Any] = {
        "state_dim": n,
        "obs_dim": 16,
        "control_dim": n,
        "disturbance_dim": n,
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
        "model_mismatch_bound": 0.347,
        "kappa_lo": 0.55,
        "kappa_hi": 0.75,
        "pA": 0.70,
        "qmin": 0.15,
        "steps_per_day": 48,
        "dt_minutes": 30,
        "coherence_window": 24,
        "default_burden_budget": 56.0,
        "circadian_locked_controls": [5, 6],
        "R_brier_max": 0.05,
        "omega_min_factor": 0.005,
        "T_C_max": 50,
        "k_calib": 1.0,
        "sigma_dither": 0.08,
        "epsilon_control": 0.50,
        "missing_fraction_target": 0.516,
        "mode1_base_rate": 0.16,
        # Benchmark parameters
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "steps_per_episode": T,
        "profile_name": "asymmetric_ablation",
        # Asymmetric J coupling matrix (JSON-serialisable)
        "J_coupling": J.tolist(),
    }

    # Verify J does not destabilise A_k for the maladaptive basin
    from hdr_validation.model.slds import make_evaluation_model, spectral_radius
    rng_check = np.random.default_rng(101)
    model = make_evaluation_model(cfg, rng_check)
    basin_mal = model.basins[1]  # rho=0.96 maladaptive
    dt = float(cfg["dt_minutes"]) / 60.0  # hours
    A_coupled = basin_mal.A + dt * J
    sr = spectral_radius(A_coupled)
    if sr >= 1.0:
        # Scale J down to maintain stability
        scale = 0.95 / sr
        J *= scale
        cfg["J_coupling"] = J.tolist()

    return cfg


def _kappa_schedule_8b(t: int, T: int, cfg: dict) -> float:
    """Time-varying kappa_hat for Stage 8b.

    Same ramp as Stage 08 but starting further below kappa_lo
    to exercise coherence penalty more aggressively.
    """
    kappa_lo = float(cfg.get("kappa_lo", 0.55))
    kappa_hi = float(cfg.get("kappa_hi", 0.75))
    kappa_start = kappa_lo - 0.15
    ramp_end = int(T * 2 / 3)
    if ramp_end <= 0:
        return kappa_hi
    if t >= ramp_end:
        return kappa_hi
    return kappa_start + (kappa_hi - kappa_start) * (t / ramp_end)


def _get_kappa_hat_8b(
    ablation_cfg: AblationConfig, t: int, T: int, cfg: dict,
) -> float:
    """Compute effective kappa_hat for Stage 8b.

    Uses R_Brier = 0.04 (near R_Brier_max = 0.05) so calibration
    adjustment produces a measurably larger kappa overshoot than
    Stage 08's R_Brier = 0.03.
    """
    from hdr_validation.inference.ici import compute_p_A_robust

    kappa_lo = float(cfg.get("kappa_lo", 0.55))
    kappa_hi = float(cfg.get("kappa_hi", 0.75))
    kappa_base = _kappa_schedule_8b(t, T, cfg)

    # Elevated R_Brier to exercise calibration adjustment
    R_brier_episode = 0.04

    p_A_base = float(cfg.get("pA", 0.70))

    if ablation_cfg.use_calibration:
        k_calib = float(cfg.get("k_calib", 1.0))
        p_A_robust = compute_p_A_robust(
            p_A=p_A_base,
            k_calib=k_calib,
            R_brier=R_brier_episode,
        )
        overshoot = (p_A_robust - p_A_base) / max(1.0 - p_A_base, 1e-6)
        kappa_hat = kappa_base * (1.0 - overshoot * 0.15)
    else:
        kappa_hat = kappa_base

    return float(np.clip(kappa_hat, kappa_lo - 0.20, kappa_hi))


def _run_episode_8b(
    cfg: dict[str, Any],
    basin_idx: int,
    rng: np.random.Generator,
    ablation_cfg: AblationConfig,
) -> dict[str, Any]:
    """Run one Stage 8b episode with asymmetric J coupling.

    Initial displacement is concentrated on WEAKLY-coupled axes [3..7]
    so that J-proportional coherence weighting has differential effect.
    """
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.control.lqr import dlqr
    from hdr_validation.model.coherence import coherence_grad, coherence_penalty

    model_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    eval_model = make_evaluation_model(cfg, model_rng)
    basin = eval_model.basins[basin_idx]
    target = build_target_set(basin_idx, cfg)
    T = cfg["steps_per_episode"]
    n = cfg["state_dim"]
    lambda_u = float(cfg.get("lambda_u", 0.1))

    # Build variant config — include or exclude J_coupling
    variant_cfg = dict(cfg)
    variant_cfg["w2"] = ablation_cfg.w2
    variant_cfg["w3"] = ablation_cfg.w3
    if not ablation_cfg.use_calibration:
        variant_cfg["R_brier_max"] = 1.0
        variant_cfg["k_calib"] = 0.0

    # Only provide J_coupling when coherence is active (w3 > 0)
    # so that mpc_only and mpc_plus_surrogate use standard MPC
    if ablation_cfg.w3 <= 0:
        variant_cfg.pop("J_coupling", None)

    # Pooled LQR baseline
    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * lambda_u
    try:
        K_pool, _ = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
    except Exception:
        K_pool = np.zeros((n, n))

    # Initial state: displacement concentrated on WEAKLY-coupled axes [3..7]
    # Shared starting point for both policies
    x0 = np.zeros(n)
    x0[3:] = rng.normal(loc=0.8, scale=0.3, size=n - 3)  # strong displacement
    x0[:3] = rng.normal(loc=0.0, scale=0.1, size=3)       # minimal displacement
    x_hdr = x0.copy()
    x_base = x0.copy()
    x_ref = np.zeros(n)
    P_hat = np.eye(n) * 0.2

    hdr_cost = 0.0
    baseline_cost = 0.0

    coupling_scales: list[float] = []
    kappa_hats: list[float] = []
    u_hdr_norms: list[float] = []
    u_base_norms: list[float] = []

    for t in range(T):
        # State costs on independent trajectories
        hdr_state_cost = float(np.dot(x_hdr, x_hdr))
        base_state_cost = float(np.dot(x_base, x_base))

        kappa_hat_t = _get_kappa_hat_8b(ablation_cfg, t, T, variant_cfg)

        # HDR Mode A control (on HDR trajectory)
        try:
            res = solve_mode_a(
                x_hdr, P_hat, basin, target,
                kappa_hat=kappa_hat_t,
                config=variant_cfg, step=t,
            )
            u_hdr = res.u
        except Exception:
            u_hdr = np.zeros(cfg["control_dim"])

        # Baseline control on baseline trajectory (pooled LQR)
        u_base = -K_pool @ (x_base - x_ref)
        u_base = np.clip(u_base, -0.6, 0.6)

        hdr_cost += hdr_state_cost + lambda_u * float(np.dot(u_hdr, u_hdr))
        baseline_cost += base_state_cost + lambda_u * float(np.dot(u_base, u_base))

        # Diagnostics
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

        # Shared process noise for paired comparison
        w = rng.multivariate_normal(np.zeros(n), basin.Q)

        # Advance independent trajectories
        x_hdr = basin.A @ x_hdr + basin.B @ u_hdr + basin.b + w
        x_base = basin.A @ x_base + basin.B @ u_base + basin.b + w

    gain = (baseline_cost - hdr_cost) / max(baseline_cost, 1e-12)

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


def run_stage_08b(
    n_seeds: int = 20,
    n_ep: int = 30,
    T: int = 256,
    output_dir: Path | None = None,
    fast_mode: bool = False,
) -> dict:
    """Run the Stage 08b multi-axis asymmetric ablation study.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds.
    n_ep : int
        Episodes per seed.
    T : int
        Steps per episode.
    output_dir : Path or None
        Directory for output files. Defaults to results/stage_08b/.
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

    cfg = _make_benchmark_config_8b(n_seeds=n_seeds, n_ep=n_ep, T=T)

    if output_dir is None:
        output_dir = ROOT / "results" / "stage_08b"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [101 + i * 101 for i in range(n_seeds)]

    N_MAL_MIN = 6
    forced_mal_set: set[tuple[int, int]] = set()
    if n_seeds * n_ep < 20:
        count = 0
        for s_idx in range(n_seeds):
            for e_idx in range(n_ep):
                if count >= N_MAL_MIN:
                    break
                forced_mal_set.add((s_idx, e_idx))
                count += 1

    variant_gains: dict[str, list[float]] = {v.name: [] for v in ABLATION_VARIANTS}
    episode_results: dict[str, list[dict]] = {v.name: [] for v in ABLATION_VARIANTS}

    total_maladaptive = 0
    for seed_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ep_idx in range(n_ep):
            is_forced = (seed_idx, ep_idx) in forced_mal_set
            is_mal = is_forced or (rng.random() < 0.30)
            basin_idx = 1 if is_mal else rng.choice([0, 2])

            if basin_idx != 1:
                continue

            total_maladaptive += 1
            ep_rng_base = int(seed * 10000 + ep_idx)

            for abl_cfg in ABLATION_VARIANTS:
                ep_rng = np.random.default_rng(ep_rng_base)
                result = _run_episode_8b(cfg, basin_idx=1, rng=ep_rng, ablation_cfg=abl_cfg)
                variant_gains[abl_cfg.name].append(result["gain"])
                episode_results[abl_cfg.name].append(result)

    # Compute summary statistics
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

    # J matrix diagnostics
    J = np.asarray(cfg["J_coupling"])
    strong_norms = np.linalg.norm(J[:3, :], axis=1)
    weak_norms = np.linalg.norm(J[3:, :], axis=1)

    result_json: dict[str, Any] = {
        "scenario": "multi_axis_asymmetric",
        "variants": variants_out,
        "n_seeds": n_seeds,
        "n_ep_per_seed": n_ep,
        "T": T,
        "total_maladaptive_episodes": total_maladaptive,
        "J_diagnostics": {
            "strong_axis_mean_norm": round(float(strong_norms.mean()), 4),
            "weak_axis_mean_norm": round(float(weak_norms.mean()), 4),
            "row_norm_ratio": round(float(strong_norms.mean() / weak_norms.mean()), 2),
        },
    }

    MIN_MAL_FOR_VALID_RESULT = 5
    if total_maladaptive < MIN_MAL_FOR_VALID_RESULT:
        import warnings
        warnings.warn(
            f"Stage 08b: only {total_maladaptive} maladaptive episodes collected "
            f"(minimum {MIN_MAL_FOR_VALID_RESULT} required for valid ablation statistics). "
            f"Results are NOT suitable for manuscript use.",
            stacklevel=2,
        )
        result_json["results_are_valid"] = False
        result_json["validity_note"] = (
            f"N_mal={total_maladaptive} < {MIN_MAL_FOR_VALID_RESULT}: vacuous output"
        )
    else:
        result_json["results_are_valid"] = True
        result_json["validity_note"] = f"N_mal={total_maladaptive}: valid"

    # Primary criterion: hdr_full >= mpc_only
    hdr_full_gain = variants_out["hdr_full"]["mean_gain"]
    mpc_only_gain = variants_out["mpc_only"]["mean_gain"]
    ablation_criterion_met = hdr_full_gain >= mpc_only_gain
    result_json["ablation_criterion_met"] = ablation_criterion_met

    # Marginal contribution checks
    coherence_marginal = (
        variants_out["mpc_plus_coherence"]["mean_gain"] - mpc_only_gain
    )
    calib_marginal = (
        variants_out["hdr_full"]["mean_gain"]
        - variants_out["hdr_no_calib"]["mean_gain"]
    )
    result_json["coherence_marginal_gain"] = round(coherence_marginal, 6)
    result_json["calibration_marginal_gain"] = round(calib_marginal, 6)

    w_tau_rho096 = 0.5 / (1 - 0.96**2)
    T_production = cfg.get("steps_per_episode", 256)
    if ablation_criterion_met:
        result_json["ablation_criterion_note"] = (
            f"hdr_full ({hdr_full_gain:+.4f}) >= mpc_only ({mpc_only_gain:+.4f})"
        )
    else:
        result_json["ablation_criterion_note"] = (
            f"hdr_full ({hdr_full_gain:+.4f}) < mpc_only ({mpc_only_gain:+.4f}) — "
            f"EXPECTED_AT_SHORT_T: tau-tilde weight w_tau={w_tau_rho096:.2f} "
            f"penalises recovery attempts at T={T_production} "
            f"before escape benefit is realised."
        )

    if T_production >= 128 and not ablation_criterion_met:
        result_json["results_are_valid"] = False
        result_json["validity_note"] = (
            result_json.get("validity_note", "")
            + f" | ablation_criterion_met=False at T={T_production} — investigate."
        )

    # Save JSON
    out_path = output_dir / "ablation_asymmetric_results.json"
    out_path.write_text(json.dumps(result_json, indent=2))

    # Print ASCII table
    print("\n  Stage 08b — Multi-Axis Asymmetric Ablation")
    print(f"  J row-norm ratio: {strong_norms.mean() / weak_norms.mean():.1f}x "
          f"(strong={strong_norms.mean():.3f}, weak={weak_norms.mean():.3f})")
    print("\n" + "+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")
    print("| {:20s} | {:8s} | {:20s} | {:8s} |".format("Variant", "Gain", "95% CI", "Win Rate"))
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")
    for name, v in variants_out.items():
        gain_str = f"{v['mean_gain']:+.1%}"
        ci_str = f"[{v['ci_lo']:+.1%}, {v['ci_hi']:+.1%}]"
        win_str = f"{v['win_rate']:.1%}"
        print("| {:20s} | {:8s} | {:20s} | {:8s} |".format(name, gain_str, ci_str, win_str))
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 22 + "+" + "-" * 10 + "+")

    print(f"\n  Coherence marginal gain:    {coherence_marginal:+.4f}")
    print(f"  Calibration marginal gain:  {calib_marginal:+.4f}")

    status = "PASS" if ablation_criterion_met else "FAIL"
    print(f"\n  [{status}] hdr_full gain ({hdr_full_gain:+.4f}) >= mpc_only gain ({mpc_only_gain:+.4f})")
    if not ablation_criterion_met and T_production < 128:
        print(f"  Note: EXPECTED_AT_SHORT_T (T={T_production}, w_tau={w_tau_rho096:.1f})")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_08b()
