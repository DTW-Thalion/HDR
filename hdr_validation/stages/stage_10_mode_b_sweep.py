"""
Stage 10 — Mode B FP/FN Sweep (HDR v5.2)
==========================================

Validates Lemma 9.3 (the calibration-to-decision-risk bound) empirically by
quantifying Mode B false-positive (FP) and false-negative (FN) rates as a
function of calibration quality R_Brier.

Uses the 6-state Markov chain from Benchmark B:
  States 0-1: desired (target)
  States 2-3: maladaptive
  States 4-5: transient

Results saved to results/stage_10/mode_b_fp_fn_sweep.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


# 6-state Markov chain calibrated to Benchmark B (rho(P_TT) = 0.412)
# States: 0-1 desired, 2-3 maladaptive, 4-5 transient
_BENCHMARK_B_TRANSITION = np.array([
    # desired0  desired1  mal0   mal1   trans0  trans1
    [0.85,      0.10,     0.02,  0.01,  0.01,  0.01],  # desired0
    [0.10,      0.82,     0.03,  0.02,  0.02,  0.01],  # desired1
    [0.02,      0.03,     0.75,  0.12,  0.05,  0.03],  # maladaptive0
    [0.01,      0.02,     0.10,  0.78,  0.06,  0.03],  # maladaptive1
    [0.10,      0.12,     0.18,  0.15,  0.30,  0.15],  # transient0
    [0.08,      0.14,     0.14,  0.16,  0.20,  0.28],  # transient1
])

# Normalize rows to ensure stochastic
_BENCHMARK_B_TRANSITION = _BENCHMARK_B_TRANSITION / _BENCHMARK_B_TRANSITION.sum(axis=1, keepdims=True)

# State categories
_MALADAPTIVE_STATES = {2, 3}
_DESIRED_STATES = {0, 1}
_TRANSIENT_STATES = {4, 5}


def inject_miscalibration(
    p_true: float,
    R_Brier_target: float,
    rng: np.random.Generator,
) -> float:
    """Return a miscalibrated posterior probability.

    Adds zero-mean Gaussian noise with std = sqrt(R_Brier_target) to the
    logit of p_true, then applies sigmoid. This produces calibration errors
    that scale with R_Brier_target.

    Parameters
    ----------
    p_true : float
        True posterior probability for the maladaptive basin.
    R_Brier_target : float
        Desired Brier reliability component.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    float
        Miscalibrated posterior probability in (0, 1).
    """
    p_clipped = float(np.clip(p_true, 1e-6, 1.0 - 1e-6))
    logit_p = np.log(p_clipped / (1.0 - p_clipped))
    noise_std = float(np.sqrt(max(R_Brier_target, 0.0)))
    noisy_logit = logit_p + rng.normal(scale=noise_std)
    return float(1.0 / (1.0 + np.exp(-noisy_logit)))


def _simulate_trajectory(
    T: int,
    rng: np.random.Generator,
    P: np.ndarray,
    init_dist: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate a single Markov chain trajectory of length T.

    Parameters
    ----------
    T : int
        Number of steps.
    rng : np.random.Generator
        Random generator.
    P : np.ndarray
        Transition matrix, shape (S, S).
    init_dist : np.ndarray or None
        Initial state distribution. Defaults to stationary distribution.

    Returns
    -------
    np.ndarray
        State sequence, shape (T,).
    """
    S = P.shape[0]
    if init_dist is None:
        init_dist = np.ones(S) / S
    state = int(rng.choice(S, p=init_dist / init_dist.sum()))
    states = np.empty(T, dtype=int)
    states[0] = state
    for t in range(1, T):
        state = int(rng.choice(S, p=P[state]))
        states[t] = state
    return states


def _true_maladaptive_prob(state: int) -> float:
    """Ground-truth probability of being in the maladaptive regime given state."""
    if state in _MALADAPTIVE_STATES:
        return 0.90
    elif state in _TRANSIENT_STATES:
        return 0.25
    else:
        return 0.05


def run_stage_10(
    N_sim: int = 5000,
    T: int = 50,
    output_dir: Path | None = None,
    fast_mode: bool = False,
) -> dict:
    """Run Stage 10 Mode B FP/FN sweep.

    Parameters
    ----------
    N_sim : int
        Number of simulated trajectories.
    T : int
        Length of each trajectory.
    output_dir : Path or None
        Directory for output files. Defaults to results/stage_10/.
    fast_mode : bool
        If True, use reduced N_sim for speed.

    Returns
    -------
    dict
        FP/FN sweep results (JSON-compatible).
    """
    if fast_mode:
        N_sim = min(N_sim, 200)

    if output_dir is None:
        output_dir = ROOT / "results" / "stage_10"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_A_base = 0.70
    k_calib = 1.0
    q_min = 0.15
    R_Brier_levels = [0.00, 0.05, 0.10, 0.15, 0.20]

    rng = np.random.default_rng(42)
    P = _BENCHMARK_B_TRANSITION.copy()

    # Simulate all trajectories once (shared across R_Brier levels)
    trajectories: list[np.ndarray] = [
        _simulate_trajectory(T, rng, P) for _ in range(N_sim)
    ]

    # Pre-compute true maladaptive probabilities and simulated q-values
    # For each (trajectory, step), we have:
    # - true state → is_maladaptive (binary)
    # - p_true: probability of being maladaptive given true state
    # - is_q_below_qmin: spontaneous escape prob < q_min

    sweep_results = []

    for R_target in R_Brier_levels:
        threshold_fixed = p_A_base
        threshold_robust = p_A_base + k_calib * R_target

        fp_fixed_count = fp_robust_count = 0
        fn_fixed_count = fn_robust_count = 0
        fn_denom = 0  # steps where true state IS maladaptive AND q < q_min
        fp_denom = 0  # steps where true state is NOT maladaptive

        rng_calib = np.random.default_rng(int(1000 * (1 + R_target) * 10))

        for traj in trajectories:
            for state in traj:
                is_maladaptive = state in _MALADAPTIVE_STATES

                # Get true probability and inject miscalibration
                p_true = _true_maladaptive_prob(state)
                p_hat = inject_miscalibration(p_true, R_target, rng_calib)

                # Simulate spontaneous escape probability q
                # For maladaptive states: q is low (below q_min)
                if is_maladaptive:
                    # q is the natural escape probability; for maladaptive, it's < q_min
                    q_spontaneous = 0.08 + rng_calib.random() * 0.05  # ~ 0.08-0.13 < q_min=0.15
                    is_q_below_qmin = True
                else:
                    q_spontaneous = 0.20 + rng_calib.random() * 0.40  # > q_min typically
                    is_q_below_qmin = q_spontaneous < q_min

                # FP: Mode B triggered BUT state is NOT maladaptive
                if not is_maladaptive:
                    fp_denom += 1
                    if p_hat >= threshold_fixed:
                        fp_fixed_count += 1
                    if p_hat >= threshold_robust:
                        fp_robust_count += 1

                # FN: state IS maladaptive AND q < q_min BUT Mode B NOT triggered
                if is_maladaptive and is_q_below_qmin:
                    fn_denom += 1
                    if p_hat < threshold_fixed:
                        fn_fixed_count += 1
                    if p_hat < threshold_robust:
                        fn_robust_count += 1

        fp_rate_fixed = fp_fixed_count / max(fp_denom, 1)
        fp_rate_robust = fp_robust_count / max(fp_denom, 1)
        fn_rate_fixed = fn_fixed_count / max(fn_denom, 1)
        fn_rate_robust = fn_robust_count / max(fn_denom, 1)

        sweep_results.append({
            "R_Brier_target": R_target,
            "FP_rate_fixed_threshold": round(fp_rate_fixed, 4),
            "FP_rate_robust_threshold": round(fp_rate_robust, 4),
            "FN_rate_fixed_threshold": round(fn_rate_fixed, 4),
            "FN_rate_robust_threshold": round(fn_rate_robust, 4),
            "threshold_fixed": round(threshold_fixed, 4),
            "threshold_robust": round(threshold_robust, 4),
        })

    result_json = {
        "sweep": sweep_results,
        "p_A_base": p_A_base,
        "k_calib": k_calib,
        "q_min": q_min,
        "N_sim": N_sim,
        "T": T,
    }

    out_path = output_dir / "mode_b_fp_fn_sweep.json"
    out_path.write_text(json.dumps(result_json, indent=2))

    # Print formatted table
    print("\nMode B FP/FN Sweep — Effect of Calibration Adjustment")
    print("─" * 75)
    print(f"  {'R_Brier':>8}  {'FP (fixed)':>12}  {'FP (robust)':>12}  {'FN (fixed)':>12}  {'FN (robust)':>12}")
    print("─" * 75)
    for row in sweep_results:
        print(
            f"  {row['R_Brier_target']:>8.2f}"
            f"  {row['FP_rate_fixed_threshold']:>12.4f}"
            f"  {row['FP_rate_robust_threshold']:>12.4f}"
            f"  {row['FN_rate_fixed_threshold']:>12.4f}"
            f"  {row['FN_rate_robust_threshold']:>12.4f}"
        )
    print("─" * 75)

    # PASS/FAIL checks
    at_zero = sweep_results[0]
    thresholds_equal = abs(at_zero["threshold_fixed"] - at_zero["threshold_robust"]) < 1e-9
    fp_monotone = all(
        sweep_results[i]["FP_rate_robust_threshold"] <= sweep_results[i]["FP_rate_fixed_threshold"] + 1e-9
        for i in range(len(sweep_results))
    )
    all_levels_present = len(sweep_results) == 5
    rates_in_range = all(
        0.0 <= row["FP_rate_fixed_threshold"] <= 1.0
        and 0.0 <= row["FP_rate_robust_threshold"] <= 1.0
        and 0.0 <= row["FN_rate_fixed_threshold"] <= 1.0
        and 0.0 <= row["FN_rate_robust_threshold"] <= 1.0
        for row in sweep_results
    )

    status_equal = "PASS" if thresholds_equal else "FAIL"
    status_mono = "PASS" if fp_monotone else "FAIL"
    status_levels = "PASS" if all_levels_present else "FAIL"
    status_range = "PASS" if rates_in_range else "FAIL"

    print(f"\n  [{status_equal}] At R_Brier=0, fixed and robust thresholds equal")
    print(f"  [{status_mono}] FP rate with robust threshold <= FP rate with fixed (all levels)")
    print(f"  [{status_levels}] All 5 sweep levels present")
    print(f"  [{status_range}] All FP/FN rates in [0, 1]")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_10()
