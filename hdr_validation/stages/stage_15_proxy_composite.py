"""
Stage 15 — Proxy-Composite Estimation Benchmark
============================================================
Validates Claim 32: Proxy-composite estimation quality.

Two estimators are compared:
  (1) Pseudoinverse (lstsq) — single-step, ignores dynamics
  (2) Kalman filter — uses per-basin A_k for prediction, full C_k for update

The Kalman filter should maintain RMSE ratio < 2x at sigma_proxy=0.5.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _run_pinv_estimator(
    basin: Any,
    cfg: dict,
    rng: np.random.Generator,
    sigma_proxy: float,
    T: int,
) -> tuple[float, list[float]]:
    """Run pseudoinverse estimator, return (rmse, errors_sq_list)."""
    n = cfg["state_dim"]
    m = cfg["obs_dim"]
    x_true = rng.normal(size=n) * 0.1
    errors_sq = []
    for _ in range(T):
        x_true = basin.A @ x_true + rng.normal(scale=0.1, size=n)
        y_direct = basin.C @ x_true + rng.normal(scale=0.1, size=m)
        y_proxy = y_direct + rng.normal(scale=sigma_proxy, size=m)
        try:
            x_hat = np.linalg.lstsq(basin.C, y_proxy - basin.c, rcond=None)[0]
        except Exception:
            x_hat = np.zeros(n)
        errors_sq.append(float(np.sum((x_hat - x_true) ** 2)))
    rmse = float(np.sqrt(np.mean(errors_sq)))
    return rmse, errors_sq


def _run_kalman_estimator(
    basin: Any,
    cfg: dict,
    rng: np.random.Generator,
    sigma_proxy: float,
    T: int,
) -> tuple[float, list[float]]:
    """Run Kalman filter estimator, return (rmse, errors_sq_list).

    Uses the per-basin dynamics A_k for prediction and full observation
    model C_k for update. Process/observation noise covariances are matched
    to the simulation's noise parameters.
    """
    n = cfg["state_dim"]
    m = cfg["obs_dim"]

    # Noise covariances matching the simulation
    Q = np.eye(n) * 0.01  # process noise variance = 0.1^2
    R = np.eye(m) * (0.01 + sigma_proxy**2)  # base obs noise + proxy noise

    # Diffuse initialisation
    x_est = np.zeros(n)
    P_est = np.eye(n) * 10.0

    x_true = rng.normal(size=n) * 0.1
    errors_sq = []

    for _ in range(T):
        x_true = basin.A @ x_true + rng.normal(scale=0.1, size=n)
        y_direct = basin.C @ x_true + rng.normal(scale=0.1, size=m)
        y_proxy = y_direct + rng.normal(scale=sigma_proxy, size=m)

        # Predict
        x_pred = basin.A @ x_est + basin.b
        P_pred = basin.A @ P_est @ basin.A.T + Q

        # Update (using solve for numerical stability)
        S = basin.C @ P_pred @ basin.C.T + R
        S = 0.5 * (S + S.T)  # symmetrise
        innov = y_proxy - (basin.C @ x_pred + basin.c)
        try:
            K = np.linalg.solve(S.T, (P_pred @ basin.C.T).T).T
        except np.linalg.LinAlgError:
            K = P_pred @ basin.C.T @ np.linalg.pinv(S)

        x_est = x_pred + K @ innov
        P_est = (np.eye(n) - K @ basin.C) @ P_pred

        errors_sq.append(float(np.sum((x_est - x_true) ** 2)))

    rmse = float(np.sqrt(np.mean(errors_sq)))
    return rmse, errors_sq


def run_stage_15(
    n_scenarios: int = 5,
    sigma_values: list | None = None,
    T: int = 50,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Stage 15: Proxy-composite estimation benchmark.

    Runs both pseudoinverse and Kalman filter estimators across a sweep
    of proxy noise levels, then evaluates criteria.
    """
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.defaults import DEFAULTS

    t0 = time.perf_counter()
    sigma_values = sigma_values or [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    cfg = dict(DEFAULTS)
    cfg["max_dwell_len"] = 64
    rng = np.random.default_rng(101)
    model = make_evaluation_model(cfg, rng)

    results: dict[str, Any] = {"checks": []}
    checks = results["checks"]

    # Run both estimators across sigma sweep
    rmse_pinv: list[float] = []
    rmse_kalman: list[float] = []

    header = (
        f"  {'sigma':>6} | {'RMSE(pinv)':>10} | {'Ratio(pinv)':>11} | "
        f"{'RMSE(KF)':>10} | {'Ratio(KF)':>11} | {'KF/pinv':>8}"
    )
    sep = "  " + "-" * len(header.strip())

    print("  === Stage 15 - Proxy-Composite Estimation ===")
    print()
    print(header)
    print(sep)

    for sigma_proxy in sigma_values:
        pinv_vals = []
        kf_vals = []
        for sc in range(n_scenarios):
            # Use separate but deterministic RNG streams for fair comparison
            rng_pinv = np.random.default_rng(101 + sc * 1000 + int(sigma_proxy * 100))
            rng_kf = np.random.default_rng(101 + sc * 1000 + int(sigma_proxy * 100))
            basin = model.basins[0]

            rmse_p, _ = _run_pinv_estimator(basin, cfg, rng_pinv, sigma_proxy, T)
            rmse_k, _ = _run_kalman_estimator(basin, cfg, rng_kf, sigma_proxy, T)
            pinv_vals.append(rmse_p)
            kf_vals.append(rmse_k)

        mean_pinv = float(np.mean(pinv_vals))
        mean_kf = float(np.mean(kf_vals))
        rmse_pinv.append(mean_pinv)
        rmse_kalman.append(mean_kf)

    # Compute ratios relative to sigma=0
    base_pinv = max(rmse_pinv[0], 1e-10)
    base_kf = max(rmse_kalman[0], 1e-10)

    for i, sigma_proxy in enumerate(sigma_values):
        ratio_p = rmse_pinv[i] / base_pinv
        ratio_k = rmse_kalman[i] / base_kf
        kf_vs_pinv = rmse_kalman[i] / max(rmse_pinv[i], 1e-10)
        print(
            f"  {sigma_proxy:6.2f} | {rmse_pinv[i]:10.3f} | {ratio_p:10.2f}x | "
            f"{rmse_kalman[i]:10.3f} | {ratio_k:10.2f}x | {kf_vs_pinv:7.3f}x"
        )

    print()

    # ── Checks ────────────────────────────────────────────────────────────

    # Check 1: Pseudoinverse RMSE monotonically non-decreasing
    monotonic_pinv = all(
        rmse_pinv[i] <= rmse_pinv[i + 1] + 0.1
        for i in range(len(rmse_pinv) - 1)
    )
    checks.append({
        "check": "rmse_monotonic_in_sigma",
        "passed": monotonic_pinv,
        "value": str([f"{r:.3f}" for r in rmse_pinv]),
        "note": "Pseudoinverse RMSE should increase with proxy noise",
    })

    # Check 2: Pseudoinverse RMSE ratio at sigma=0.5 (baseline reference)
    if 0.5 in sigma_values and 0.0 in sigma_values:
        idx_0 = sigma_values.index(0.0)
        idx_05 = sigma_values.index(0.5)
        ratio_pinv_05 = rmse_pinv[idx_05] / max(rmse_pinv[idx_0], 1e-10)
        checks.append({
            "check": "rmse_ratio_at_sigma_05",
            "passed": ratio_pinv_05 < 2.0,
            "value": f"{ratio_pinv_05:.2f}",
            "note": "Pseudoinverse: should be < 2x direct RMSE",
        })

    # Check 3: No catastrophic failure at sigma=2.0
    if 2.0 in sigma_values:
        idx_2 = sigma_values.index(2.0)
        checks.append({
            "check": "no_catastrophic_failure_sigma_2",
            "passed": np.isfinite(rmse_pinv[idx_2]) and rmse_pinv[idx_2] < 100.0,
            "value": f"{rmse_pinv[idx_2]:.3f}",
            "note": "Pseudoinverse RMSE should remain bounded",
        })

    # Check 4: Kalman RMSE ratio at sigma=0.5 (the key criterion)
    if 0.5 in sigma_values and 0.0 in sigma_values:
        idx_0 = sigma_values.index(0.0)
        idx_05 = sigma_values.index(0.5)
        ratio_kf_05 = rmse_kalman[idx_05] / max(rmse_kalman[idx_0], 1e-10)
        checks.append({
            "check": "rmse_ratio_at_sigma_05_kalman",
            "passed": ratio_kf_05 < 2.0,
            "value": f"{ratio_kf_05:.2f}",
            "note": "Kalman filter: should be < 2x direct RMSE at sigma=0.5",
        })

    # Check 5: Kalman improves over pseudoinverse at all sigma > 0
    improvement_all = all(
        rmse_kalman[i] <= rmse_pinv[i] * 1.05  # allow 5% tolerance
        for i in range(len(sigma_values))
        if sigma_values[i] > 0
    )
    checks.append({
        "check": "kalman_improvement_over_pinv",
        "passed": improvement_all,
        "value": str([
            f"{rmse_kalman[i] / max(rmse_pinv[i], 1e-10):.3f}"
            for i in range(len(sigma_values))
        ]),
        "note": "Kalman RMSE / pinv RMSE should be <= 1.05 for all sigma > 0",
    })

    # ── Results assembly ──────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    results["elapsed"] = elapsed
    results["sigma_values"] = sigma_values
    results["rmse_pinv"] = rmse_pinv
    results["rmse_kalman"] = rmse_kalman
    results["comparison"] = {
        str(s): {
            "pinv_rmse": rmse_pinv[i],
            "kalman_rmse": rmse_kalman[i],
            "pinv_ratio": rmse_pinv[i] / max(rmse_pinv[0], 1e-10),
            "kalman_ratio": rmse_kalman[i] / max(rmse_kalman[0], 1e-10),
            "kalman_vs_pinv": rmse_kalman[i] / max(rmse_pinv[i], 1e-10),
        }
        for i, s in enumerate(sigma_values)
    }

    from hdr_validation.provenance import get_provenance
    results["provenance"] = get_provenance()
    out_dir = ROOT / "results" / "stage_15"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_pass = sum(1 for c in checks if c["passed"])
    print(f"  Stage 15: {n_pass}/{len(checks)} checks passed ({elapsed:.1f}s)")
    return results
