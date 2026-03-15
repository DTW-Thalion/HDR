"""
Stage 15 — Proxy-Composite Estimation Benchmark (HDR v7.0)
============================================================
Validates Claim 32: Proxy-composite estimation quality.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def run_stage_15(
    n_scenarios: int = 5,
    sigma_values: list | None = None,
    T: int = 50,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Stage 15: Proxy-composite estimation benchmark."""
    from hdr_validation.model.slds import make_evaluation_model

    t0 = time.perf_counter()
    sigma_values = sigma_values or [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    cfg = {
        "state_dim": 8, "obs_dim": 16, "control_dim": 8,
        "disturbance_dim": 8, "K": 3, "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 64,
    }
    rng = np.random.default_rng(101)
    model = make_evaluation_model(cfg, rng)

    results: dict[str, Any] = {"checks": []}
    checks = results["checks"]

    # For each sigma_proxy, compute RMSE of latent state estimation
    rmse_values = []
    for sigma_proxy in sigma_values:
        rmse_per_scenario = []
        for sc in range(n_scenarios):
            basin = model.basins[0]
            x_true = rng.normal(size=cfg["state_dim"]) * 0.1
            errors_sq = []
            for t_step in range(T):
                x_true = basin.A @ x_true + rng.normal(scale=0.1, size=cfg["state_dim"])
                # Direct observation + proxy noise
                y_direct = basin.C @ x_true + rng.normal(scale=0.1, size=cfg["obs_dim"])
                y_proxy = y_direct + rng.normal(scale=sigma_proxy, size=cfg["obs_dim"])
                # Simple estimate: pseudoinverse
                try:
                    x_hat = np.linalg.lstsq(basin.C, y_proxy - basin.c, rcond=None)[0]
                except Exception:
                    x_hat = np.zeros(cfg["state_dim"])
                errors_sq.append(float(np.sum((x_hat - x_true)**2)))
            rmse_per_scenario.append(float(np.sqrt(np.mean(errors_sq))))
        rmse_values.append(float(np.mean(rmse_per_scenario)))

    # Check 1: RMSE monotonically non-decreasing in sigma_proxy
    monotonic = all(rmse_values[i] <= rmse_values[i+1] + 0.1
                    for i in range(len(rmse_values)-1))
    checks.append({
        "check": "rmse_monotonic_in_sigma",
        "passed": monotonic,
        "value": str([f"{r:.3f}" for r in rmse_values]),
        "note": "RMSE should increase with proxy noise",
    })

    # Check 2: RMSE < 2x direct at sigma=0.5
    if 0.5 in sigma_values and 0.0 in sigma_values:
        idx_0 = sigma_values.index(0.0)
        idx_05 = sigma_values.index(0.5)
        ratio = rmse_values[idx_05] / max(rmse_values[idx_0], 1e-10)
        checks.append({
            "check": "rmse_ratio_at_sigma_05",
            "passed": ratio < 2.0,
            "value": f"{ratio:.2f}",
            "note": "Should be < 2x direct observation RMSE",
        })

    # Check 3: No catastrophic failure at sigma=2.0
    if 2.0 in sigma_values:
        idx_2 = sigma_values.index(2.0)
        checks.append({
            "check": "no_catastrophic_failure_sigma_2",
            "passed": np.isfinite(rmse_values[idx_2]) and rmse_values[idx_2] < 100.0,
            "value": f"{rmse_values[idx_2]:.3f}",
            "note": "RMSE should remain bounded",
        })

    elapsed = time.perf_counter() - t0
    results["elapsed"] = elapsed

    out_dir = ROOT / "results" / "stage_15"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_pass = sum(1 for c in checks if c["passed"])
    print(f"  Stage 15: {n_pass}/{len(checks)} checks passed ({elapsed:.1f}s)")
    return results
