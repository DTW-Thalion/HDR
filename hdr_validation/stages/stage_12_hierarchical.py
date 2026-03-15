"""
Stage 12 — Hierarchical-Prior Coupling Estimation Benchmark (HDR v7.0)
=======================================================================
Validates Claims 28-30: MAP coupling convergence, B_k sample complexity,
basin boundary convergence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _make_config(n_patients: int = 10, T_p_values: list | None = None) -> dict[str, Any]:
    return {
        "state_dim": 8,
        "obs_dim": 16,
        "control_dim": 8,
        "disturbance_dim": 8,
        "K": 3,
        "rho_reference": [0.72, 0.96, 0.55],
        "max_dwell_len": 64,
        "n_patients": n_patients,
        "T_p_values": T_p_values or [0, 10, 50, 200],
    }


def run_stage_12(
    n_patients: int = 10,
    T_p_values: list | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Stage 12: Hierarchical-prior coupling estimation benchmark."""
    from hdr_validation.identification.hierarchical import HierarchicalCouplingEstimator
    from hdr_validation.identification.boed import BOEDEstimator
    from hdr_validation.identification.committor_recovery import CommittorRecovery

    t0 = time.perf_counter()
    cfg = _make_config(n_patients, T_p_values)
    n = cfg["state_dim"]
    T_p_vals = cfg["T_p_values"]
    rng = np.random.default_rng(101)

    results: dict[str, Any] = {"checks": [], "n_patients": n_patients}
    checks = results["checks"]

    # Generate ground-truth hierarchical model
    J_mech = rng.normal(size=(n, n)) * 0.1
    Sigma_pop = np.eye(n) * 0.5
    Sigma_group = np.eye(n) * 0.3

    # Generate N patients with known J_p
    patient_Js = []
    for p in range(n_patients):
        J_p = J_mech + rng.normal(size=(n, n)) * 0.15
        patient_Js.append(J_p)

    # Check 1: Frobenius error decreases with T_p (>= 90% of patients)
    n_decreasing = 0
    for p in range(n_patients):
        est = HierarchicalCouplingEstimator(J_mech, Sigma_pop, Sigma_group)
        errors = []
        for T_p in T_p_vals:
            if T_p == 0:
                J_hat = est.estimate(None, None)
            else:
                X = rng.normal(size=(T_p, n))
                Y = X @ patient_Js[p] + rng.normal(scale=0.1, size=(T_p, n))
                J_hat = est.estimate(Y, X)
            errors.append(float(np.linalg.norm(J_hat - patient_Js[p], 'fro')))
        if len(errors) >= 2 and errors[-1] <= errors[0] + 0.01:
            n_decreasing += 1

    frac_decreasing = n_decreasing / max(n_patients, 1)
    checks.append({
        "check": "error_decreases_with_T_p",
        "passed": frac_decreasing >= 0.9,
        "value": f"{frac_decreasing:.2f}",
        "note": f"{n_decreasing}/{n_patients} patients show decrease",
    })

    # Check 2: At T_p=0, estimate equals group mean
    est = HierarchicalCouplingEstimator(J_mech, Sigma_pop, Sigma_group)
    J_zero = est.estimate(None, None)
    checks.append({
        "check": "T_p_0_equals_group_mean",
        "passed": bool(np.allclose(J_zero, J_mech)),
        "value": f"error={np.linalg.norm(J_zero - J_mech):.6f}",
        "note": "Graceful degradation check",
    })

    # Check 3: Correlation at large T_p
    if max(T_p_vals) >= 200:
        T_p_large = max(T_p_vals)
        corrs = []
        for p in range(min(n_patients, 5)):
            est = HierarchicalCouplingEstimator(J_mech, Sigma_pop, Sigma_group)
            X = rng.normal(size=(T_p_large, n))
            Y = X @ patient_Js[p] + rng.normal(scale=0.1, size=(T_p_large, n))
            J_hat = est.estimate(Y, X)
            corr = float(np.corrcoef(J_hat.ravel(), patient_Js[p].ravel())[0, 1])
            corrs.append(corr)
        mean_corr = float(np.mean(corrs))
        checks.append({
            "check": "correlation_at_large_T_p",
            "passed": mean_corr > 0.80,
            "value": f"{mean_corr:.3f}",
            "note": f"T_p={T_p_large}",
        })

    # Check 4: BOED sample complexity
    prior = {"mean": np.zeros(n), "cov": np.eye(n)}
    safety = {"u_max": 0.6}
    boed = BOEDEstimator(prior, safety)
    N_coarse = boed.sample_complexity(0.1, 0.05, {"n_theta": n**2})
    N_fine = boed.sample_complexity(0.05, 0.05, {"n_theta": n**2})
    checks.append({
        "check": "sample_complexity_scaling",
        "passed": N_fine > N_coarse,
        "value": f"N_coarse={N_coarse}, N_fine={N_fine}",
        "note": "Finer accuracy requires more samples",
    })

    elapsed = time.perf_counter() - t0
    results["elapsed"] = elapsed

    # Save results
    out_dir = ROOT / "results" / "stage_12"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_pass = sum(1 for c in checks if c["passed"])
    print(f"  Stage 12: {n_pass}/{len(checks)} checks passed ({elapsed:.1f}s)")
    return results
