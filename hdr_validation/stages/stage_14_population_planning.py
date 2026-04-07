"""
Stage 14 — Population-Prior Treatment-Planning Benchmark
====================================================================
Validates Claim 31: Population-prior planning accuracy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def run_stage_14(
    n_patients: int = 20,
    n_arms: int = 3,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Stage 14: Population-prior treatment-planning benchmark."""
    from hdr_validation.identification.population_planning import PopulationPriorPlanner

    t0 = time.perf_counter()
    rng = np.random.default_rng(101)
    n = 4
    K = 2

    results: dict[str, Any] = {"checks": []}
    checks = results["checks"]

    # Generate B_k prior (effect matrices per basin)
    B_k_prior = [rng.normal(size=(n, 2)) * 0.2 for _ in range(K)]

    # Generate approved regimens
    regimens = [rng.normal(size=2) * 0.3 for _ in range(n_arms)]

    planner = PopulationPriorPlanner(B_k_prior, regimens)

    # Generate patients and true best assignments
    assignments = []
    true_best = []
    for p in range(n_patients):
        basin_probs = rng.dirichlet(np.ones(K))
        patient = {"basin_probs": basin_probs}

        # Oracle: find best regimen for this patient
        best_cost = float('inf')
        best_idx = 0
        for idx, reg in enumerate(regimens):
            cost = 0.0
            for k in range(K):
                effect = B_k_prior[k][:, :len(reg)] @ reg
                cost += basin_probs[k] * float(np.sum(effect**2))
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        true_best.append(best_idx)

        # Planner assignment
        selected = planner.plan(patient, H=10)
        # Find which regimen was selected
        best_match = 0
        best_dist = float('inf')
        for idx, reg in enumerate(regimens):
            dist = float(np.linalg.norm(selected - reg))
            if dist < best_dist:
                best_dist = dist
                best_match = idx
        assignments.append(best_match)

    accuracy = planner.accuracy(assignments, true_best)
    random_accuracy = 1.0 / n_arms

    # Check 1: Pop-prior accuracy >= 60%
    checks.append({
        "check": "pop_prior_accuracy_60pct",
        "passed": accuracy >= 0.60 or n_patients < 5,
        "value": f"{accuracy:.2f}",
        "note": f"threshold=0.60",
    })

    # Check 2: Pop-prior > naive (random = 1/n_arms)
    checks.append({
        "check": "pop_prior_better_than_random",
        "passed": accuracy > random_accuracy,
        "value": f"acc={accuracy:.2f} vs random={random_accuracy:.2f}",
        "note": "Population prior should beat random",
    })

    elapsed = time.perf_counter() - t0
    results["elapsed"] = elapsed

    from hdr_validation.provenance import get_provenance
    results["provenance"] = get_provenance()
    out_dir = ROOT / "results" / "stage_14"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    n_pass = sum(1 for c in checks if c["passed"])
    print(f"  Stage 14: {n_pass}/{len(checks)} checks passed ({elapsed:.1f}s)")
    return results
