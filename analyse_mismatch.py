"""
analyse_mismatch.py — Audit the empirical model-mismatch distribution
======================================================================
Loads per-seed partial JSON files from the highpower run, reconstructs
the evaluation models for the same seeds/configs, and reports the actual
multiplicative mismatch between nominal and true basin A matrices.

    delta_A_k = ||A_true_k - A_nominal_k||_2 / ||A_nominal_k||_2

for each basin k across all seeds.

Usage:
    python3 analyse_mismatch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Configuration (mirrors HIGHPOWER_CONFIG) ──────────────────────────────────
HIGHPOWER_CONFIG: dict[str, Any] = {
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
    "default_burden_budget": 28.0,
    "circadian_locked_controls": [5, 6],
    "R_brier_max": 0.05,
    "omega_min_factor": 0.005,
    "T_C_max": 50,
    "k_calib": 1.0,
    "sigma_dither": 0.08,
    "epsilon_control": 0.50,
    "missing_fraction_target": 0.516,
    "mode1_base_rate": 0.16,
    "observer_mode_accuracy_approx": 0.55,
    "w3_sweep_values": [0.05, 0.10, 0.20, 0.30, 0.50],
    "profile_name": "highpower",
    "seeds": [
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    ],
    "episodes_per_experiment": 30,
    "steps_per_episode": 256,
    "mc_rollouts": 150,
    "selected_trace_cap": 5,
}


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def percentile_stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def run_mismatch_audit() -> None:
    from hdr_validation.model.slds import make_evaluation_model

    cfg = HIGHPOWER_CONFIG
    out_dir = ROOT / "results" / "stage_04" / "highpower"

    # ── Load partial JSON files ──────────────────────────────────────────────
    partial_files = sorted(out_dir.glob("seed_*_partial.json"))
    if not partial_files:
        print("ERROR: No seed_*_partial.json files found in results/stage_04/highpower/")
        print("       Run highpower_runner.py first.")
        sys.exit(1)

    print(f"Found {len(partial_files)} seed partial files.")

    # ── Reconstruct models and compute mismatches ────────────────────────────
    # The highpower runner uses rng_sim = np.random.default_rng(s + 400)
    # for the sim_model (evaluation model used in closed-loop simulation).
    # We use the same seed to reconstruct the same model.

    # "Nominal" A is taken as the mean pooled A from all seeds —
    # this is what pooled_lqr_estimated implicitly assumes.
    # For per-basin analysis: nominal A_k = A_k from first seed (seed 101),
    # and true A_k = A_k from each particular seed.
    # More precisely: the mismatch is between the sim_model.basins[k].A
    # generated with rng(s+400) and the A that any fixed controller
    # (e.g., built from seed 101) would assume.

    # We report the intra-seed mismatch: how far each seed's A_k deviates
    # from the mean A_k across all seeds (the "nominal" average).

    K = cfg["K"]
    all_A_per_basin: list[list[np.ndarray]] = [[] for _ in range(K)]

    seeds_loaded = []
    for s in cfg["seeds"]:
        partial_path = out_dir / f"seed_{s:04d}_partial.json"
        if not partial_path.exists():
            print(f"  WARNING: {partial_path.name} not found, skipping seed {s}.")
            continue
        seeds_loaded.append(s)
        rng_sim = np.random.default_rng(s + 400)
        sim_model = make_evaluation_model(cfg, rng_sim)
        for k in range(K):
            all_A_per_basin[k].append(sim_model.basins[k].A.copy())

    print(f"Loaded and reconstructed models for {len(seeds_loaded)} seeds.")

    # ── Compute nominal A as mean across all seeds ────────────────────────────
    A_nominal = [np.mean(all_A_per_basin[k], axis=0) for k in range(K)]

    # ── Compute delta_A_k for each seed × basin ───────────────────────────────
    delta_all: list[float] = []
    delta_per_basin: list[list[float]] = [[] for _ in range(K)]

    for k in range(K):
        A_nom_k = A_nominal[k]
        norm_nom = float(np.linalg.norm(A_nom_k, ord=2))
        for A_true in all_A_per_basin[k]:
            delta = float(np.linalg.norm(A_true - A_nom_k, ord=2)) / (norm_nom + 1e-15)
            delta_per_basin[k].append(delta)
            delta_all.append(delta)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL MISMATCH AUDIT — delta_A_k = ||A_true - A_nom||_2 / ||A_nom||_2")
    print("=" * 60)

    overall_stats = percentile_stats(delta_all)
    print(f"\nOverall (all seeds × all basins, N={len(delta_all)}):")
    print(f"  mean={overall_stats['mean']:.4f}  std={overall_stats['std']:.4f}")
    print(f"  p50={overall_stats['p50']:.4f}  p90={overall_stats['p90']:.4f}  "
          f"p95={overall_stats['p95']:.4f}  p99={overall_stats['p99']:.4f}  "
          f"max={overall_stats['max']:.4f}")

    basin_stats = []
    for k in range(K):
        s = percentile_stats(delta_per_basin[k])
        basin_stats.append(s)
        label = f"Basin {k} (rho={cfg['rho_reference'][k]:.2f})"
        if k == 1:
            label += " [MALADAPTIVE]"
        print(f"\n{label} (N={len(delta_per_basin[k])}):")
        print(f"  mean={s['mean']:.4f}  std={s['std']:.4f}")
        print(f"  p50={s['p50']:.4f}  p90={s['p90']:.4f}  "
              f"p95={s['p95']:.4f}  p99={s['p99']:.4f}  max={s['max']:.4f}")

    current_bound = float(cfg["model_mismatch_bound"])
    basin1_p90 = basin_stats[1]["p90"]
    print(f"\ncurrent model_mismatch_bound = {current_bound:.2f}")
    print(f"Basin 1 p90 of delta_A = {basin1_p90:.4f}")
    if basin1_p90 < current_bound:
        print(f"  => Basin 1 p90 ({basin1_p90:.4f}) < bound ({current_bound:.2f}):")
        print(f"     The current bound may be over-conservative for basin 1.")
        print(f"     Author may consider reducing to {basin1_p90:.4f} (see note in JSON).")
    else:
        print(f"  => Basin 1 p90 ({basin1_p90:.4f}) >= bound ({current_bound:.2f}):")
        print(f"     The current bound appears appropriate.")

    print("=" * 60)

    # ── Write output JSON ─────────────────────────────────────────────────────
    output = {
        "author_action_required": True,
        "note": (
            "If p90 of delta_A_k for basin 1 is below 0.20, the author "
            "may consider reducing model_mismatch_bound to the p90 value "
            "in a subsequent commit after reviewing this report."
        ),
        "methodology": (
            "delta_A_k = ||A_true_k - A_nominal_k||_2 / ||A_nominal_k||_2 "
            "where A_nominal_k = mean(A_k) across all 20 seeds. "
            "Each seed reconstructed with rng(seed + 400) matching the highpower runner."
        ),
        "n_seeds_loaded": len(seeds_loaded),
        "seeds_loaded": seeds_loaded,
        "model_mismatch_bound_current": current_bound,
        "overall_stats": overall_stats,
        "basin_stats": {
            f"basin_{k}_rho_{cfg['rho_reference'][k]:.2f}": {
                "basin_index": k,
                "rho_reference": cfg["rho_reference"][k],
                "is_maladaptive": (k == 1),
                "n_seeds": len(delta_per_basin[k]),
                **basin_stats[k],
            }
            for k in range(K)
        },
        "basin_1_p90_vs_bound": {
            "basin_1_p90": basin1_p90,
            "current_bound": current_bound,
            "p90_below_bound": bool(basin1_p90 < current_bound),
        },
    }

    out_path = out_dir / "mismatch_audit.json"
    _atomic_write_json(out_path, output)
    print(f"\nWrote mismatch_audit.json to {out_path}")


if __name__ == "__main__":
    run_mismatch_audit()
