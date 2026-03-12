"""
derive_criterion.py — Derive a theory-grounded performance criterion
=====================================================================
Uses Proposition 10.4 (ISS bound) to compute the theoretical minimum
cost advantage of the basin-aware HDR controller over the pooled LQR
baseline for basin k=1 (maladaptive, rho=0.96).

    ISS_bound = (delta_A + delta_B * ||K_pooled||_2) / alpha_k

where:
    delta_A = ||A_1 - A_pooled||_2
    delta_B = ||B_1 - B_pooled||_2
    alpha_k = Lyapunov decrease rate from DARE for basin 1
    K_pooled = LQR gain for pooled A, B

Usage:
    python3 derive_criterion.py
"""
from __future__ import annotations

import json
import math
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
    "model_mismatch_bound": 0.347,
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


def run_derive_criterion() -> None:
    from hdr_validation.model.slds import make_evaluation_model, pooled_basin
    from hdr_validation.control.lqr import dlqr, compute_alpha_from_dare

    cfg = HIGHPOWER_CONFIG
    out_dir = ROOT / "results" / "stage_04" / "highpower"

    # ── Use the first seed's model as the reference (same seed as pooled LQR build) ──
    # Use seed 101 (first seed) to get nominal basin parameters
    rng = np.random.default_rng(cfg["seeds"][0] + 400)
    eval_model = make_evaluation_model(cfg, rng)

    n = cfg["state_dim"]
    m_u = cfg["control_dim"]
    lambda_u = float(cfg["lambda_u"])

    Q_lqr = np.eye(n)
    R_lqr = np.eye(m_u) * lambda_u

    # ── Pooled model (A_pooled, B_pooled) ─────────────────────────────────────
    p_basin = pooled_basin(eval_model)
    A_pooled = p_basin.A
    B_pooled = p_basin.B

    # ── Basin 1 (maladaptive, rho=0.96) ─────────────────────────────────────
    basin1 = eval_model.basins[1]
    A_1 = basin1.A
    B_1 = basin1.B
    Q_1 = basin1.Q  # basin-specific process noise cov (used as Q in cost)

    # Use standard LQR cost (same Q, R as runners use)
    Q_cost = Q_lqr.copy()  # identity — standard stage cost
    R_cost = R_lqr.copy()

    # (i) K_pooled: LQR gain for pooled A, B
    try:
        K_pooled, _ = dlqr(A_pooled, B_pooled, Q_cost, R_cost)
    except Exception as e:
        print(f"WARNING: dlqr failed for pooled basin: {e}")
        K_pooled = np.zeros((m_u, n))

    # (ii) K_optimal: LQR gain for basin-1 A, B
    try:
        K_optimal, _ = dlqr(A_1, B_1, Q_cost, R_cost)
    except Exception as e:
        print(f"WARNING: dlqr failed for basin 1: {e}")
        K_optimal = np.zeros((m_u, n))

    # (iii) delta_A and delta_B
    delta_A = float(np.linalg.norm(A_1 - A_pooled, ord=2))
    delta_B = float(np.linalg.norm(B_1 - B_pooled, ord=2))
    norm_K_pooled = float(np.linalg.norm(K_pooled, ord=2))
    norm_K_optimal = float(np.linalg.norm(K_optimal, ord=2))

    # (iv) alpha_k via compute_alpha_from_dare
    try:
        alpha_k = compute_alpha_from_dare(A_1, B_1, Q_cost, R_cost)
    except Exception as e:
        print(f"WARNING: compute_alpha_from_dare failed: {e}, using fallback 0.95")
        alpha_k = 0.95

    # (v) ISS-predicted cost-advantage lower bound
    ISS_bound = (delta_A + delta_B * norm_K_pooled) / alpha_k

    # (vi) Convert to fractional gain using mean pooled_lqr_estimated cost for basin-1 episodes
    # Load the highpower summary to get basin-1 mean cost from pooled_lqr_estimated
    summary_path = out_dir / "highpower_summary.json"
    mean_pooled_cost_basin1 = None
    gains_mal_list = None

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            # The summary has gains_maladaptive_all and hdr_vs_pe_maladaptive_mean
            # We need the mean pooled cost for basin-1 episodes.
            # The gains are fractional: (cost_pe - cost_hdr) / cost_pe
            # We don't have raw costs in the summary, only gains.
            # So we estimate ISS_bound_fractional = ISS_bound / C_ref
            # where C_ref = ISS_bound / mean_gain * mean_gain
            # Use the mean gain from the high-power run to infer C_ref proxy:
            # ISS_bound_fractional ~= ISS_bound / mean_pooled_cost
            # We need a proxy for mean_pooled_cost.
            # Best approach: load per-seed partial files to get raw costs.
            gains_mal_list = summary.get("gains_maladaptive_all", None)
        except Exception:
            pass

    # Load raw costs from partial files to compute mean pooled cost for basin 1
    partial_files = sorted(out_dir.glob("seed_*_partial.json"))
    all_pe_costs_basin1 = []
    all_hdr_costs_basin1 = []

    for pf in partial_files:
        try:
            d = json.loads(pf.read_text())
            ep_costs = d.get("ep_costs", {})
            ep_basins = d.get("ep_basins", [])
            pe_costs = ep_costs.get("pooled_lqr_estimated", [])
            hdr_costs = ep_costs.get("hdr_main", [])
            for i, b in enumerate(ep_basins):
                if b == 1 and i < len(pe_costs):
                    all_pe_costs_basin1.append(float(pe_costs[i]))
                if b == 1 and i < len(hdr_costs):
                    all_hdr_costs_basin1.append(float(hdr_costs[i]))
        except Exception:
            pass

    if all_pe_costs_basin1:
        mean_pooled_cost_basin1 = float(np.mean(all_pe_costs_basin1))
        ISS_bound_fractional = ISS_bound / mean_pooled_cost_basin1
    else:
        # Fallback: use a reference episode cost estimate
        # T=256, mean ||x||^2 ~ rho^2/(1-rho^2) * sigma_w^2 * T
        # For basin 1: rho=0.96, sigma_w=sqrt(0.07), T=256
        rho1 = 0.96
        sigma_w_sq = 0.07  # basin1.Q diagonal
        T = cfg["steps_per_episode"]
        # Lyapunov-based steady-state variance proxy
        steady_state_var = sigma_w_sq / (1.0 - rho1**2)
        mean_pooled_cost_basin1 = T * (n * steady_state_var + lambda_u * n * 0.01)
        ISS_bound_fractional = ISS_bound / mean_pooled_cost_basin1
        print(f"WARNING: No partial files found; using estimated reference cost {mean_pooled_cost_basin1:.2f}")

    ISS_bound_pct = ISS_bound_fractional * 100.0

    # Suggested theory-grounded criterion = floor(ISS_bound_fractional * 0.5 * 100) / 100
    theory_criterion = math.floor(ISS_bound_fractional * 0.5 * 100) / 100

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  THEORY-GROUNDED CRITERION (Proposition 10.4 ISS Bound)")
    print("=" * 62)
    print(f"\nBasin 1 (maladaptive, rho={cfg['rho_reference'][1]:.2f}):")
    print(f"  ||K_pooled||_2   = {norm_K_pooled:.4f}")
    print(f"  ||K_optimal||_2  = {norm_K_optimal:.4f}")
    print(f"  delta_A          = {delta_A:.4f}   (||A_1 - A_pooled||_2)")
    print(f"  delta_B          = {delta_B:.4f}   (||B_1 - B_pooled||_2)")
    print(f"  alpha_k          = {alpha_k:.4f}   (Lyapunov decrease rate)")
    print(f"\nISS bound (absolute):     {ISS_bound:.4f}")
    print(f"Mean pooled cost (basin1): {mean_pooled_cost_basin1:.4f}")
    print(f"ISS bound (fractional):   {ISS_bound_fractional:.4f}  ({ISS_bound_pct:.2f}%)")
    print(f"\nTheory-grounded criterion  = floor(ISS_frac * 0.5 * 100) / 100 = {theory_criterion:.4f}")
    print(f"Pre-registered criterion   = +0.030")
    if theory_criterion < 0.030:
        print(f"  => Theory criterion ({theory_criterion:.4f}) < pre-registered (+0.030).")
        print(f"     See note in theory_criterion.json for author action.")
    else:
        print(f"  => Theory criterion ({theory_criterion:.4f}) >= pre-registered (+0.030).")
        print(f"     Pre-registered criterion appears appropriate.")
    print("=" * 62)

    # ── Write output JSON ─────────────────────────────────────────────────────
    output = {
        "author_action_required": True,
        "note": (
            "If the theory-grounded criterion is below +0.030, the author "
            "should consider whether the pre-registered +0.030 criterion "
            "was appropriately calibrated. This is an author decision, "
            "not an automated update."
        ),
        "methodology": (
            "ISS_bound = (delta_A + delta_B * ||K_pooled||_2) / alpha_k "
            "where delta_A = ||A_1 - A_pooled||_2, delta_B = ||B_1 - B_pooled||_2, "
            "alpha_k = Lyapunov decrease rate from DARE (Prop 9.1), "
            "K_pooled = LQR gain for pooled A, B. "
            "Reference model: seed 101 + 400."
        ),
        "basin_index": 1,
        "rho_reference": float(cfg["rho_reference"][1]),
        "delta_A": delta_A,
        "delta_B": delta_B,
        "alpha_k": alpha_k,
        "norm_K_pooled": norm_K_pooled,
        "norm_K_optimal": norm_K_optimal,
        "ISS_bound_absolute": ISS_bound,
        "mean_pooled_cost_basin1": mean_pooled_cost_basin1,
        "n_basin1_episodes_in_average": len(all_pe_costs_basin1),
        "ISS_bound_fractional": ISS_bound_fractional,
        "ISS_bound_pct": ISS_bound_pct,
        "theory_criterion": theory_criterion,
        "theory_criterion_formula": "floor(ISS_bound_fractional * 0.5 * 100) / 100",
        "preregistered_criterion": 0.030,
        "theory_criterion_below_preregistered": bool(theory_criterion < 0.030),
    }

    out_path = out_dir / "theory_criterion.json"
    _atomic_write_json(out_path, output)
    print(f"\nWrote theory_criterion.json to {out_path}")


if __name__ == "__main__":
    run_derive_criterion()
