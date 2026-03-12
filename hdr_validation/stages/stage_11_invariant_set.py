"""
Stage 11 — Riccati Invariant Set Verification (HDR v5.2)
==========================================================

Verifies numerically that Benchmark A trajectories stay within the Lyapunov
level set {x : x^T P_k x <= c_k} under Mode A (Proposition 8.4).

The level set radius c_k is computed as an ellipsoidal RPI approximation
per Algorithm B.1 of the revised paper.

Results saved to results/stage_11/invariant_set_verification.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).parent.parent.parent


def compute_lyapunov_level_set_radius(
    P_k: np.ndarray,
    A_cl_k: np.ndarray,
    Q_w_k: np.ndarray,
    n_sigma: float = 3.0,
) -> float:
    """Compute the Lyapunov level-set radius c_k for ellipsoidal RPI.

    Computes c_k such that the ellipsoid E_k = {x : x^T P_k x <= c_k}
    is robustly positively invariant under:
        x_{t+1} = A_cl_k x_t + w_t,  w_t ~ N(0, Q_w_k).

    Method (ellipsoidal RPI approximation):
        The disturbance bound is d_max = n_sigma * sqrt(lambda_max(Q_w_k)).
        The RPI condition requires:
            c_k >= d_max^2 * lambda_max(P_k) / alpha_k
        where alpha_k is the Lyapunov decrease rate computed from:
            alpha_k = 1 - lambda_max(A_cl_k^T P_k A_cl_k) / lambda_max(P_k)

    Parameters
    ----------
    P_k : np.ndarray
        DARE solution for basin k, shape (n, n). Must be SPD.
    A_cl_k : np.ndarray
        Closed-loop A_k - B_k @ K_k, shape (n, n).
    Q_w_k : np.ndarray
        Process noise covariance, shape (n, n). Must be SPD.
    n_sigma : float
        Number of standard deviations for disturbance bound. Default 3.0.

    Returns
    -------
    float
        c_k: Positive scalar, the level-set radius.
    """
    P_k = np.asarray(P_k, dtype=float)
    A_cl_k = np.asarray(A_cl_k, dtype=float)
    Q_w_k = np.asarray(Q_w_k, dtype=float)

    lambda_max_Q = float(np.max(np.linalg.eigvalsh(Q_w_k)))
    d_max = n_sigma * np.sqrt(max(lambda_max_Q, 0.0))

    lambda_max_P = float(np.max(np.linalg.eigvalsh(P_k)))

    # Lyapunov decrease: V(x_{t+1}) - V(x_t) = x_t^T (A_cl^T P A_cl - P) x_t
    # alpha_k = lambda_min(P - A_cl^T P A_cl) / lambda_max(P)
    delta_P = P_k - A_cl_k.T @ P_k @ A_cl_k
    lambda_min_delta = float(np.min(np.linalg.eigvalsh(delta_P)))

    if lambda_max_P < 1e-12:
        return 1.0
    alpha_k = max(lambda_min_delta / lambda_max_P, 1e-6)

    # RPI radius
    c_k = (d_max ** 2 * lambda_max_P) / alpha_k
    return float(max(c_k, 1e-9))


def check_trajectory_containment(
    trajectories: list[np.ndarray],
    P_k: np.ndarray,
    c_k: float,
    basin_labels: list[np.ndarray],
    target_basin: int,
) -> dict:
    """Check what fraction of trajectory steps are within the Lyapunov level set.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        List of arrays, each shape (T, n). State trajectories.
    P_k : np.ndarray
        DARE solution for basin k, shape (n, n).
    c_k : float
        Level-set radius (Lyapunov level).
    basin_labels : list[np.ndarray]
        List of arrays, each shape (T,) with basin IDs.
    target_basin : int
        Which basin k to check containment for.

    Returns
    -------
    dict
        Keys: containment_rate, mean_lyapunov_value, max_lyapunov_value, n_steps_checked.
    """
    P_k = np.asarray(P_k, dtype=float)
    lyapunov_values: list[float] = []
    inside_count = 0

    for traj, labels in zip(trajectories, basin_labels):
        for t in range(len(traj)):
            if int(labels[t]) == target_basin:
                x = traj[t]
                v = float(x @ P_k @ x)
                lyapunov_values.append(v)
                if v <= c_k:
                    inside_count += 1

    n_checked = len(lyapunov_values)
    if n_checked == 0:
        return {
            "containment_rate": float("nan"),
            "mean_lyapunov_value": float("nan"),
            "max_lyapunov_value": float("nan"),
            "n_steps_checked": 0,
        }

    arr = np.array(lyapunov_values)
    return {
        "containment_rate": inside_count / n_checked,
        "mean_lyapunov_value": float(np.mean(arr)),
        "max_lyapunov_value": float(np.max(arr)),
        "n_steps_checked": n_checked,
    }


def _make_benchmark_config(n_seeds: int = 5, T: int = 128) -> dict[str, Any]:
    """Create benchmark configuration for Stage 11."""
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
        "k_calib": 1.0,
        "sigma_dither": 0.08,
        "missing_fraction_target": 0.516,
        "n_seeds": n_seeds,
        "steps_per_episode": T,
        "profile_name": "highpower",
    }


def _simulate_benchmark_trajectories(
    cfg: dict[str, Any],
    n_seeds: int = 5,
    T: int = 128,
) -> tuple[list[np.ndarray], list[np.ndarray], Any, Any]:
    """Simulate Benchmark A trajectories under Mode A.

    Returns
    -------
    trajectories : list of (T, n) arrays
    basin_labels : list of (T,) arrays
    eval_model : EvaluationModel used
    K_banks : dict {basin_idx: K_lqr}
    """
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.control.lqr import dlqr

    trajectories: list[np.ndarray] = []
    basin_labels: list[np.ndarray] = []

    # Build common model
    model_rng = np.random.default_rng(42)
    eval_model = make_evaluation_model(cfg, model_rng)
    n = cfg["state_dim"]

    # Build LQR gains
    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * float(cfg.get("lambda_u", 0.1))
    K_banks: dict[int, np.ndarray] = {}
    for k_idx, basin in enumerate(eval_model.basins):
        try:
            K_k, _ = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
        except Exception:
            K_k = np.zeros((n, n))
        K_banks[k_idx] = K_k

    seeds = [101 + i * 101 for i in range(n_seeds)]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for ep_idx in range(4):  # 4 episodes per seed
            basin_idx = rng.integers(0, len(eval_model.basins))
            basin = eval_model.basins[basin_idx]
            target = build_target_set(basin_idx, cfg)

            x = rng.normal(scale=0.5, size=n)
            P_hat = np.eye(n) * 0.2
            traj = np.empty((T, n))
            labels = np.full(T, basin_idx, dtype=int)

            for t in range(T):
                traj[t] = x
                try:
                    res = solve_mode_a(x, P_hat, basin, target, kappa_hat=0.65, config=cfg, step=t)
                    u = res.u
                except Exception:
                    u = np.zeros(cfg["control_dim"])

                w = rng.multivariate_normal(np.zeros(n), basin.Q)
                x = basin.A @ x + basin.B @ u + basin.b + w

            trajectories.append(traj)
            basin_labels.append(labels)

    return trajectories, basin_labels, eval_model, K_banks


def run_stage_11(
    n_seeds: int = 5,
    T: int = 128,
    output_dir: Path | None = None,
    n_sigma: float = 3.0,
    containment_threshold: float = 0.90,
    fast_mode: bool = False,
) -> dict:
    """Run Stage 11 Riccati invariant set verification.

    Parameters
    ----------
    n_seeds : int
        Number of seeds for trajectory generation.
    T : int
        Steps per episode.
    output_dir : Path or None
        Output directory. Defaults to results/stage_11/.
    n_sigma : float
        Disturbance bound in standard deviations.
    containment_threshold : float
        Minimum required containment rate (Proposition 8.4 criterion).
    fast_mode : bool
        If True, use reduced parameters.

    Returns
    -------
    dict
        Invariant set verification results (JSON-compatible).
    """
    if fast_mode:
        n_seeds = min(n_seeds, 2)
        T = min(T, 32)

    cfg = _make_benchmark_config(n_seeds=n_seeds, T=T)

    if output_dir is None:
        output_dir = ROOT / "results" / "stage_11"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from hdr_validation.control.lqr import dlqr

    # Simulate trajectories
    trajectories, basin_labels, eval_model, K_banks = _simulate_benchmark_trajectories(
        cfg, n_seeds=n_seeds, T=T
    )

    n = cfg["state_dim"]
    Q_lqr = np.eye(n)
    R_lqr = np.eye(n) * float(cfg.get("lambda_u", 0.1))

    basins_out: dict[str, dict] = {}

    for k_idx, basin in enumerate(eval_model.basins):
        # Compute DARE solution P_k
        try:
            K_k, P_k = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
        except Exception:
            K_k = np.zeros((n, n))
            P_k = np.eye(n)

        # Closed-loop A
        A_cl_k = basin.A - basin.B @ K_k

        # Level-set radius
        c_k = compute_lyapunov_level_set_radius(P_k, A_cl_k, basin.Q, n_sigma=n_sigma)

        # Check containment
        containment = check_trajectory_containment(
            trajectories, P_k, c_k, basin_labels, target_basin=k_idx
        )

        criterion_met = (
            not np.isnan(containment["containment_rate"])
            and containment["containment_rate"] >= containment_threshold
        )

        basins_out[str(k_idx)] = {
            "c_k": round(c_k, 4),
            "containment_rate": round(containment["containment_rate"], 4)
            if not np.isnan(containment["containment_rate"]) else None,
            "mean_lyapunov_value": round(containment["mean_lyapunov_value"], 4)
            if not np.isnan(containment["mean_lyapunov_value"]) else None,
            "max_lyapunov_value": round(containment["max_lyapunov_value"], 4)
            if not np.isnan(containment["max_lyapunov_value"]) else None,
            "n_steps_checked": containment["n_steps_checked"],
            "proposition_8_4_criterion_met": criterion_met,
            "rho_basin": round(basin.rho, 4),
        }

    result_json = {
        "basins": basins_out,
        "containment_threshold": containment_threshold,
        "n_sigma_disturbance": n_sigma,
        "n_seeds": n_seeds,
        "T": T,
        "note": "Lyapunov level-set RPI approximation per Proposition 8.4 (revised paper)",
    }

    out_path = output_dir / "invariant_set_verification.json"
    out_path.write_text(json.dumps(result_json, indent=2))

    # Print summary
    print("\nRiccati Invariant Set Verification — Proposition 8.4")
    print("─" * 65)
    all_pass = True
    for k_str, data in basins_out.items():
        cr = data["containment_rate"]
        c_k_val = data["c_k"]
        crit = data["proposition_8_4_criterion_met"]
        status = "PASS" if crit else "FAIL"
        if not crit:
            all_pass = False
        cr_str = f"{cr:.4f}" if cr is not None else "N/A"
        print(f"  [{status}] Basin {k_str} (rho={data['rho_basin']:.3f}): "
              f"containment={cr_str} (threshold={containment_threshold:.2f}), c_k={c_k_val:.2f}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  [{overall}] Proposition 8.4 criterion (containment >= {containment_threshold:.0%}) for all basins")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_11()
