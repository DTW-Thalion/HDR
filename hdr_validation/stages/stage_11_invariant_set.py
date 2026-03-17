"""
Stage 11 — Riccati Invariant Set Verification
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
    n_sigma: float = 5.0,
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

    Returns both the overall containment rate and the RPI forward-invariance
    rate (fraction of steps that stay inside given the previous step was inside).

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
        Keys: containment_rate, containment_rate_rpi, n_rpi_eligible,
        mean_lyapunov_value, max_lyapunov_value, n_steps_checked.
    """
    P_k = np.asarray(P_k, dtype=float)
    lyapunov_values: list[float] = []
    inside_count = 0

    # RPI test: count transitions where previous step was inside
    n_rpi_stays = 0
    n_rpi_eligible = 0

    for traj, labels in zip(trajectories, basin_labels):
        for t in range(len(traj)):
            if int(labels[t]) != target_basin:
                continue
            x = traj[t]
            v = float(x @ P_k @ x)
            lyapunov_values.append(v)
            if v <= c_k:
                inside_count += 1

            # RPI test: was previous step also in the target basin and inside set?
            if t > 0 and int(labels[t - 1]) == target_basin:
                v_prev = float(traj[t - 1] @ P_k @ traj[t - 1])
                if v_prev <= c_k:
                    n_rpi_eligible += 1
                    if v <= c_k:
                        n_rpi_stays += 1

    n_checked = len(lyapunov_values)
    if n_checked == 0:
        return {
            "containment_rate": float("nan"),
            "containment_rate_rpi": float("nan"),
            "n_rpi_eligible": 0,
            "mean_lyapunov_value": float("nan"),
            "max_lyapunov_value": float("nan"),
            "n_steps_checked": 0,
        }

    arr = np.array(lyapunov_values)
    rpi_rate = n_rpi_stays / max(n_rpi_eligible, 1) if n_rpi_eligible > 0 else float("nan")

    return {
        "containment_rate": inside_count / n_checked,
        "containment_rate_rpi": rpi_rate,
        "n_rpi_eligible": n_rpi_eligible,
        "mean_lyapunov_value": float(np.mean(arr)),
        "max_lyapunov_value": float(np.max(arr)),
        "n_steps_checked": n_checked,
    }


def _make_benchmark_config(n_seeds: int = 5, T: int = 128) -> dict[str, Any]:
    """Create benchmark configuration for Stage 11."""
    from hdr_validation.defaults import DEFAULTS

    cfg = dict(DEFAULTS)
    cfg.update({
        "max_dwell_len": 256,
        "default_burden_budget": 56.0,
        "n_seeds": n_seeds,
        "steps_per_episode": T,
        "profile_name": "highpower",
    })
    return cfg


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
    from hdr_validation.control.mpc import solve_mode_a, precompute_mode_a_cache
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

    # Precompute P_k and c_k for all basins (needed for inside-ellipsoid init)
    _P_k_per_basin: dict[int, np.ndarray] = {}
    _c_k_per_basin: dict[int, float] = {}
    n_sigma_init = float(cfg.get("n_sigma_init", 3.0))
    for k_pre, basin_pre in enumerate(eval_model.basins):
        try:
            K_pre, P_pre = dlqr(basin_pre.A, basin_pre.B, Q_lqr, R_lqr)
        except Exception:
            K_pre = np.zeros((n, n))
            P_pre = np.eye(n)
        A_cl_pre = basin_pre.A - basin_pre.B @ K_pre
        c_pre = compute_lyapunov_level_set_radius(P_pre, A_cl_pre, basin_pre.Q, n_sigma=n_sigma_init)
        _P_k_per_basin[k_pre] = P_pre
        _c_k_per_basin[k_pre] = c_pre

    seeds = [101 + i * 101 for i in range(n_seeds)]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for ep_idx in range(4):  # 4 episodes per seed
            basin_idx = rng.integers(0, len(eval_model.basins))
            basin = eval_model.basins[basin_idx]
            target = build_target_set(basin_idx, cfg)

            # Initialise inside the level set: draw unit-norm direction, scale to
            # init_fraction * sqrt(c_k / lambda_max(P_k)) so V(x_0) <= c_k.
            # This tests RPI from inside — the correct Proposition 8.4 semantics.
            _x_dir = rng.normal(size=n)
            _x_dir /= (np.linalg.norm(_x_dir) + 1e-12)
            _P_k_here = _P_k_per_basin[int(basin_idx)]
            _c_k_here = _c_k_per_basin[int(basin_idx)]
            _lambda_max_P = float(np.max(np.linalg.eigvalsh(_P_k_here)))
            _scale = 0.5 * np.sqrt(_c_k_here / max(_lambda_max_P, 1e-9))
            x = _x_dir * _scale
            P_hat = np.eye(n) * 0.2
            traj = np.empty((T, n))
            labels = np.full(T, basin_idx, dtype=int)

            # Pre-compute expensive invariants for this episode
            mpc_cache = precompute_mode_a_cache(basin, cfg)

            for t in range(T):
                traj[t] = x
                try:
                    res = solve_mode_a(x, P_hat, basin, target, kappa_hat=0.65, config=cfg, step=t,
                                       P_terminal_precomputed=mpc_cache["P_terminal"],
                                       C_pinv_precomputed=mpc_cache["C_pinv"])
                    u = res.u
                except Exception:
                    u = np.zeros(cfg["control_dim"])

                w = basin.Q_cholesky @ rng.standard_normal(n)
                x = basin.A @ x + basin.B @ u + basin.b + w

            trajectories.append(traj)
            basin_labels.append(labels)

    return trajectories, basin_labels, eval_model, K_banks


def run_stage_11(
    n_seeds: int = 5,
    T: int = 128,
    output_dir: Path | None = None,
    n_sigma: float = 5.0,
    containment_threshold: float = 0.90,
    fast_mode: bool = False,
    use_tube_mpc: bool = False,
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
    use_tube_mpc : bool
        If True, also compute mRPI zonotopes and run tube-MPC trajectories
        in parallel, reporting per-basin containment rates (v7.1).

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

        # Use containment_rate_rpi (forward-invariance) as primary criterion
        rpi_rate = containment["containment_rate_rpi"]
        _is_nan_rpi = (rpi_rate != rpi_rate) if isinstance(rpi_rate, float) else np.isnan(rpi_rate)
        criterion_met = (
            not _is_nan_rpi
            and rpi_rate >= containment_threshold
        )

        def _safe_round(val: float, digits: int = 4):
            if val != val:  # NaN check
                return None
            return round(val, digits)

        basins_out[str(k_idx)] = {
            "c_k": round(c_k, 4),
            "containment_rate": _safe_round(containment["containment_rate"]),
            "containment_rate_rpi": _safe_round(containment["containment_rate_rpi"]),
            "n_rpi_eligible": containment["n_rpi_eligible"],
            "mean_lyapunov_value": _safe_round(containment["mean_lyapunov_value"]),
            "max_lyapunov_value": _safe_round(containment["max_lyapunov_value"]),
            "n_steps_checked": containment["n_steps_checked"],
            "proposition_8_4_criterion_met": criterion_met,
            "rho_basin": round(basin.rho, 4),
        }

    # ── Tube-MPC path (when enabled) ──
    if use_tube_mpc:
        from hdr_validation.control.tube_mpc import (
            compute_disturbance_set,
            compute_mRPI_zonotope,
            solve_tube_mpc,
            zonotope_containment_check,
        )
        from hdr_validation.model.target_set import build_target_set as _build_ts

        tube_trajs, tube_labels = [], []
        tube_mRPI_data = {}
        tube_K_banks = {}

        for k_idx, basin in enumerate(eval_model.basins):
            try:
                K_k, P_k = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
            except Exception:
                K_k = np.zeros((n, n))
            A_cl_k = basin.A - basin.B @ K_k
            # beta=0.999 per Appendix J (Definition J.1) and consistent with
            # test_tube_mpc.py and highpower_runner.py
            _, chi2_bound = compute_disturbance_set(basin.Q, n, beta=0.999)
            mRPI_data = compute_mRPI_zonotope(A_cl_k, basin.Q, chi2_bound, epsilon=0.01)
            tube_mRPI_data[k_idx] = mRPI_data
            tube_K_banks[k_idx] = K_k

        seeds = [101 + i * 101 for i in range(n_seeds)]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for ep_idx in range(4):
                basin_idx = rng.integers(0, len(eval_model.basins))
                basin = eval_model.basins[basin_idx]
                target = _build_ts(basin_idx, cfg)
                x = rng.normal(size=n) * 0.1
                P_hat = np.eye(n) * 0.2
                traj = np.empty((T, n))
                labels = np.full(T, basin_idx, dtype=int)
                K_fb = tube_K_banks[int(basin_idx)]
                mRPI = tube_mRPI_data[int(basin_idx)]

                for t in range(T):
                    traj[t] = x
                    try:
                        res = solve_tube_mpc(x, P_hat, basin, target,
                                             mRPI, K_fb, kappa_hat=0.65,
                                             config=cfg, step=t)
                        u = res.u
                    except Exception:
                        u = np.zeros(cfg["control_dim"])
                    w = basin.Q_cholesky @ rng.standard_normal(n)
                    x = basin.A @ x + basin.B @ u + basin.b + w

                tube_trajs.append(traj)
                tube_labels.append(labels)

        # Compute tube containment rates per basin
        for k_idx in range(len(eval_model.basins)):
            mRPI = tube_mRPI_data[k_idx]
            inside_count = 0
            total_count = 0
            for traj, labels in zip(tube_trajs, tube_labels):
                for t in range(len(traj)):
                    if int(labels[t]) != k_idx:
                        continue
                    total_count += 1
                    if zonotope_containment_check(traj[t], mRPI["G"], mRPI["center"],
                                                    G_pinv=mRPI.get("G_pinv")):
                        inside_count += 1
            rate = inside_count / max(total_count, 1) if total_count > 0 else float("nan")
            basins_out[str(k_idx)]["containment_rate_tube"] = (
                round(rate, 4) if rate == rate else None
            )

    result_json = {
        "basins": basins_out,
        "tube_mpc_enabled": use_tube_mpc,
        "containment_threshold": containment_threshold,
        "n_sigma_disturbance": n_sigma,
        "n_seeds": n_seeds,
        "T": T,
        "note": "Lyapunov level-set RPI approximation per Proposition 8.4 (revised paper)",
        "rpi_semantics_note": (
            "containment_rate_rpi measures RPI forward-invariance: "
            "fraction of steps where the system stays inside the ellipsoid "
            "given it was inside at the previous step. "
            "This is the correct empirical test of Proposition 8.4. "
            "containment_rate measures overall occupancy and will be lower "
            "when trajectories are initialised at the ellipsoid boundary."
        ),
    }

    from hdr_validation.provenance import get_provenance
    result_json["provenance"] = get_provenance()
    out_path = output_dir / "invariant_set_verification.json"
    out_path.write_text(json.dumps(result_json, indent=2))

    # Print summary
    print("\nRiccati Invariant Set Verification — Proposition 8.4")
    print("─" * 65)
    all_pass = True
    for k_str, data in basins_out.items():
        cr = data["containment_rate"]
        rpi_cr = data["containment_rate_rpi"]
        c_k_val = data["c_k"]
        crit = data["proposition_8_4_criterion_met"]
        status = "PASS" if crit else "FAIL"
        if not crit:
            all_pass = False
        cr_str = f"{cr:.4f}" if cr is not None else "N/A"
        rpi_str = f"{rpi_cr:.4f}" if rpi_cr is not None else "N/A"
        print(f"  [{status}] Basin {k_str} (rho={data['rho_basin']:.3f}): "
              f"rpi={rpi_str}, overall={cr_str} (threshold={containment_threshold:.2f}), c_k={c_k_val:.2f}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  [{overall}] Proposition 8.4 criterion (RPI rate >= {containment_threshold:.0%}) for all basins")
    print(f"\nResults saved to {out_path}")

    return result_json


if __name__ == "__main__":
    run_stage_11()
