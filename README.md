from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from .target_set import TargetSet


def tau_tilde(x: np.ndarray, target: TargetSet, Q: np.ndarray, rho: float, method: str = "box") -> float:
    dist2 = target.dist2(x, Q=Q, method=method)
    denom = max(1.0 - rho**2, 1e-6)
    return float(dist2 / denom)


def lyapunov_cost(A: np.ndarray, Q: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    P = solve_discrete_lyapunov(A, Q)
    x = np.asarray(x, dtype=float)
    return float(x.T @ P @ x), P


def dare_terminal_cost(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def tau_sandwich(A: np.ndarray, Q: np.ndarray, x: np.ndarray, target: TargetSet, rho: float) -> dict[str, float]:
    proj = target.project_box(x)
    xbar = np.asarray(x) - proj
    tau_h = tau_tilde(x, target, Q, rho, method="box")
    tau_L, P = lyapunov_cost(A, Q, xbar)
    eigvals = np.linalg.eigvalsh(np.sqrt(Q) @ P @ np.linalg.pinv(np.sqrt(Q)))
    eigvals = np.real(eigvals)
    return {
        "tau_tilde": float(tau_h),
        "tau_L": float(tau_L),
        "lower_coeff": float(np.min(eigvals)),
        "upper_coeff": float(np.max(eigvals)),
    }

## Reproducing Benchmark A (high-power run)

The headline Benchmark A result (N_mal = 123, 20 seeds × 20 episodes)
is produced by a standalone script that is NOT part of run_all.py:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    python highpower_runner.py

Outputs are written to results/stage_04/highpower/:
  highpower_summary.json   — machine-readable metrics and per-seed gains
  highpower_table.txt      — human-readable results table
  manuscript_language.txt  — recommended manuscript wording

Expected values (fixed seeds 101–2020):
  N_maladaptive : 123
  Mean gain     : +0.028  (95 % CI [+0.021, +0.036])
  Win rate      : 0.772
  Safety delta  : +0.0004

## Environment setup (required on multi-core Linux)

Before running any script or pytest, pin BLAS threads to prevent
non-determinism and hangs on shared compute nodes:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

Add these lines to your shell profile or CI environment.
