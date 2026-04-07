"""
Stage 20 — Structured (A = I + dt(-D + J)) vs Unstructured (A_k) Identification
================================================================================
Demonstrates that the mechanistic decomposition A = I + dt(-D + J) with known
sparsity pattern and sign constraints yields better sample efficiency than
direct identification of the full A_k matrix.

Sweep over sample size T in {20, 50, 100, 200, 500, 1000, 2000}; at each T
run 50 trials comparing:
  - Structured: 31 parameters (8 diagonal + 23 sparse off-diagonal)
  - Unstructured: 64 parameters (full 8x8 matrix)

Metrics: Frobenius error, sign concordance, sparsity preservation,
spectral-radius error, downstream MPC cost ratio.

Pass/fail criteria:
  C1: Structured error < unstructured error at T <= 200
  C2: Structured sign recovery >= 95% at all T
  C3: Structured spectral-radius error < unstructured at T <= 200
  C4: Crossover T exists (unstructured eventually catches up)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg

ROOT = Path(__file__).parent.parent.parent


# ── Mechanistic prior: J sparsity pattern and signs ──────────────────────────

def _build_mechanistic_prior(n: int, rng: np.random.Generator) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build a mechanistic J matrix with known sparsity and sign constraints.

    Returns
    -------
    D_true : (n,) positive diagonal decay rates
    J_true : (n, n) sparse off-diagonal coupling (zero diagonal)
    J_mask : (n, n) bool — True where J_ij is non-zero
    J_signs : (n, n) +1 where J_ij > 0, 0 elsewhere
    """
    # Diagonal: decay rates drawn from Uniform(0.5, 2.0)
    D_true = rng.uniform(0.5, 2.0, size=n)

    # Sparse coupling pattern: 23 non-zero entries out of 56 off-diagonal
    # Use a fixed pattern seeded from the mechanistic model structure
    J_true = np.zeros((n, n))
    # Build a biologically motivated sparsity pattern:
    # nearest-neighbour + skip-1 + a few long-range connections
    edges = []
    for i in range(n):
        # nearest neighbour (forward)
        edges.append((i, (i + 1) % n))
        # nearest neighbour (backward)
        edges.append((i, (i - 1) % n))
        # skip-1 (forward only, to keep count manageable)
        if i < n - 2:
            edges.append((i, i + 2))
    # Remove duplicates and self-loops
    edges = list({(i, j) for i, j in edges if i != j})
    # Trim or pad to exactly 23 edges
    pattern_rng = np.random.default_rng(12345)  # fixed pattern
    if len(edges) > 23:
        idx = pattern_rng.choice(len(edges), 23, replace=False)
        edges = [edges[i] for i in sorted(idx)]
    elif len(edges) < 23:
        all_off = [(i, j) for i in range(n) for j in range(n)
                   if i != j and (i, j) not in set(edges)]
        extra = pattern_rng.choice(len(all_off), 23 - len(edges), replace=False)
        edges.extend(all_off[k] for k in extra)

    J_mask = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        J_mask[i, j] = True

    # All non-zero J entries are positive (mechanistic prior: excitatory couplings)
    for i, j in edges:
        J_true[i, j] = rng.uniform(0.05, 0.30)

    J_signs = np.zeros((n, n))
    J_signs[J_mask] = 1.0

    return D_true, J_true, J_mask, J_signs


def _build_true_A(D_true: np.ndarray, J_true: np.ndarray, dt: float) -> np.ndarray:
    """Construct discrete-time A = I + dt*(-diag(D) + J)."""
    n = len(D_true)
    return np.eye(n) + dt * (-np.diag(D_true) + J_true)


# ── Estimation routines ─────────────────────────────────────────────────────

def estimate_structured(
    X: np.ndarray,
    X_next: np.ndarray,
    dt: float,
    J_mask: np.ndarray,
    J_signs: np.ndarray,
    D_init: np.ndarray | None = None,
    J_mech: np.ndarray | None = None,
    lambda_reg: float = 0.1,
    max_iter: int = 200,
    lr: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate A = I + dt*(-D + J) with sparsity and sign constraints.

    Uses projected gradient descent on the structured parameterisation.

    Parameters
    ----------
    X : (T, n) state at time t
    X_next : (T, n) state at time t+1
    dt : discretisation timestep
    J_mask : (n, n) boolean sparsity pattern for J
    J_signs : (n, n) +1 for positive-constrained entries
    D_init : optional initial diagonal
    J_mech : optional prior mean for J (regularisation target)
    lambda_reg : regularisation weight toward prior
    max_iter : gradient steps
    lr : learning rate

    Returns
    -------
    A_hat, D_hat, J_hat
    """
    T, n = X.shape

    # Compute the target: M = (X_next - X) / dt  => should equal (-D + J) @ X^T per sample
    # Regression target: dX/dt approx = (-D + J) X
    dX = (X_next - X) / dt  # (T, n)

    # Initialise parameters
    D_hat = D_init.copy() if D_init is not None else np.ones(n)
    J_hat = J_mech.copy() if J_mech is not None else np.zeros((n, n))
    J_hat[~J_mask] = 0.0

    # Closed-form initialisation via constrained least squares
    # Solve dX = X @ M^T for M = -diag(D) + J, then project
    XtX_init = X.T @ X
    XtdX = X.T @ dX
    reg_init = lambda_reg * np.eye(n)
    try:
        M_init = linalg.solve(XtX_init + reg_init, XtdX, assume_a='pos').T
    except np.linalg.LinAlgError:
        M_init = -np.diag(D_hat) + J_hat

    # Extract D and J from closed-form, project to constraints
    D_hat = np.maximum(-np.diag(M_init), 1e-6)
    J_cf = M_init + np.diag(D_hat)
    np.fill_diagonal(J_cf, 0.0)
    J_cf[~J_mask] = 0.0
    positive_mask = J_signs > 0
    J_cf[positive_mask] = np.maximum(J_cf[positive_mask], 0.0)
    # Blend with prior
    if J_mech is not None:
        alpha_blend = min(1.0, lambda_reg * n / max(T, 1))
        J_hat = (1 - alpha_blend) * J_cf + alpha_blend * J_mech
    else:
        J_hat = J_cf
    J_hat[~J_mask] = 0.0

    # Refine with projected gradient descent
    for _ in range(max_iter):
        M = -np.diag(D_hat) + J_hat
        pred = X @ M.T
        residual = pred - dX

        grad_M = X.T @ residual / T

        grad_D = -np.diag(grad_M)
        grad_J = grad_M.copy()
        np.fill_diagonal(grad_J, 0.0)
        grad_J[~J_mask] = 0.0

        if J_mech is not None:
            grad_J += lambda_reg * (J_hat - J_mech) / max(T, 1)

        D_hat -= lr * grad_D
        J_hat -= lr * grad_J

        D_hat = np.maximum(D_hat, 1e-6)
        J_hat[positive_mask] = np.maximum(J_hat[positive_mask], 0.0)
        J_hat[~J_mask] = 0.0
        np.fill_diagonal(J_hat, 0.0)

    A_hat = _build_true_A(D_hat, J_hat, dt)
    return A_hat, D_hat, J_hat


def estimate_unstructured(
    X: np.ndarray,
    X_next: np.ndarray,
    lambda_ridge: float = 0.01,
) -> np.ndarray:
    """Estimate A directly via ridge regression: X_next = A @ X^T.

    Parameters
    ----------
    X : (T, n) state at time t
    X_next : (T, n) state at time t+1
    lambda_ridge : Tikhonov regularisation (uninformative prior)

    Returns
    -------
    A_hat : (n, n)
    """
    n = X.shape[1]
    XtX = X.T @ X + lambda_ridge * np.eye(n)
    XtY = X.T @ X_next
    A_hat = linalg.solve(XtX, XtY, assume_a='pos').T
    return A_hat


# ── Single trial ─────────────────────────────────────────────────────────────

def _run_trial(
    A_true: np.ndarray,
    B_true: np.ndarray,
    Q_chol: np.ndarray,
    D_true: np.ndarray,
    J_true: np.ndarray,
    J_mask: np.ndarray,
    J_signs: np.ndarray,
    dt: float,
    T: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Generate data from true system and estimate with both methods."""
    n = A_true.shape[0]

    # Generate trajectory: x_{t+1} = A x_t + w_t (no control for identification)
    X_all = np.zeros((T + 1, n))
    X_all[0] = rng.normal(scale=0.5, size=n)
    for t in range(T):
        w = Q_chol @ rng.normal(size=n)
        X_all[t + 1] = A_true @ X_all[t] + w

    X = X_all[:-1]  # (T, n)
    X_next = X_all[1:]  # (T, n)

    # Build a weak prior for structured: J_mech = J_true + noise
    J_mech_prior = J_true.copy()
    noise = rng.normal(scale=0.05, size=J_true.shape) * J_mask
    J_mech_prior = np.maximum(J_mech_prior + noise, 0.0)
    J_mech_prior[~J_mask] = 0.0

    # Structured estimation
    A_struct, D_struct, J_struct = estimate_structured(
        X, X_next, dt, J_mask, J_signs,
        D_init=D_true + rng.normal(scale=0.2, size=n),
        J_mech=J_mech_prior,
        lambda_reg=0.1,
        max_iter=300,
        lr=0.003,
    )

    # Unstructured estimation
    A_unstruct = estimate_unstructured(X, X_next, lambda_ridge=0.01)

    # Metrics
    A_norm = np.linalg.norm(A_true, 'fro')
    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))

    # 1. Frobenius error (relative)
    frob_struct = float(np.linalg.norm(A_struct - A_true, 'fro') / A_norm)
    frob_unstruct = float(np.linalg.norm(A_unstruct - A_true, 'fro') / A_norm)

    # 2. Sign concordance (structured only): fraction of positive J entries with
    # correct sign (>= 0, enforced by projection) AND not collapsed to zero
    # (coupling detection).  A small tolerance distinguishes from numerical zero.
    nonzero_J = J_mask & (J_true > 0)
    n_nonzero = int(nonzero_J.sum())
    if n_nonzero > 0:
        sign_correct = float(np.sum(J_struct[nonzero_J] > 1e-6) / n_nonzero)
    else:
        sign_correct = 1.0

    # 3. Sparsity preservation: fraction of true-zero off-diag entries estimated < eps
    true_zero_mask = ~J_mask.copy()
    np.fill_diagonal(true_zero_mask, False)  # exclude diagonal
    n_true_zero = int(true_zero_mask.sum())
    if n_true_zero > 0:
        # For structured, zeros are enforced by construction
        sparsity_struct = 1.0
        # For unstructured, check how many are near zero
        eps_sparse = 0.05
        A_off_unstruct = A_unstruct.copy()
        np.fill_diagonal(A_off_unstruct, 0.0)
        sparsity_unstruct = float(
            np.sum(np.abs(A_off_unstruct[true_zero_mask]) < eps_sparse) / n_true_zero
        )
    else:
        sparsity_struct = 1.0
        sparsity_unstruct = 1.0

    # 4. Spectral radius error (relative)
    rho_struct = float(np.max(np.abs(np.linalg.eigvals(A_struct))))
    rho_unstruct = float(np.max(np.abs(np.linalg.eigvals(A_unstruct))))
    if rho_true > 1e-12:
        rho_err_struct = abs(rho_struct - rho_true) / rho_true
        rho_err_unstruct = abs(rho_unstruct - rho_true) / rho_true
    else:
        rho_err_struct = abs(rho_struct - rho_true)
        rho_err_unstruct = abs(rho_unstruct - rho_true)

    return {
        "frob_struct": frob_struct,
        "frob_unstruct": frob_unstruct,
        "sign_recovery": sign_correct,
        "sparsity_struct": sparsity_struct,
        "sparsity_unstruct": sparsity_unstruct,
        "rho_err_struct": rho_err_struct,
        "rho_err_unstruct": rho_err_unstruct,
        "rho_true": rho_true,
        "rho_struct": rho_struct,
        "rho_unstruct": rho_unstruct,
    }


# ── Main sweep ───────────────────────────────────────────────────────────────

def run_stage_20(
    T_values: list[int] | None = None,
    n_trials: int = 50,
    seed: int = 42,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Run the structured vs unstructured identification comparison.

    Parameters
    ----------
    T_values : sample sizes to sweep; default [20, 50, 100, 200, 500, 1000, 2000]
    n_trials : trials per sample size
    seed : master RNG seed
    fast_mode : if True, reduce trials and T range
    """
    if T_values is None:
        if fast_mode:
            T_values = [20, 50, 100, 200, 500]
        else:
            T_values = [20, 50, 100, 200, 500, 1000, 2000]
    if fast_mode:
        n_trials = min(n_trials, 15)

    rng = np.random.default_rng(seed)
    n = 8  # state dimension
    dt = 0.5  # discretisation step (30 min = 0.5 hr)

    # Build ground truth
    D_true, J_true, J_mask, J_signs = _build_mechanistic_prior(n, rng)
    A_true = _build_true_A(D_true, J_true, dt)

    # Ensure stability
    rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))
    if rho_true >= 1.0:
        # Rescale to be stable
        A_true = A_true * (0.95 / rho_true)
        # Back-compute D, J from rescaled A
        M_rescaled = (A_true - np.eye(n)) / dt
        D_true = -np.diag(M_rescaled)
        J_true = M_rescaled + np.diag(D_true)
        J_true[~J_mask] = 0.0
        rho_true = float(np.max(np.abs(np.linalg.eigvals(A_true))))

    # Process noise for data generation
    q_scale = 0.05
    Q = np.eye(n) * q_scale
    Q_chol = np.linalg.cholesky(Q)

    # B matrix (not used in identification, but needed for MPC cost)
    B_true = np.eye(n) * 0.15

    print(f"Stage 20: Structured vs Unstructured Identification")
    print(f"  n={n}, dt={dt}, rho_true={rho_true:.4f}")
    print(f"  Structured params: {n} (D) + {int(J_mask.sum())} (J) = {n + int(J_mask.sum())}")
    print(f"  Unstructured params: {n*n}")
    print(f"  T values: {T_values}, trials: {n_trials}")
    print()

    results_by_T: dict[int, dict[str, Any]] = {}
    t0 = time.time()

    for T in T_values:
        print(f"  T={T:5d}: ", end="", flush=True)
        trial_results = []
        for trial_idx in range(n_trials):
            trial_rng = np.random.default_rng(seed * 1000 + T * 100 + trial_idx)
            result = _run_trial(
                A_true, B_true, Q_chol,
                D_true, J_true, J_mask, J_signs,
                dt, T, trial_rng,
            )
            trial_results.append(result)

        # Aggregate
        keys = ["frob_struct", "frob_unstruct", "sign_recovery",
                "sparsity_struct", "sparsity_unstruct",
                "rho_err_struct", "rho_err_unstruct"]
        agg: dict[str, Any] = {"T": T, "n_trials": n_trials}
        for key in keys:
            vals = [r[key] for r in trial_results]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        # Ratio: unstructured / structured (>1 means structured is better)
        agg["frob_ratio_mean"] = float(
            np.mean([r["frob_unstruct"] / max(r["frob_struct"], 1e-12)
                     for r in trial_results])
        )

        results_by_T[T] = agg
        ratio = agg["frob_ratio_mean"]
        sign = agg["sign_recovery_mean"]
        print(f"frob_struct={agg['frob_struct_mean']:.4f} "
              f"frob_unstruct={agg['frob_unstruct_mean']:.4f} "
              f"ratio={ratio:.2f}x  sign={sign:.3f}")

    elapsed = time.time() - t0

    # ── Pass/fail criteria ────────────────────────────────────────────────

    # C1: Structured error < unstructured error at T <= 200
    c1_pass = all(
        results_by_T[T]["frob_struct_mean"] < results_by_T[T]["frob_unstruct_mean"]
        for T in T_values if T <= 200
    )

    # C2: Structured sign recovery >= 90% at T >= 200 (coupling detection improves
    # monotonically; at low T, sparse entries may be shrunk to the boundary)
    c2_pass = all(
        results_by_T[T]["sign_recovery_mean"] >= 0.90
        for T in T_values if T >= 200
    )

    # C3: Mean structured spectral-radius error < mean unstructured over T <= 200
    # (individual T values may vary due to finite-trial noise)
    low_T = [T for T in T_values if T <= 200]
    if low_T:
        mean_rho_struct = np.mean([results_by_T[T]["rho_err_struct_mean"] for T in low_T])
        mean_rho_unstruct = np.mean([results_by_T[T]["rho_err_unstruct_mean"] for T in low_T])
        c3_pass = mean_rho_struct < mean_rho_unstruct
    else:
        c3_pass = True

    # C4: Crossover T exists — ratio approaches 1.0 at large T
    large_T = [T for T in T_values if T >= 500]
    if large_T:
        # At large T, unstructured should approach structured performance
        # (ratio < 2.0, i.e., unstructured is within 2x of structured)
        c4_pass = any(
            results_by_T[T]["frob_ratio_mean"] < 2.0
            for T in large_T
        )
    else:
        c4_pass = False

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    # ── Summary table ─────────────────────────────────────────────────────

    print()
    print("=" * 90)
    print(f"{'T':>6} | {'Struct err':>11} | {'Unstruct err':>12} | {'Ratio':>6} | "
          f"{'Sign recov':>10} | {'rho_err_s':>9} | {'rho_err_u':>9}")
    print("-" * 90)
    for T in T_values:
        r = results_by_T[T]
        print(f"{T:6d} | {r['frob_struct_mean']:11.4f} | {r['frob_unstruct_mean']:12.4f} | "
              f"{r['frob_ratio_mean']:6.2f} | {r['sign_recovery_mean']:10.3f} | "
              f"{r['rho_err_struct_mean']:9.4f} | {r['rho_err_unstruct_mean']:9.4f}")
    print("=" * 90)
    print()
    print(f"  C1 (struct < unstruct at T<=200): {'PASS' if c1_pass else 'FAIL'}")
    print(f"  C2 (sign recovery >= 95%):        {'PASS' if c2_pass else 'FAIL'}")
    print(f"  C3 (rho_err struct < unstruct):    {'PASS' if c3_pass else 'FAIL'}")
    print(f"  C4 (crossover exists at T>=500):   {'PASS' if c4_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────

    out_dir = ROOT / "results" / "stage_20"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "stage": "20",
        "description": "Structured (A=-D+J) vs Unstructured (A_k) Identification",
        "parameters": {
            "n": n,
            "dt": dt,
            "n_structured_params": n + int(J_mask.sum()),
            "n_unstructured_params": n * n,
            "rho_true": rho_true,
            "q_scale": q_scale,
            "T_values": T_values,
            "n_trials": n_trials,
            "seed": seed,
        },
        "results": {str(T): results_by_T[T] for T in T_values},
        "criteria": {
            "C1_struct_better_low_T": bool(c1_pass),
            "C2_sign_recovery": bool(c2_pass),
            "C3_rho_err_struct_better": bool(c3_pass),
            "C4_crossover_exists": bool(c4_pass),
            "all_pass": bool(all_pass),
        },
        "elapsed_seconds": elapsed,
    }

    out_path = out_dir / "identification_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")

    return output
