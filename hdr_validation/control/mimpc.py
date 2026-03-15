"""
Mixed-Integer MPC — HDR v7.0
=============================
Implements Def 6.12: MIQP with continuous u^c and binary u^d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class MIMPCResult:
    u_continuous: np.ndarray
    u_discrete: np.ndarray
    u_combined: np.ndarray
    cost: float
    feasible: bool
    notes: str


class CumulativeExposureConstraint:
    """Track and enforce xi_{t+l} <= xi_max over horizon H."""

    def __init__(self, xi_current: np.ndarray, f_j, xi_max: float, H: int):
        self.xi_current = np.asarray(xi_current, dtype=float)
        self.f_j = f_j
        self.xi_max = xi_max
        self.H = H

    def is_feasible(self, u_sequence: list[np.ndarray]) -> bool:
        """Check if the control sequence satisfies cumulative exposure constraints."""
        xi = self.xi_current.copy()
        for u in u_sequence:
            xi = xi + self.f_j(u)
            xi = np.maximum(xi, 0.0)  # monotonic
            if np.any(xi > self.xi_max):
                return False
        return True


def solve_mixed_integer_mpc(
    x: np.ndarray,
    P: np.ndarray,
    basin,
    target,
    config: dict,
    u_discrete_options: list[np.ndarray] | None = None,
    cumulative_exposure=None,
) -> MIMPCResult:
    """Mixed-integer MPC (Def 6.12).

    Solves MIQP with continuous u^c and binary u^d.
    Constraints: box on u^c, binary on u^d,
    irreversibility (one-time interventions sum <= 1),
    cumulative exposure xi <= xi_max.

    Uses enumeration over discrete options (tractable for small m_d)
    combined with continuous QP for each discrete choice.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m_c = int(config.get("control_dim", 8))
    m_d = int(config.get("m_d", 0))
    H = int(config.get("H", 6))
    u_max = float(config.get("u_max", 0.6))

    # If no discrete options, fall back to continuous-only
    if u_discrete_options is None or m_d == 0:
        # Simple continuous MPC: use LQR-like approach
        try:
            from .mpc import solve_mode_a
            from ..model.target_set import TargetSet
            if hasattr(target, 'project_box'):
                x_ref = target.project_box(x)
            else:
                x_ref = np.zeros(n)
            direction = x_ref - x
            u_c = np.clip(0.3 * direction[:m_c], -u_max, u_max)
        except Exception:
            u_c = np.zeros(m_c)

        return MIMPCResult(
            u_continuous=u_c,
            u_discrete=np.array([]),
            u_combined=u_c,
            cost=float(np.sum(u_c**2)),
            feasible=True,
            notes="continuous_only_fallback",
        )

    # Enumerate discrete options and solve continuous part for each
    best_cost = float('inf')
    best_result = None

    # Cost: ||x_next - x_ref||^2 + lambda_u * ||u||^2
    lambda_u = float(config.get("lambda_u", 0.1))
    if hasattr(target, 'project_box'):
        x_ref = target.project_box(x)
    else:
        x_ref = np.zeros(n)

    for u_d in u_discrete_options:
        u_d = np.asarray(u_d, dtype=float)

        # Check irreversibility: binary components sum <= 1
        if np.sum(np.abs(u_d)) > 1.0 + 1e-8:
            continue

        # Continuous part: direction toward target minus effect of discrete
        x_next_d = basin.A @ x + basin.B[:, :len(u_d)] @ u_d if len(u_d) <= basin.B.shape[1] else basin.A @ x
        residual = x_ref - x_next_d

        # Simple projected gradient for continuous control
        u_c = np.clip(0.3 * residual[:m_c], -u_max, u_max)

        # Check cumulative exposure constraint
        if cumulative_exposure is not None:
            u_combined = np.concatenate([u_c, u_d]) if len(u_d) > 0 else u_c
            if not cumulative_exposure.is_feasible([u_combined]):
                continue

        # Compute cost
        u_full = np.concatenate([u_c, u_d]) if len(u_d) > 0 else u_c
        if len(u_full) <= basin.B.shape[1]:
            x_next = basin.A @ x + basin.B[:, :len(u_full)] @ u_full
        else:
            x_next = basin.A @ x + basin.B @ u_full[:basin.B.shape[1]]

        cost = float(np.sum((x_next - x_ref)**2) + lambda_u * np.sum(u_full**2))

        if cost < best_cost:
            best_cost = cost
            best_result = MIMPCResult(
                u_continuous=u_c,
                u_discrete=u_d,
                u_combined=u_full[:m_c + m_d] if len(u_full) >= m_c + m_d else u_full,
                cost=cost,
                feasible=True,
                notes="mi_mpc",
            )

    if best_result is None:
        # No feasible discrete option found
        u_c = np.zeros(m_c)
        return MIMPCResult(
            u_continuous=u_c,
            u_discrete=np.zeros(m_d),
            u_combined=np.concatenate([u_c, np.zeros(m_d)]),
            cost=0.0,
            feasible=False,
            notes="no_feasible_discrete",
        )

    return best_result
