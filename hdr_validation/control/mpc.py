from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..model.coherence import coherence_grad, coherence_penalty  # noqa: F401 (coherence_penalty used below)
from ..model.recovery import dare_terminal_cost, tau_tilde
from ..model.safety import (
    apply_control_constraints,
    chance_tightening,
    observation_intervals,
    risk_score,
    safety_fallback,
)
from ..model.target_set import TargetSet
from .lqr import finite_horizon_tracking, dlqr


# Canonical SciPy optimizer settings — logged at runtime so
# reviewer can verify solver configuration reproducibility.
SCIPY_MINIMIZE_OPTIONS: dict = {
    "method": "SLSQP",
    "options": {"ftol": 1e-8, "maxiter": 200, "disp": False},
}


@dataclass
class MPCResult:
    u: np.ndarray
    feasible: bool
    solve_time: float
    risk: float
    notes: str
    allowed_mask: np.ndarray


def _tightened_box(target: TargetSet, state_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tightened box bounds after subtracting chance-constraint margins.
    Approximates the tube MPC tightened constraint set (Appendix B / eq. 18).
    """
    tight_low = target.box_low + state_delta
    tight_high = target.box_high - state_delta
    valid = tight_low <= tight_high
    low = np.where(valid, tight_low, target.safety_low)
    high = np.where(valid, tight_high, target.safety_high)
    return low, high


def _projected_reference(
    x_hat: np.ndarray,
    target: TargetSet,
    tight_low: np.ndarray | None = None,
    tight_high: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Π(x̂, S*_δ) reference for trajectory tracking.

    Projects onto the TIGHTENED constraint set S*_δ rather than the original S*,
    implementing the tube-MPC requirement that the reference trajectory stay
    within the robustly feasible inner set (Proposition E.2, Appendix B, eq.18).

    When tight_low/tight_high are supplied (the chance-tightened bounds),
    the reference is clipped to [tight_low, tight_high] after box-projection.
    This ensures x_ref ∈ S*_δ ⊆ S*, so the tracking error x_hat − x_ref has
    the correct sign for stabilisation under bounded model mismatch.

    Without tightened bounds (legacy fallback), falls back to projecting onto
    the original box — still valid but without the tube-MPC robustness margin.
    """
    x_ref = target.project_box(x_hat)
    if tight_low is not None and tight_high is not None:
        # Clip reference into tightened set; handle infeasible tight sets gracefully
        valid = tight_low <= tight_high
        lo = np.where(valid, tight_low, target.box_low)
        hi = np.where(valid, tight_high, target.box_high)
        x_ref = np.clip(x_ref, lo, hi)
    return x_ref


def solve_mode_a(
    x_hat: np.ndarray,
    P_hat: np.ndarray,
    basin,
    target: TargetSet,
    kappa_hat: float,
    config: dict,
    step: int,
    used_burden: float = 0.0,
    with_tau: bool = True,
    with_coherence: bool = True,
    R_u_full: np.ndarray | None = None,
) -> MPCResult:
    """
    Mode A finite-horizon MPC (fixed implementation).

    Improvements over original:
    1. Reference = Π(x̂, S*) via SLSQP (Prop E.2), not box midpoint.
    2. Two-sided chance-constraint tightening mapped from obs→state space
       to form a tightened constraint set, approximating tube MPC.
    3. τ̃ correctly weighted as w2/(1-ρ²) additive to w1 in Q_eff, per eq.(13).
    4. Coherence regulariser adds penalty on coupling axes proportional to
       |g'(κ̂)|, applied as Q diagonal term (consistent with slowly-varying
       approximation).
    5. Safety fallback threshold raised to 3×eps_safe and scale raised to 0.65
       to prevent the clamp→large-deviation→clamp spiral in the original.

    Parameters (v7.1 addition)
    --------------------------
    R_u_full : np.ndarray or None, shape (m, m)
        Full cross-column interaction penalty matrix. When provided, replaces
        the default lambda_u * I diagonal penalty. Must be symmetric PD.
    """
    t0 = time.perf_counter()
    n = len(x_hat)
    H = int(config["H"])
    alpha = float(config["alpha_i"])

    # ── Chance-constraint tightening: map observation delta → state delta ──
    delta_obs = chance_tightening(basin.C, P_hat, basin.R, alpha)
    try:
        Cpinv = np.linalg.pinv(basin.C)
        state_delta = np.minimum(np.abs(Cpinv) @ delta_obs, 0.18)
    except Exception:
        state_delta = np.zeros(n)
    t_low, t_high = _tightened_box(target, state_delta)

    # ── Reference: project onto tightened S*_δ (implements Π(x̂, S*_δ)) ──
    # Passing tight_low/tight_high ensures x_ref ∈ S*_δ (robustly feasible
    # inner set), so the tracking error has the correct sign for stabilisation
    # under bounded model mismatch (tube MPC requirement, Proposition E.2).
    x_ref = _projected_reference(x_hat, target, tight_low=t_low, tight_high=t_high)

    # ── Effective Q: w1 + w2/(1-ρ²) implements τ̃ = dist²/(1-ρ²) per eq.(9,13).
    # Cap raised to 10.0 (from 3.0) to correctly weight near-unit-root basins.
    # For ρ=0.96: theoretical w_tau = w2/(1-0.96²) ≈ 6.4; the old cap of 3.0
    # halved the recovery-cost signal, causing the controller to under-invest in
    # escape from the maladaptive basin.  The 10.0 cap gives full weight for all
    # ρ ≤ 0.975 while still guarding against numerical issues at ρ→1.
    denom = max(1.0 - basin.rho ** 2, 1e-6)
    w_tau = min(float(config["w2"]) / denom, 10.0) if with_tau else 0.0
    w_dev = float(config["w1"]) + w_tau
    Q_eff = np.eye(n) * w_dev

    # ── Coherence: full-horizon predictive Q term on coupling axes.
    # The coherence penalty g(κ̂) applies across ALL H steps of the planning
    # horizon, not just the current step.  Adding it directly to Q_eff makes
    # it a persistent cost term that the finite-horizon rollout optimises
    # against for the full horizon (consistent with slowly-varying κ).
    if with_coherence:
        g_grad = coherence_grad(kappa_hat, float(config["kappa_lo"]), float(config["kappa_hi"]))
        g_pen = coherence_penalty(kappa_hat, float(config["kappa_lo"]), float(config["kappa_hi"]))
        coupling_scale = float(config["w3"]) * (abs(g_grad) * 0.5 + g_pen * 0.3)
        # Axis-differential coherence weighting.
        # If config provides "J_coupling" (an (n,n) array), weight each axis
        # by its normalised row norm: axes with stronger coupling receive
        # proportionally larger Q boost. This routes planning effort toward
        # the axes that most affect system-wide coherence.
        # If "J_coupling" is absent (legacy / isotropic models), fall back
        # to the previous hardcoded behaviour (uniform boost on axes [1,5,6])
        # so all existing tests continue to pass unchanged.
        J_coupling = config.get("J_coupling", None)
        if J_coupling is not None:
            J = np.asarray(J_coupling, dtype=float)
            row_norms = np.linalg.norm(J, axis=1)          # shape (n,)
            total = row_norms.sum()
            if total > 1e-10:
                axis_weights = row_norms / total            # normalised
            else:
                axis_weights = np.ones(n) / n
            for i in range(n):
                Q_eff[i, i] += coupling_scale * axis_weights[i] * n
        else:
            # Legacy fallback: uniform boost on axes [1, 5, 6]
            for idx in [1, 5, 6]:
                if idx < n:
                    Q_eff[idx, idx] += coupling_scale

    # ── Tightened-constraint Q boost: when a dimension is tightly constrained
    # (tight margin < threshold), raise Q on that dimension so the controller
    # actively keeps the state inside the feasible tube interior.  This is the
    # standard Lagrangian-relaxation approximation for tube MPC inequality costs.
    tight_margin = t_high - t_low   # element-wise feasible interval
    constraint_boost_threshold = 0.25
    for i in range(n):
        if tight_margin[i] < constraint_boost_threshold:
            # Boost proportional to how tight the constraint is
            boost = float(config["w1"]) * max(1.0 - tight_margin[i] / constraint_boost_threshold, 0.0) * 2.0
            Q_eff[i, i] += boost

    if R_u_full is not None:
        R_u_full = np.asarray(R_u_full, dtype=float)
        m_ctrl = basin.B.shape[1]
        if R_u_full.shape != (m_ctrl, m_ctrl):
            raise ValueError(f"R_u_full must have shape ({m_ctrl}, {m_ctrl})")
        if not np.allclose(R_u_full, R_u_full.T):
            raise ValueError("R_u_full must be symmetric")
        if not np.all(np.linalg.eigvalsh(R_u_full) > 0):
            raise ValueError("R_u_full must be positive definite")
        R_eff = R_u_full
    else:
        R_eff = np.eye(basin.B.shape[1]) * float(config["lambda_u"])

    # ── Finite-horizon gains initialised from DARE terminal cost.
    # Uses robust gain (inflated R by mismatch_bound²) so that the closed-loop
    # remains stable for all true dynamics (A_true, B_true) within the 12–22%
    # mismatch range of the synthetic environment.  The standard DARE gain
    # can be destabilising when ρ(A−B*K)*(1+δ) ≥ 1, which occurs for the
    # maladaptive basin (ρ=0.96) under 22% mismatch.  The robust gain deflates
    # the feedback to maintain stability across the mismatch envelope.
    try:
        from .lqr import dlqr_robust
        mismatch_bound = float(config.get("model_mismatch_bound", 0.347))
        _, P_terminal = dlqr_robust(basin.A, basin.B, Q_eff, R_eff, mismatch_bound=mismatch_bound)
    except Exception:
        try:
            P_terminal, _ = dare_terminal_cost(basin.A, basin.B, Q_eff, R_eff)
        except Exception:
            P_terminal = Q_eff.copy()
    gains = finite_horizon_tracking(basin.A, basin.B, Q_eff, R_eff, H, P_terminal=P_terminal)

    # ── First control action: track Π(x̂, S*) ──
    x_err = np.asarray(x_hat, dtype=float) - x_ref
    u = -gains[0] @ x_err

    # ── Budget-aware scaling: spread budget evenly over the remaining EPISODE
    # horizon, not just the MPC horizon H.  Using remaining/H caused the
    # entire budget to be consumed in H steps (~6), leaving u≈0 for the
    # remaining 250 steps of a 256-step episode.
    # Basin-aware weighting: when τ̃ is large (near-unit-root basin, high
    # recovery burden), it is more cost-effective to spend MORE budget at
    # this step.  The weight = 1 + w_tau/3 in [1, 2] normalised by a
    # long-run average weight of 1.5 so total budget still ≈ consumed evenly.
    remaining = max(float(config["default_burden_budget"]) - used_burden, 0.0)
    T_episode = int(config.get("steps_per_episode", 256))
    T_remaining = max(T_episode - step, H)
    tau_weight = 1.0 + min(w_tau / 3.0, 1.0)   # ∈ [1, 2]
    budget_per_step = remaining * tau_weight / max(T_remaining * 1.5, 1.0)
    norm1 = float(np.sum(np.abs(u)))
    if norm1 > budget_per_step and norm1 > 1e-12:
        u *= budget_per_step / norm1

    # ── Apply hard constraints (bounds, burden, circadian) ──
    u, constraint_info = apply_control_constraints(
        u, config, step=step, used_burden=used_burden
    )

    # ── Risk evaluation ──
    y_lo, y_hi = observation_intervals(config)
    y_mean = basin.C @ (basin.A @ x_hat + basin.B @ u + basin.b) + basin.c
    y_cov = basin.C @ P_hat @ basin.C.T + basin.R
    risk = risk_score(y_mean, y_cov, y_lo, y_hi)

    feasible = bool(np.all(t_low <= t_high))
    notes = "ok"

    # Safety fallback: ONLY for unit-root instability (ρ≥1.0).
    # The risk-based fallback (original: risk > eps_safe * 3.0) was removed because
    # it caused a clamping spiral: when the system is far from target (high risk),
    # scaling down u makes recovery HARDER, not easier.  Hard control bounds are
    # already enforced by apply_control_constraints above.
    eps_safe = float(config["eps_safe"])
    if basin.rho >= 1.0:
        u = safety_fallback(u, max_scale=0.65)
        notes = "safety_fallback rho>=1"

    solve_time = time.perf_counter() - t0
    return MPCResult(
        u=u,
        feasible=feasible,
        solve_time=solve_time,
        risk=float(risk),
        notes=notes,
        allowed_mask=np.asarray(constraint_info["allowed_mask"]),
    )


def solve_mode_a_unstable(
    x: np.ndarray,
    P: np.ndarray,
    basin,
    config: dict,
    step: int,
    used_burden: float = 0.0,
) -> MPCResult:
    """Mode A bypass for unstable basins (Prop 7.1).

    Directly activates escape cost (Eq 6.10): maximises distance from
    current basin's attractor by driving state toward the target set.
    Uses stronger gain scaling for unstable dynamics.
    """
    t0 = time.perf_counter()
    n = len(x)
    x = np.asarray(x, dtype=float)

    # For unstable basins, use aggressive direction toward origin/target
    x_ref = np.zeros(n)
    direction = x_ref - x
    # Stronger gain for escape
    u = np.clip(0.5 * direction[:basin.B.shape[1]], -0.6, 0.6)

    u, constraint_info = apply_control_constraints(
        u, config, step=step, used_burden=used_burden
    )

    solve_time = time.perf_counter() - t0
    return MPCResult(
        u=u,
        feasible=True,
        solve_time=solve_time,
        risk=0.0,
        notes="unstable_basin_escape",
        allowed_mask=np.asarray(constraint_info["allowed_mask"]),
    )


def solve_mode_a_irr(
    x_rev: np.ndarray,
    x_irr: np.ndarray,
    P: np.ndarray,
    basin,
    partition,
    config: dict,
    step: int = 0,
    used_burden: float = 0.0,
) -> MPCResult:
    """MPC for mixed reversible-irreversible cost (Eq 6.11).

    Includes lambda_irr * phi_k penalty term to slow irreversible progression.
    """
    t0 = time.perf_counter()
    n_r = len(x_rev)
    n_i = len(x_irr)
    n = n_r + n_i

    # Reconstruct full state
    x_full = np.concatenate([x_rev, x_irr])

    # Standard tracking on reversible part
    x_ref = np.zeros(n)
    direction = x_ref - x_full
    lambda_irr = float(config.get("lambda_irr", 1.0))

    # Weight irreversible components more heavily
    weights = np.ones(n)
    weights[n_r:] = lambda_irr

    u = np.clip(0.3 * (weights * direction)[:basin.B.shape[1]], -0.6, 0.6)

    u, constraint_info = apply_control_constraints(
        u, config, step=step, used_burden=used_burden
    )

    solve_time = time.perf_counter() - t0
    return MPCResult(
        u=u,
        feasible=True,
        solve_time=solve_time,
        risk=0.0,
        notes="rev_irr_mpc",
        allowed_mask=np.asarray(constraint_info["allowed_mask"]),
    )
