"""
Mode C — Active Identification Mode
================================================

Mode C is the third operating mode of HDR (co-equal with Mode A and Mode B).
It activates when the Inference-Control Interface (ICI) detects that inference
quality is insufficient for Mode A guarantees or Mode B safety.

Objective (Proposition 7.1):
    u*_C = argmax_{u ∈ U_safe} tr(F_k(θ_k; u))

subject to constraints (14)–(17), where F_k is the (approximate) Fisher
information matrix for basin k parameters given control input u.

In the non-oracle validation setting, we proxy tr(F_k) using a persistent-
excitation criterion: maximise the minimum singular value of the sample
regressor matrix.  This is tractable, parameter-free, and aligns with
Assumption F.1 of the paper.

Entry conditions (any one triggers Mode C):
  (i)   μ̂ ≥ μ̄_required          — mode-error probability exceeds ISS bound
  (ii)  R_Brier ≥ R_max           — posterior is miscalibrated
  (iii) T_k_eff < ω_min (any k)   — below regime boundary

Exit conditions (all must be satisfied):
  (i)  μ̂ < μ̄_required
  (ii) R_Brier < R_max
  (iii) T_k_eff ≥ ω_min for all k

Fallback: if Mode C runs for T_C_max consecutive steps without exiting,
a degradation flag is set and Mode A is forced with the best available
(possibly miscalibrated) posterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Mode C Action: Fisher-information-maximising dither
# ─────────────────────────────────────────────────────────────────────────────

def mode_c_action(
    x_hat: np.ndarray,
    control_dim: int,
    sigma_dither: float,
    rng: np.random.Generator,
    used_burden: float,
    budget: float,
    u_max: float = 0.35,
) -> np.ndarray:
    """Generate a Mode C dither action that maximises persistent excitation.

    Per Appendix H.5, dither amplitude σ_ζ is calibrated so that the
    excitation matrix Φ_N stays non-singular while remaining within budget B.

    Simple implementation: zero-mean Gaussian dither clipped by safety bounds.
    A production system would solve the Fisher-information QP; this approximation
    is appropriate for the in silico validation.
    """
    remaining = max(budget - used_burden, 0.0)
    # Scale dither down as budget runs out
    effective_sigma = sigma_dither * min(1.0, remaining / max(budget, 1e-8) + 0.1)
    u = rng.normal(scale=effective_sigma, size=control_dim)
    u = np.clip(u, -u_max, u_max)
    return u


def fisher_information_proxy(
    regressors: np.ndarray,
) -> float:
    """Proxy for tr(F_k): minimum singular value of the regressor matrix.

    Higher values indicate better persistent excitation.

    Parameters
    ----------
    regressors : (N, p) array of stacked regressor vectors [x_t; u_t; 1]

    Returns
    -------
    sigma_min : minimum singular value of X^T X / N  (or 0 if N < p)
    """
    if regressors.shape[0] < 2:
        return 0.0
    XtX = regressors.T @ regressors / regressors.shape[0]
    sv = np.linalg.svd(XtX, compute_uv=False)
    return float(sv[-1])  # minimum singular value


# ─────────────────────────────────────────────────────────────────────────────
# Mode C State Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModeCTracker:
    """Tracks Mode C episode state across steps.

    Attributes
    ----------
    active          : whether Mode C is currently active
    steps_in_mode_c : consecutive steps spent in Mode C
    T_C_max         : maximum allowed consecutive Mode C steps
    degradation_flag: set True when T_C_max is exceeded
    regressors      : rolling buffer of [x_hat; u; 1] vectors for Fisher proxy
    regressor_max_len: max rows to keep in rolling buffer
    initial_R_brier : R_Brier value at Mode C entry (for improvement tracking)
    initial_T_k_eff : T_k_eff values at Mode C entry
    """
    T_C_max: int = 50
    regressor_max_len: int = 256
    active: bool = False
    steps_in_mode_c: int = 0
    degradation_flag: bool = False
    regressors: list = field(default_factory=list)
    initial_R_brier: float = float("nan")
    initial_T_k_eff: list = field(default_factory=list)
    _entry_step: int = 0

    def enter(self, step: int, R_brier: float, T_k_eff_per_basin: list[float]) -> None:
        self.active = True
        self._entry_step = step
        self.steps_in_mode_c = 0
        self.initial_R_brier = R_brier
        self.initial_T_k_eff = list(T_k_eff_per_basin)

    def tick(self, u: np.ndarray, x_hat: np.ndarray) -> None:
        """Record regressor and increment counter."""
        if self.active:
            regressor = np.concatenate([x_hat, u, [1.0]])
            self.regressors.append(regressor)
            if len(self.regressors) > self.regressor_max_len:
                self.regressors.pop(0)
            self.steps_in_mode_c += 1
            if self.steps_in_mode_c >= self.T_C_max:
                self.degradation_flag = True

    def should_exit(
        self,
        mu_hat: float,
        mu_bar_required: float,
        R_brier: float,
        R_brier_max: float,
        T_k_eff_per_basin: list[float],
        omega_min: float,
    ) -> bool:
        """Return True if all exit conditions are satisfied."""
        cond_i_ok = mu_hat < mu_bar_required
        cond_ii_ok = R_brier < R_brier_max
        cond_iii_ok = all(t >= omega_min for t in T_k_eff_per_basin)
        return bool(cond_i_ok and cond_ii_ok and cond_iii_ok)

    def exit(self) -> None:
        self.active = False
        self.steps_in_mode_c = 0
        self.regressors = []

    @property
    def fisher_proxy(self) -> float:
        if len(self.regressors) < 4:
            return 0.0
        R = np.asarray(self.regressors)
        return fisher_information_proxy(R)


# ─────────────────────────────────────────────────────────────────────────────
# Entry / Exit logic helpers
# ─────────────────────────────────────────────────────────────────────────────

def mode_c_entry_conditions(
    mu_hat: float,
    mu_bar_required: float,
    R_brier: float,
    R_brier_max: float,
    T_k_eff_per_basin: list[float],
    omega_min: float,
) -> dict[str, bool]:
    """Evaluate all three Mode C entry conditions.

    Returns dict with keys 'condition_i', 'condition_ii', 'condition_iii',
    and 'any_triggered'.
    """
    c_i = bool(mu_hat >= mu_bar_required)
    c_ii = bool(R_brier >= R_brier_max)
    c_iii = bool(any(t < omega_min for t in T_k_eff_per_basin))
    return {
        "condition_i": c_i,
        "condition_ii": c_ii,
        "condition_iii": c_iii,
        "any_triggered": bool(c_i or c_ii or c_iii),
    }


def supervisor_mode_select(
    ici_state: dict,
    mode_b_conditions_met: bool,
    mode_c_active: bool,
    degradation_flag: bool,
    t_k_eff_below_threshold: bool = False,
) -> str:
    """Triple-mode supervisor logic (Table 3, HDR v7.3).

    Priority: Mode C > Mode B > Mode A

    Parameters
    ----------
    ici_state                : output of compute_ici_state()
    mode_b_conditions_met    : all three Mode B entry conditions satisfied
    mode_c_active            : Mode C is already running
    degradation_flag         : Mode C T_C_max exceeded
    t_k_eff_below_threshold  : any basin has T_k_eff < omega_min; when True,
                               Mode C permanently preempts Mode B regardless of
                               whether Mode C has formally exited.

    Returns
    -------
    'mode_c', 'mode_b', or 'mode_a'
    """
    if degradation_flag:
        return "mode_a"  # forced fallback with degradation flag
    if t_k_eff_below_threshold:
        return "mode_c"  # ICI gate: T_k_eff too low → Mode C preempts Mode B
    if mode_c_active or ici_state.get("mode_c_recommended", False):
        return "mode_c"
    if mode_b_conditions_met:
        return "mode_b"
    return "mode_a"
