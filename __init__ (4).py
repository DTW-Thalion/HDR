"""
Inference–Control Interface (ICI) — HDR v5.0
=============================================

Implements the three ICI conditions from Section 9 and the quantified ISS bound
(Proposition 9.1), compound missingness–base-rate–mixing bound (Proposition 9.2),
Brier reliability calibration (Definition 8.2), and posterior calibration adjustment
(Definition 8.1 / p_A^robust).

All functions are pure numpy; no cvxpy or LMI solvers required.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 9.2 — Compound Missingness / Base-Rate / Mixing Bound
# ─────────────────────────────────────────────────────────────────────────────

def compute_T_k_eff(
    T: float,
    pi_k: float,
    p_miss: float,
    rho_k: float,
) -> float:
    """Effective sample count for basin k under compound degradation.

    T_k^eff = T * pi_k * (1 - p_miss) * (1 - rho_k)

    Parameters
    ----------
    T       : total available time steps
    pi_k    : empirical base rate of basin k occupancy  (e.g. 0.16 for maladaptive)
    p_miss  : fraction of observations that are missing  (e.g. 0.516)
    rho_k   : spectral radius of basin k dynamics matrix (e.g. 0.96)

    Returns
    -------
    T_k_eff : effective independent observations (float, may be << 1)
    """
    return float(T * max(pi_k, 0.0) * max(1.0 - p_miss, 0.0) * max(1.0 - rho_k, 0.0))


def compute_degradation_factor(pi_k: float, p_miss: float, rho_k: float) -> float:
    """Factor by which T is reduced: T_k_eff = T * degradation_factor."""
    return float(max(pi_k, 0.0) * max(1.0 - p_miss, 0.0) * max(1.0 - rho_k, 0.0))


def compute_omega_min(n_theta: int, epsilon: float = 0.10, delta: float = 0.05) -> float:
    """Regime boundary ω_min: T_k_eff must exceed this for valid inference.

    Below ω_min, IMM+Kalman is insufficient; Mode C should activate.

    Based on Heuristic Scaling Proposition H.12:
    ω_min = C * (n_theta / epsilon^2) * log(1/delta)
    where C=1 (normalised) yields a practical threshold.
    """
    return float((n_theta / (epsilon ** 2)) * np.log(1.0 / delta))


# ─────────────────────────────────────────────────────────────────────────────
# Proposition 9.1 — Quantified ISS Bound
# ─────────────────────────────────────────────────────────────────────────────

def compute_mu_bar_required(
    epsilon_control: float,
    alpha: float,
    delta_A: float,
    delta_B: float,
    K_lqr_norm: float,
) -> float:
    """Upper bound on allowable mode-error probability for ISS guarantee.

    From Proposition H.3:  γ(μ̄) ∝ √μ̄ · (ΔA + ΔB‖K_k‖) / α

    We invert to find μ̄_required such that γ(μ̄_required) ≤ ε_control:

        μ̄_required = [ε_control * α / (ΔA + ΔB * ‖K_k‖)]²

    Parameters
    ----------
    epsilon_control : maximum allowable residual steady-state deviation
    alpha           : stage-cost decrease rate (Lyapunov decrease factor)
    delta_A         : max dynamics mismatch ‖A_j − A_k‖₂
    delta_B         : max input-map mismatch ‖B_j − B_k‖₂
    K_lqr_norm      : operator norm ‖K_LQR‖₂

    Returns
    -------
    mu_bar_required : maximum allowable μ̄  (in [0, 1])
    """
    denom = delta_A + delta_B * K_lqr_norm
    if denom < 1e-12:
        return 1.0  # no mismatch possible → any μ̄ is fine
    return float(np.clip((epsilon_control * alpha / denom) ** 2, 0.0, 1.0))


def compute_iss_residual(
    mu_bar: float,
    alpha: float,
    delta_A: float,
    delta_B: float,
    K_lqr_norm: float,
) -> float:
    """Residual steady-state deviation under mode-error probability μ̄.

    γ(μ̄) = √μ̄ · (ΔA + ΔB‖K_k‖) / α
    """
    return float(np.sqrt(mu_bar) * (delta_A + delta_B * K_lqr_norm) / max(alpha, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Definition 8.2 — Brier Reliability (Calibration) Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def brier_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Decompose Brier score into reliability (calibration), resolution, and uncertainty.

    Brier = Reliability − Resolution + Uncertainty

    Parameters
    ----------
    y_true  : binary ground-truth labels (0/1)
    y_prob  : predicted probability for positive class
    n_bins  : number of equal-width probability bins

    Returns
    -------
    dict with keys:
      brier_score      : overall Brier score
      reliability      : calibration error term (lower is better; 0 = perfect)
      resolution       : discrimination term (higher is better)
      uncertainty      : base-rate entropy term
      n_samples        : number of samples
      bin_confidences  : per-bin mean predicted probability
      bin_frequencies  : per-bin empirical positive frequency
      bin_counts       : number of samples per bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_rate = float(np.mean(y_true))
    uncertainty = float(base_rate * (1.0 - base_rate))
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    bin_conf, bin_freq, bin_cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bin_conf.append(float((lo + hi) / 2))
            bin_freq.append(float("nan"))
            bin_cnt.append(0)
            continue
        p_bar = float(np.mean(y_prob[mask]))
        o_bar = float(np.mean(y_true[mask]))
        reliability += (cnt / n) * (p_bar - o_bar) ** 2
        resolution += (cnt / n) * (o_bar - base_rate) ** 2
        bin_conf.append(p_bar)
        bin_freq.append(o_bar)
        bin_cnt.append(cnt)
    return {
        "brier_score": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "n_samples": n,
        "bin_confidences": bin_conf,
        "bin_frequencies": bin_freq,
        "bin_counts": bin_cnt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Definition 8.1 — Calibration-Adjusted Mode B Threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_p_A_robust(
    p_A: float,
    k_calib: float,
    R_brier: float,
) -> float:
    """Calibration-adjusted Mode B entry threshold.

    p_A^robust = p_A + k_calib * R_Brier

    A miscalibrated posterior (high R_Brier) inflates the threshold,
    making Mode B entry more conservative.

    Parameters
    ----------
    p_A      : nominal Mode B entry threshold (default 0.70)
    k_calib  : scaling factor for calibration penalty (default 1.0)
    R_brier  : Brier reliability component (from brier_reliability())

    Returns
    -------
    p_A_robust : adjusted threshold clipped to [p_A, 0.99]
    """
    return float(np.clip(p_A + k_calib * R_brier, p_A, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# Theorem H.10 — Horizon Truncation Error
# ─────────────────────────────────────────────────────────────────────────────

def compute_epsilon_H(rho_star: float, H: int) -> float:
    """Horizon-truncation error term from Theorem H.10.

    ε_H = (ρ*)^H   where ρ* = max_u ρ(P^u_{TT})
    """
    return float(rho_star ** H)


def compute_mode_b_suboptimality_bound(
    epsilon_q: float,
    delta_P: float,
    H: int,
    rho_star: float,
) -> float:
    """Full Mode B suboptimality bound (Theorem H.10).

    Δ ≤ 2*ε_q + δ_P*H + ε_H
    """
    eps_H = compute_epsilon_H(rho_star, H)
    return float(2.0 * epsilon_q + delta_P * H + eps_H)


# ─────────────────────────────────────────────────────────────────────────────
# ICI State Vector — full diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def compute_ici_state(
    mu_hat: float,
    mu_bar_required: float,
    R_brier: float,
    R_brier_max: float,
    T_k_eff_per_basin: list[float],
    omega_min: float,
) -> dict:
    """Compute the full ICI state vector (Algorithm 1, Step 5 in v5.0).

    Returns
    -------
    ici_state dict with:
      condition_i   : μ̂ ≥ μ̄_required  → Mode C recommended
      condition_ii  : R_Brier ≥ R_max  → Mode C recommended
      condition_iii : any T_k_eff < ω_min → Mode C recommended
      mode_c_recommended : any condition True
      worst_basin_idx    : basin with smallest T_k_eff
      worst_T_k_eff      : smallest T_k_eff value
    """
    cond_i = bool(mu_hat >= mu_bar_required)
    cond_ii = bool(R_brier >= R_brier_max)
    worst_k = int(np.argmin(T_k_eff_per_basin)) if T_k_eff_per_basin else 0
    worst_T = float(T_k_eff_per_basin[worst_k]) if T_k_eff_per_basin else float("nan")
    cond_iii = bool(worst_T < omega_min)
    return {
        "condition_i": cond_i,
        "condition_ii": cond_ii,
        "condition_iii": cond_iii,
        "mode_c_recommended": bool(cond_i or cond_ii or cond_iii),
        "worst_basin_idx": worst_k,
        "worst_T_k_eff": worst_T,
        "mu_hat": mu_hat,
        "mu_bar_required": mu_bar_required,
        "R_brier": R_brier,
        "R_brier_max": R_brier_max,
        "omega_min": omega_min,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Posterior calibration via isotonic regression (Platt / isotonic)
# ─────────────────────────────────────────────────────────────────────────────

def isotonic_calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Simple monotone (isotonic-style) posterior calibration via bin averaging.

    Returns a lookup array of shape (n_bins,) mapping probability bin midpoint
    to calibrated probability.  Applied via ``apply_calibration``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    cal_map = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if np.sum(mask) == 0:
            cal_map[i] = (lo + hi) / 2.0
        else:
            cal_map[i] = float(np.mean(y_true[mask]))
    # Enforce monotonicity via pool-adjacent-violators (PAV, simplified)
    for i in range(1, n_bins):
        if cal_map[i] < cal_map[i - 1]:
            cal_map[i] = cal_map[i - 1]
    return cal_map


def apply_calibration(y_prob_raw: np.ndarray, cal_map: np.ndarray) -> np.ndarray:
    """Apply bin-based calibration map to raw probabilities."""
    y_prob_raw = np.asarray(y_prob_raw, dtype=float)
    n_bins = len(cal_map)
    bin_indices = np.clip((y_prob_raw * n_bins).astype(int), 0, n_bins - 1)
    return cal_map[bin_indices]
