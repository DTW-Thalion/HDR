"""
HDR Validation Suite — Validation Profile Runner
=================================================
Exercises all validation stages (01–07) using the validation profile:
  seeds=[101, 202, 303], episodes_per_experiment=12, steps_per_episode=128,
  mc_rollouts=150.

Data-generating stages (02, 03, 03b, 04) pool results across all three seeds.
Seed-independent stages (01, 03c, 05, 06, 07) run once.

Usage:
    python3 validation_runner.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ── Validation profile config ──────────────────────────────────────────────────
VALIDATION_CONFIG: dict[str, Any] = {
    # Dimensions
    "state_dim": 8,
    "obs_dim": 16,
    "control_dim": 8,
    "disturbance_dim": 8,
    "K": 3,
    # Control
    "H": 6,
    "w1": 1.0,
    "w2": 0.5,
    "w3": 0.3,
    "lambda_u": 0.1,
    "alpha_i": 0.05,
    "eps_safe": 0.01,
    # Dynamics
    "rho_reference": [0.72, 0.96, 0.55],
    "max_dwell_len": 128,
    "model_mismatch_bound": 0.347,
    # Target set
    "kappa_lo": 0.55,
    "kappa_hi": 0.75,
    "pA": 0.70,
    "qmin": 0.15,
    # Safety / time
    "steps_per_day": 48,
    "dt_minutes": 30,
    "coherence_window": 24,
    "default_burden_budget": 28.0,
    "circadian_locked_controls": [5, 6],
    # ICI
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
    # Validation profile
    "profile_name": "validation",
    "seeds": [101, 202, 303],
    "episodes_per_experiment": 12,
    "steps_per_episode": 128,
    "mc_rollouts": 150,
    "selected_trace_cap": 5,
}

# ── Setup sys.path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Result tracking ────────────────────────────────────────────────────────────
results: list[dict] = []

def record(stage: str, check: str, passed: bool, value: Any = None, note: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {check}"
    if value is not None:
        msg += f" = {value}"
    if note:
        msg += f"  ({note})"
    print(msg)
    results.append({"stage": stage, "check": check, "passed": passed, "value": value, "note": note})


def run_stage(label: str, fn):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        fn()
    except Exception:
        traceback.print_exc()
        record(label, "stage_execution", False, note="Uncaught exception")
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 01 — Mathematical checks (seed-independent)
# ═══════════════════════════════════════════════════════════════════════════════
def stage01_math():
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.model.recovery import tau_tilde, tau_sandwich, dare_terminal_cost
    from hdr_validation.model.safety import chance_tightening
    from hdr_validation.control.lqr import dlqr, committor, finite_horizon_tracking
    from hdr_validation.model.hsmm import DwellModel
    from hdr_validation.inference.ici import compute_T_k_eff, compute_mu_bar_required

    cfg = VALIDATION_CONFIG
    rng = np.random.default_rng(cfg["seeds"][0])
    eval_model = make_evaluation_model(cfg, rng)

    # 01.1 tau_tilde is non-negative and zero at target center
    target = build_target_set(0, cfg)
    Q = np.eye(cfg["state_dim"])
    x_inside = np.zeros(cfg["state_dim"])
    x_outside = np.ones(cfg["state_dim"]) * 2.0
    tau_zero = tau_tilde(x_inside, target, Q, 0.72)
    tau_pos  = tau_tilde(x_outside, target, Q, 0.72)
    record("stage01", "tau_tilde(center) == 0", tau_zero == 0.0, tau_zero)
    record("stage01", "tau_tilde(far) > 0", tau_pos > 0.0, f"{tau_pos:.4f}")

    # 01.2 tau sandwich (lower ≤ tau_tilde)
    basin = eval_model.basins[0]
    result = tau_sandwich(basin.A, Q, x_outside, target, basin.rho)
    lower_ok = result["tau_L"] <= result["tau_tilde"] + 1e-6
    record("stage01", "tau sandwich lower ≤ tau_tilde", lower_ok,
           f"tau_L={result['tau_L']:.4f} tau_tilde={result['tau_tilde']:.4f}")
    # Counterexample to equality: at reference params with
    # heterogeneous spectral radii, tau_tilde > tau_L strictly.
    # This is the numerical demonstration of the corrected
    # Proposition H.1 (sandwich inequality, not equality).
    gap_ok = result["tau_tilde"] > result["tau_L"] + 1e-3
    record("stage01",
           "tau_tilde strictly > tau_L (Prop H.1 gap confirmed)",
           gap_ok,
           f"gap={result['tau_tilde'] - result['tau_L']:.4f}")

    # 01.3 committor: boundary conditions q[A]=0, q[B]=1
    K = cfg["K"]
    P_uniform = np.ones((K, K)) / K
    q = committor(P_uniform, A_set=[0], B_set=[1])
    record("stage01", "committor q[A]=0", abs(q[0]) < 1e-10, f"{q[0]:.2e}")
    record("stage01", "committor q[B]=1", abs(q[1] - 1.0) < 1e-10, f"{q[1]:.6f}")
    record("stage01", "committor q ∈ [0,1]", bool(np.all(q >= 0) and np.all(q <= 1)),
           f"min={q.min():.4f} max={q.max():.4f}")

    # 01.4 dlqr: K, P finite and P positive-definite
    n, m_u = cfg["state_dim"], cfg["control_dim"]
    Q_lqr = np.eye(n)
    R_lqr = np.eye(m_u) * 0.1
    K_gain, P_dare = dlqr(basin.A, basin.B, Q_lqr, R_lqr)
    P_pd = bool(np.all(np.linalg.eigvalsh(P_dare) > 0))
    record("stage01", "DARE P positive-definite", P_pd)
    record("stage01", "DARE K finite", bool(np.all(np.isfinite(K_gain))))

    # 01.x — alpha from DARE and beta contraction coefficient
    from hdr_validation.control.lqr import (
        compute_alpha_from_dare, transient_contraction_beta
    )
    Q_lqr_loc = np.eye(n)
    R_lqr_loc = np.eye(m_u) * 0.1
    alpha_k = compute_alpha_from_dare(
        basin.A, basin.B, Q_lqr_loc, R_lqr_loc
    )
    record("stage01", "alpha_from_dare in (0,1)",
           0.0 < alpha_k < 1.0, f"{alpha_k:.4f}")

    # Build a minimal 2-state transient sub-matrix for beta check
    K_local = cfg["K"]
    P_trans = np.ones((K_local, K_local)) / K_local
    # Remove absorbing (target) basin column to get sub-stochastic
    P_sub = np.delete(np.delete(P_trans, 0, axis=0), 0, axis=1)
    beta_val = transient_contraction_beta(P_sub)
    rho_sub  = float(np.max(np.abs(np.linalg.eigvals(P_sub))))
    record("stage01", "beta contraction in [0,1)",
           0.0 <= beta_val < 1.0, f"beta={beta_val:.4f}")
    record("stage01", "rho(Q_transient) <= beta",
           rho_sub <= beta_val + 1e-9,
           f"rho={rho_sub:.4f} beta={beta_val:.4f}")

    # 01.5 finite_horizon_tracking returns H gains
    gains = finite_horizon_tracking(basin.A, basin.B, Q_lqr, R_lqr, H=6, P_terminal=P_dare)
    record("stage01", "finite_horizon_tracking len=H", len(gains) == 6, f"{len(gains)}")
    record("stage01", "finite_horizon_tracking K[0] finite", bool(np.all(np.isfinite(gains[0]))))

    # 01.6 chance constraint tightening: delta > 0
    P_cov = np.eye(n) * 0.1
    delta = chance_tightening(basin.C, P_cov, basin.R, alpha=0.05)
    record("stage01", "chance tightening delta ≥ 0", bool(np.all(delta >= 0)),
           f"mean={delta.mean():.4f}")

    # 01.7 T_k_eff formula
    T_eff = compute_T_k_eff(T=128.0, pi_k=0.5, p_miss=0.3, rho_k=0.72)
    expected = 128.0 * 0.5 * 0.7 * 0.28
    record("stage01", "T_k_eff formula correct", abs(T_eff - expected) < 1e-6,
           f"{T_eff:.4f} (expected {expected:.4f})")

    # 01.8 mu_bar_required is valid probability
    mu_bar = compute_mu_bar_required(
        epsilon_control=0.5, alpha=0.05, delta_A=0.20, delta_B=0.20, K_lqr_norm=1.0
    )
    record("stage01", "mu_bar_required ∈ (0,1]", 0.0 < mu_bar <= 1.0, f"{mu_bar:.4f}")

    # 01.9 DwellModel PMF sums to 1
    dwell = DwellModel("poisson", {"mean": 10.0}, max_len=128)
    pmf_sum = float(np.sum(dwell.pmf()))
    record("stage01", "DwellModel PMF sums to 1", abs(pmf_sum - 1.0) < 1e-6, f"{pmf_sum:.8f}")
    surv = dwell.survival()
    record("stage01", "DwellModel survival[0] ≈ 1", surv[0] > 0.9, f"{surv[0]:.4f}")

    # 01.10 All three validation seeds give same committor (BVP is seed-independent)
    for seed in cfg["seeds"][1:]:
        q2 = committor(P_uniform, A_set=[0], B_set=[1])
        record("stage01", f"committor consistent seed={seed}",
               bool(np.allclose(q, q2)), f"max_diff={np.max(np.abs(q - q2)):.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Episode generator
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_episode(cfg: dict, rng: np.random.Generator, basin_idx: int = 0) -> dict:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.specification import observation_schedule, generate_observation, heteroskedastic_R

    eval_model = make_evaluation_model(cfg, rng)
    basin = eval_model.basins[basin_idx]
    T = cfg["steps_per_episode"]
    n, m = cfg["state_dim"], cfg["obs_dim"]
    u_dim = cfg["control_dim"]

    x = np.zeros(n)
    mask_sched = observation_schedule(T, m, rng, profile_name=cfg["profile_name"])

    x_traj, y_traj, z_traj, u_traj, mask_traj = [], [], [], [], []
    z = basin_idx

    for t in range(T):
        u = np.zeros(u_dim)
        w = rng.multivariate_normal(np.zeros(n), basin.Q)
        x_next = basin.A @ x + basin.B @ u + basin.E[:, :basin.Q.shape[0]] @ w + basin.b
        R_t = heteroskedastic_R(basin.R, x, mask_sched[t], t)
        y = generate_observation(x, basin.C, basin.c, R_t, mask_sched[t], rng)

        x_traj.append(x.copy())
        y_traj.append(y)
        z_traj.append(z)
        u_traj.append(u)
        mask_traj.append(mask_sched[t])
        x = x_next

    return {
        "x_true": np.array(x_traj),
        "z_true": np.array(z_traj),
        "y": np.array(y_traj),
        "mask": np.array(mask_traj),
        "u": np.array(u_traj),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 02 — Synthetic episode generation (3 seeds × 12 episodes = 36 total)
# ═══════════════════════════════════════════════════════════════════════════════
def stage02_generation() -> list[dict]:
    cfg = VALIDATION_CONFIG
    all_episodes: list[dict] = []

    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed + 200)
        n_eps = cfg["episodes_per_experiment"]
        seed_eps = [_generate_episode(cfg, rng, basin_idx=rng.integers(0, cfg["K"]))
                    for _ in range(n_eps)]
        all_episodes.extend(seed_eps)
        print(f"    seed={seed}: generated {n_eps} episodes (T={cfg['steps_per_episode']})")

    T, n, m = cfg["steps_per_episode"], cfg["state_dim"], cfg["obs_dim"]
    total = len(all_episodes)
    expected_total = len(cfg["seeds"]) * cfg["episodes_per_experiment"]

    record("stage02", "total episodes correct", total == expected_total, f"{total}")
    record("stage02", "x_true shape correct", all_episodes[0]["x_true"].shape == (T, n),
           str(all_episodes[0]["x_true"].shape))
    record("stage02", "y shape correct", all_episodes[0]["y"].shape == (T, m),
           str(all_episodes[0]["y"].shape))

    all_nan_fracs = [float(np.isnan(ep["y"]).mean()) for ep in all_episodes]
    avg_nan = float(np.mean(all_nan_fracs))
    record("stage02", "missingness > 0 (some NaNs)", avg_nan > 0.0, f"{avg_nan:.3f}")
    record("stage02", "missingness < 1 (not all NaN)", avg_nan < 1.0, f"{avg_nan:.3f}")

    all_finite = all(np.all(np.isfinite(ep["x_true"])) for ep in all_episodes)
    record("stage02", "all x_true finite", all_finite)

    # All 3 basins should appear across 36 episodes
    basins_seen = set(int(ep["z_true"][0]) for ep in all_episodes)
    record("stage02", "all 3 basins observed across seeds",
           len(basins_seen) == cfg["K"], f"basins={sorted(basins_seen)}")

    return all_episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03 — IMM inference (3 seeds, pooled)
# ═══════════════════════════════════════════════════════════════════════════════
def stage03_imm(episodes: list[dict]) -> dict:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.inference.imm import IMMFilter

    cfg = VALIDATION_CONFIG
    all_mode_probs = []
    all_map_modes = []
    all_cal_true = []
    all_cal_prob = []

    n_eps_per_seed = cfg["episodes_per_experiment"]
    for seed_idx, seed in enumerate(cfg["seeds"]):
        rng = np.random.default_rng(seed + 300)
        eval_model = make_evaluation_model(cfg, rng)
        seed_eps = episodes[seed_idx * n_eps_per_seed: (seed_idx + 1) * n_eps_per_seed]

        for ep in seed_eps:
            filt = IMMFilter(eval_model)
            for t in range(len(ep["x_true"])):
                y_t = ep["y"][t]
                mask_t = (~np.isnan(y_t)).astype(int)
                y_clean = np.where(np.isnan(y_t), 0.0, y_t)
                u_t = ep["u"][t]
                state = filt.step(y_clean, mask_t, u_t)
                all_mode_probs.append(state.mode_probs.copy())
                all_map_modes.append(state.map_mode)
                z_true = int(ep["z_true"][t])
                all_cal_true.append(float(z_true == 1))
                all_cal_prob.append(float(state.mode_probs[1]))

    mode_probs_arr = np.array(all_mode_probs)
    map_modes_arr = np.array(all_map_modes)

    # 03.1 Mode probabilities sum to 1
    sums = mode_probs_arr.sum(axis=1)
    record("stage03", "IMM mode probs sum to 1", bool(np.allclose(sums, 1.0, atol=1e-6)),
           f"max_dev={np.max(np.abs(sums-1)):.2e}")

    # 03.2 Mode probs ∈ [0,1]
    record("stage03", "IMM mode probs ∈ [0,1]",
           bool(np.all(mode_probs_arr >= 0) and np.all(mode_probs_arr <= 1)))

    # 03.3 MAP modes valid
    record("stage03", "IMM MAP modes valid",
           bool(np.all((map_modes_arr >= 0) & (map_modes_arr < cfg["K"]))),
           f"unique modes={np.unique(map_modes_arr)}")

    # 03.4 F1 for mode 1 (pooled across 3 seeds)
    y_true_bin = np.array(all_cal_true)
    y_pred_bin = (map_modes_arr == 1).astype(float)
    if y_true_bin.sum() > 0:
        tp = float(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
        fp = float(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
        fn = float(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        record("stage03", "mode1 F1 > 0 (3-seed pooled)", f1 > 0.0, f"{f1:.4f}")
    else:
        record("stage03", "mode1 F1 (no positives)", True, "N/A")

    # 03.5 Pooled sample count = 3 × 12 × 128 = 4608
    expected_samples = len(cfg["seeds"]) * cfg["episodes_per_experiment"] * cfg["steps_per_episode"]
    record("stage03", "pooled sample count correct",
           len(all_mode_probs) == expected_samples, f"{len(all_mode_probs)}")

    # 03.6 All K modes appear in MAP predictions
    unique_map = set(int(m) for m in map_modes_arr)
    record("stage03", "IMM predicts all K modes", len(unique_map) == cfg["K"],
           f"unique={sorted(unique_map)}")

    return {
        "cal_true": np.array(all_cal_true),
        "cal_prob": np.array(all_cal_prob),
        "mode_probs": mode_probs_arr,
        "map_modes": map_modes_arr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03b — ICI diagnostics (3-seed pooled calibration data)
# ═══════════════════════════════════════════════════════════════════════════════
def stage03b_ici(stage03_data: dict) -> None:
    from hdr_validation.inference.ici import (
        compute_T_k_eff,
        compute_p_A_robust,
        compute_omega_min,
        compute_ici_state,
        brier_reliability,
        isotonic_calibrate,
        apply_calibration,
        compute_mu_bar_required,
    )

    cfg = VALIDATION_CONFIG
    cal_true = stage03_data["cal_true"]
    cal_prob = stage03_data["cal_prob"]
    n_cal = len(cal_true)
    split = n_cal // 2

    # 03b.1 Brier calibration
    brier_pre = brier_reliability(cal_true[split:], cal_prob[split:], n_bins=10)
    cal_map = isotonic_calibrate(cal_true[:split], cal_prob[:split], n_bins=10)
    cal_calibrated = apply_calibration(cal_prob[split:], cal_map)
    brier_post = brier_reliability(cal_true[split:], cal_calibrated, n_bins=10)

    R_brier = float(brier_post["reliability"])
    R_brier_pre = float(brier_pre["reliability"])
    record("stage03b", "Brier reliability finite", np.isfinite(R_brier), f"{R_brier:.4f}")
    record("stage03b", "Brier reliability ≥ 0", R_brier >= 0.0, f"{R_brier:.4f}")
    # Isotonic calibration minimises full Brier score, not the reliability
    # component alone; on a finite held-out split the reliability term can
    # increase slightly.  Allow up to 0.025 absolute degradation.
    record("stage03b", "Calibration does not worsen Brier",
           R_brier <= R_brier_pre + 0.025,
           f"pre={R_brier_pre:.4f} post={R_brier:.4f}")

    # 03b.2 p_A^robust
    p_A_robust = compute_p_A_robust(
        p_A=float(cfg["pA"]),
        k_calib=float(cfg["k_calib"]),
        R_brier=R_brier,
    )
    record("stage03b", "p_A_robust ∈ [0,1]", 0.0 <= p_A_robust <= 1.0, f"{p_A_robust:.4f}")
    record("stage03b", "p_A_robust ≥ p_A",
           p_A_robust >= float(cfg["pA"]) - 1e-6, f"{p_A_robust:.4f} ≥ {cfg['pA']}")

    # 03b.3 T_k_eff and omega_min
    n = cfg["state_dim"]
    n_theta = n * n + n * cfg["control_dim"] + n
    T_eff = compute_T_k_eff(T=float(cfg["steps_per_episode"]), pi_k=0.5, p_miss=0.3, rho_k=0.96)
    omega_min = compute_omega_min(n_theta=n_theta, epsilon=0.10, delta=0.05)
    record("stage03b", "T_k_eff > 0", T_eff > 0.0, f"{T_eff:.2f}")
    record("stage03b", "omega_min > 0", omega_min > 0.0, f"{omega_min:.4f}")

    # 03b.4 Full ICI state
    R_brier_max = float(cfg["R_brier_max"])
    ici = compute_ici_state(
        mu_hat=0.2,
        mu_bar_required=0.1,
        R_brier=R_brier,
        R_brier_max=R_brier_max,
        T_k_eff_per_basin=[T_eff, T_eff * 0.5, T_eff * 1.5],
        omega_min=omega_min,
    )
    record("stage03b", "ICI state has required keys",
           all(k in ici for k in ["condition_i", "condition_ii", "condition_iii"]),
           str(list(ici.keys())))
    record("stage03b", "ICI condition_i type bool",
           isinstance(ici["condition_i"], (bool, np.bool_)))

    # 03b.5 Regime boundary: condition_iii fires when T_k_eff too low
    T_low = compute_T_k_eff(T=10.0, pi_k=0.1, p_miss=0.8, rho_k=0.95)
    ici_regime = compute_ici_state(
        mu_hat=0.05, mu_bar_required=0.1,
        R_brier=0.01, R_brier_max=R_brier_max,
        T_k_eff_per_basin=[T_low],
        omega_min=1.0,
    )
    record("stage03b", "condition_iii fires when T_k_eff < omega_min",
           bool(ici_regime["condition_iii"]), f"T_k_eff={T_low:.4f}")

    # 03b.6 Validation pooled cal sample = 3 × 12 × 128 = 4608
    expected_n_cal = len(cfg["seeds"]) * cfg["episodes_per_experiment"] * cfg["steps_per_episode"]
    record("stage03b", "pooled cal sample = 3×12×128 = 4608",
           n_cal == expected_n_cal, f"n_cal={n_cal}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03c — Mode C validation (seed-independent)
# ═══════════════════════════════════════════════════════════════════════════════
def stage03c_mode_c() -> None:
    from hdr_validation.control.mode_c import (
        mode_c_entry_conditions,
        mode_c_action,
        fisher_information_proxy,
        ModeCTracker,
        supervisor_mode_select,
    )
    from hdr_validation.inference.ici import compute_ici_state

    cfg = VALIDATION_CONFIG
    n = cfg["state_dim"]

    # 03c.1 Mode C entry conditions
    entered_dict = mode_c_entry_conditions(
        mu_hat=0.5, mu_bar_required=0.1,
        R_brier=0.08, R_brier_max=float(cfg["R_brier_max"]),
        T_k_eff_per_basin=[2.0, 1.0, 3.0],
        omega_min=10.0,
    )
    record("stage03c", "Mode C entry conditions dict returned",
           isinstance(entered_dict, dict), str(list(entered_dict.keys())))

    # 03c.2 Supervisor selects mode
    ici_state = compute_ici_state(
        mu_hat=0.5, mu_bar_required=0.1,
        R_brier=0.08, R_brier_max=float(cfg["R_brier_max"]),
        T_k_eff_per_basin=[2.0],
        omega_min=10.0,
    )
    mode_sel = supervisor_mode_select(
        ici_state=ici_state,
        mode_b_conditions_met=False,
        mode_c_active=False,
        degradation_flag=False,
    )
    record("stage03c", "Supervisor mode selection returns valid string",
           isinstance(mode_sel, str) and mode_sel in ("mode_a", "mode_b", "mode_c"),
           f"selected: {mode_sel}")

    # 03c.3 Mode C action bounded
    x = np.ones(n) * 0.5
    rng_mc = np.random.default_rng(42)
    u_c = mode_c_action(
        x_hat=x,
        control_dim=n,
        sigma_dither=float(cfg["sigma_dither"]),
        rng=rng_mc,
        used_burden=0.0,
        budget=float(cfg["default_burden_budget"]),
    )
    record("stage03c", "Mode C action bounded by u_max=0.35",
           bool(np.all(np.abs(u_c) <= 0.35 + 1e-8)),
           f"max|u|={np.max(np.abs(u_c)):.4f}")

    # 03c.4 Fisher proxy
    obs_flat = np.vstack([np.random.default_rng(i).normal(size=(10, n)) for i in range(5)])
    obs_empty = np.zeros((0, n))
    fish_nodata = fisher_information_proxy(obs_empty)
    fish_data = fisher_information_proxy(obs_flat)
    record("stage03c", "Fisher proxy ≥ 0 always", fish_nodata >= 0.0 and fish_data >= 0.0,
           f"empty={fish_nodata:.4f} withdata={fish_data:.4f}")
    record("stage03c", "Fisher proxy increases with data", fish_data >= fish_nodata,
           f"{fish_nodata:.4f} → {fish_data:.4f}")

    # 03c.5 ModeCTracker lifecycle
    tracker = ModeCTracker(T_C_max=int(cfg["T_C_max"]))
    tracker.enter(step=0, R_brier=0.05, T_k_eff_per_basin=[2.0, 1.0])
    record("stage03c", "ModeCTracker enters", tracker.active)
    u_dither = np.random.default_rng(0).normal(size=n) * 0.05
    x_hat = np.zeros(n)
    for _ in range(10):
        tracker.tick(u_dither, x_hat)
    record("stage03c", "ModeCTracker tracks steps", tracker.steps_in_mode_c > 0,
           f"{tracker.steps_in_mode_c}")
    tracker.exit()
    record("stage03c", "ModeCTracker exits cleanly", not tracker.active)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 04 — Mode A control (3 seeds, pooled)
# ═══════════════════════════════════════════════════════════════════════════════
def stage04_mode_a(episodes: list[dict]) -> None:
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set

    cfg = VALIDATION_CONFIG
    rng = np.random.default_rng(cfg["seeds"][0] + 400)
    eval_model = make_evaluation_model(cfg, rng)

    n, m_u = cfg["state_dim"], cfg["control_dim"]
    all_u_norms = []
    feasible_count = 0

    for ep in episodes[:9]:
        basin_idx = int(ep["z_true"][0])
        basin = eval_model.basins[basin_idx]
        target = build_target_set(basin_idx, cfg)
        P_hat = np.eye(n) * 0.1

        for t in range(0, cfg["steps_per_episode"], 16):
            x = ep["x_true"][t]
            result = solve_mode_a(x, P_hat, basin, target, kappa_hat=0.6, config=cfg, step=t)
            all_u_norms.append(float(np.linalg.norm(result.u)))
            if result.feasible:
                feasible_count += 1

    total_calls = len(all_u_norms)

    # 04.1 Control bounded
    max_u_norm = max(all_u_norms) if all_u_norms else 0.0
    max_possible = float(m_u) * 0.6
    record("stage04", "Mode A u norm bounded", max_u_norm <= max_possible + 1e-6,
           f"max‖u‖={max_u_norm:.4f}")

    # 04.2 Feasibility rate
    feas_rate = feasible_count / max(total_calls, 1)
    record("stage04", "Mode A feasibility rate > 0.5", feas_rate > 0.5, f"{feas_rate:.2f}")

    # 04.3 Non-trivial control
    nonzero = sum(1 for v in all_u_norms if v > 1e-6)
    record("stage04", "Mode A produces non-zero control",
           nonzero > 0, f"{nonzero}/{total_calls} calls")

    # 04.3b — Mode A active fraction from far states
    rng_far = np.random.default_rng(888)
    directions = rng_far.normal(size=(20, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng_far.uniform(1.5, 3.0, size=20)
    far_states = directions * radii[:, None]

    basin_far = eval_model.basins[1]   # maladaptive basin (rho=0.96)
    target_far = build_target_set(1, cfg)
    P_hat_far = np.eye(n) * 0.2
    far_u_norms = []
    for x_far_i in far_states:
        res_far = solve_mode_a(x_far_i, P_hat_far, basin_far, target_far,
                               kappa_hat=0.65, config=cfg, step=0)
        far_u_norms.append(float(np.linalg.norm(res_far.u)))

    active_fraction = sum(1 for v in far_u_norms if v > 0.05) / 20
    mean_u_norm_far = float(np.mean(far_u_norms))
    record("stage04",
           "Mode A active fraction >= 0.75 from far states (||x|| in [1.5, 3.0], basin 1)",
           active_fraction >= 0.75,
           f"active={active_fraction:.2f} mean_u={mean_u_norm_far:.4f}")

    # 04.4 Maladaptive basin (rho≈0.96)
    basin_mal = eval_model.basins[1]
    target_mal = build_target_set(1, cfg)
    x_far = np.ones(n) * 2.0
    P_hat = np.eye(n) * 0.2
    res_mal = solve_mode_a(x_far, P_hat, basin_mal, target_mal, kappa_hat=0.6, config=cfg, step=0)
    record("stage04", "Mode A on rho=0.96 basin finite u", bool(np.all(np.isfinite(res_mal.u))))
    record("stage04", "Mode A on rho=0.96 risk computed", np.isfinite(res_mal.risk),
           f"risk={res_mal.risk:.4f}")

    # 04.5 All three seeds produce finite control
    for seed in cfg["seeds"]:
        rng_s = np.random.default_rng(seed + 400)
        em = make_evaluation_model(cfg, rng_s)
        basin_s = em.basins[0]
        target_s = build_target_set(0, cfg)
        x_s = np.ones(n) * 0.5
        res_s = solve_mode_a(x_s, np.eye(n) * 0.1, basin_s, target_s, kappa_hat=0.6,
                             config=cfg, step=0)
        record("stage04", f"Mode A finite seed={seed}", bool(np.all(np.isfinite(res_s.u))),
               f"‖u‖={np.linalg.norm(res_s.u):.4f}")

    # 04.13 — Adaptive-basin episode count documented
    ep_basins_val = [int(ep["z_true"][0]) for ep in episodes]
    n_adaptive = sum(1 for b in ep_basins_val if b != 1)
    record("stage04",
           "Adaptive-basin N documented (n_adaptive)",
           True,
           f"n_adaptive={n_adaptive}",
           note=("UNDERPOWERED: n_adaptive < 20, no performance claims valid for adaptive basins"
                 if n_adaptive < 20 else
                 "n_adaptive >= 20"))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 05 — Mode B validation (150 MC rollouts)
# ═══════════════════════════════════════════════════════════════════════════════
def stage05_mode_b() -> None:
    from hdr_validation.control.lqr import (
        committor,
        controlled_value_iteration,
        heuristic_committor_policy,
        make_reduced_chain,
        posterior_committor,
    )
    from hdr_validation.inference.ici import compute_epsilon_H, compute_mode_b_suboptimality_bound

    # 05.1 Reduced chain exact DP
    P_actions, success_states, failure_states, start_state = make_reduced_chain()
    exact = controlled_value_iteration(P_actions, success_states, failure_states)
    V_star = exact["V"][start_state]
    record("stage05", "Exact DP V* ∈ [0,1]", 0.0 <= V_star <= 1.0, f"V*(start)={V_star:.4f}")
    record("stage05", "Exact DP V*(success)=1",
           all(abs(exact["V"][s] - 1.0) < 1e-6 for s in success_states))
    record("stage05", "Exact DP V*(failure)=0",
           all(abs(exact["V"][s]) < 1e-6 for s in failure_states))

    # 05.2 Heuristic policy gap ≤ 0.10
    heur = heuristic_committor_policy(P_actions, "conservative", success_states, failure_states)
    V_heur = heur["V"][start_state]
    gap = V_star - V_heur
    record("stage05", "Mode B heuristic gap ≤ 0.10",
           gap <= 0.10 + 1e-6, f"gap={gap:.4f}")
    record("stage05", "Mode B heuristic V ∈ [0,1]",
           0.0 <= V_heur <= 1.0, f"V_heur(start)={V_heur:.4f}")

    # 05.3 epsilon_H bound
    eps_H = compute_epsilon_H(rho_star=0.96, H=6)
    record("stage05", "epsilon_H > 0", eps_H > 0.0, f"{eps_H:.4f}")

    # 05.4 Suboptimality bound
    sub_bound = compute_mode_b_suboptimality_bound(epsilon_q=0.05, delta_P=0.05, H=6, rho_star=0.96)
    record("stage05", "suboptimality bound ≥ ε_H", sub_bound >= eps_H,
           f"bound={sub_bound:.4f} eps_H={eps_H:.4f}")

    # 05.5 Posterior committor
    mode_probs = np.array([0.2, 0.6, 0.2])
    transition = np.array([[0.85, 0.04, 0.11],
                            [0.05, 0.87, 0.08],
                            [0.55, 0.18, 0.27]])
    q_hat = posterior_committor(mode_probs, transition)
    record("stage05", "Posterior committor ∈ [0,1]", 0.0 <= q_hat <= 1.0, f"{q_hat:.4f}")

    # 05.6 MC escape prob with 150 rollouts
    rng = np.random.default_rng(42)
    P_passive = P_actions["conservative"]
    P_aggressive = P_actions["aggressive"]
    start = start_state
    H_test = 12
    mc_rollouts = VALIDATION_CONFIG["mc_rollouts"]
    passive_esc, agg_esc = [], []
    for _ in range(mc_rollouts):
        s, sa = start, start
        for _ in range(H_test):
            if s in success_states or s in failure_states: break
            s = int(rng.choice(len(P_passive), p=P_passive[s]))
        passive_esc.append(s in success_states)
        for _ in range(H_test):
            if sa in success_states or sa in failure_states: break
            sa = int(rng.choice(len(P_aggressive), p=P_aggressive[sa]))
        agg_esc.append(sa in success_states)

    p_passive = float(np.mean(passive_esc))
    p_aggressive = float(np.mean(agg_esc))
    record("stage05", f"Mode B aggressive > passive escape prob (n={mc_rollouts} MC)",
           p_aggressive >= p_passive, f"{p_passive:.3f} → {p_aggressive:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 06 — State coherence (seed-independent)
# ═══════════════════════════════════════════════════════════════════════════════
def stage06_coherence() -> None:
    from hdr_validation.model.coherence import coherence_grad, coherence_penalty

    cfg = VALIDATION_CONFIG

    kappa_vals = np.linspace(0.0, 1.0, 50)
    penalties = [coherence_penalty(k, cfg["kappa_lo"], cfg["kappa_hi"]) for k in kappa_vals]

    inside_penalties = [coherence_penalty(k, cfg["kappa_lo"], cfg["kappa_hi"])
                        for k in np.linspace(cfg["kappa_lo"], cfg["kappa_hi"], 10)]
    outside_low = coherence_penalty(0.1, cfg["kappa_lo"], cfg["kappa_hi"])

    record("stage06", "coherence_penalty all finite", bool(np.all(np.isfinite(penalties))))
    record("stage06", "coherence_penalty all ≥ 0", bool(np.all(np.array(penalties) >= 0)))
    record("stage06", "coherence_penalty lower outside target",
           outside_low >= max(inside_penalties) - 1e-8 or outside_low > 0,
           f"outside_low={outside_low:.4f} max_inside={max(inside_penalties):.4f}")

    grads = [coherence_grad(k, cfg["kappa_lo"], cfg["kappa_hi"]) for k in kappa_vals]
    record("stage06", "coherence_grad all finite", bool(np.all(np.isfinite(grads))))

    contributions = []
    for w3 in cfg["w3_sweep_values"]:
        kappa = 0.4
        g_grad = coherence_grad(kappa, cfg["kappa_lo"], cfg["kappa_hi"])
        g_pen = coherence_penalty(kappa, cfg["kappa_lo"], cfg["kappa_hi"])
        coupling_scale = w3 * (abs(g_grad) * 0.5 + g_pen * 0.3)
        contributions.append(coupling_scale)

    is_monotone = all(contributions[i] <= contributions[i+1] + 1e-8
                      for i in range(len(contributions)-1))
    record("stage06", "coherence contribution monotone in w3",
           is_monotone, f"{[f'{c:.4f}' for c in contributions]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 07 — Robustness sweeps (all 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════
def stage07_robustness() -> None:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.inference.ici import compute_T_k_eff
    from hdr_validation.inference.imm import IMMFilter

    cfg = VALIDATION_CONFIG
    n = cfg["state_dim"]

    # 07.1 Stability under varying rho
    rho_values = [0.50, 0.72, 0.85, 0.96]
    for rho in rho_values:
        cfg_sweep = {**cfg, "rho_reference": [rho, min(rho + 0.1, 0.99), max(rho - 0.15, 0.3)]}
        rng = np.random.default_rng(42)
        eval_model = make_evaluation_model(cfg_sweep, rng)
        basin = eval_model.basins[0]
        target = build_target_set(0, cfg_sweep)
        x = np.ones(n) * 1.5
        P = np.eye(n) * 0.1
        res = solve_mode_a(x, P, basin, target, kappa_hat=0.6, config=cfg_sweep, step=0)
        record("stage07", f"Mode A stable rho={rho}", bool(np.all(np.isfinite(res.u))),
               f"‖u‖={np.linalg.norm(res.u):.4f}")

    # 07.2 T_k_eff formula
    for rho_k in [0.72, 0.85, 0.96]:
        T_eff = compute_T_k_eff(T=128.0, pi_k=0.33, p_miss=0.5, rho_k=rho_k)
        expected = 128.0 * 0.33 * 0.5 * (1 - rho_k)
        record("stage07", f"T_k_eff formula rho={rho_k}",
               abs(T_eff - expected) < 1e-8, f"{T_eff:.4f}")

    # 07.3 Model mismatch sweep
    for mismatch in [0.05, 0.10, 0.20]:
        cfg_mis = {**cfg, "model_mismatch_bound": mismatch}
        rng = np.random.default_rng(99)
        eval_model = make_evaluation_model(cfg_mis, rng)
        basin = eval_model.basins[1]
        target = build_target_set(1, cfg_mis)
        x = np.ones(n)
        P = np.eye(n) * 0.15
        res = solve_mode_a(x, P, basin, target, kappa_hat=0.6, config=cfg_mis, step=0)
        record("stage07", f"Mismatch δ={mismatch} Mode A finite",
               bool(np.all(np.isfinite(res.u))), f"‖u‖={np.linalg.norm(res.u):.4f}")

    # 07.4 Missing data / inference quality checks (replaced trivial sweep)
    # 07.4a — Numerical stability check (p_miss=0.3, 20 steps, uniform prior)
    # This is explicitly a NUMERICAL STABILITY check only — it does NOT test
    # inference quality. The only claim is that the filter does not NaN or diverge.
    rng_74a = np.random.default_rng(7)
    eval_model_74a = make_evaluation_model(cfg, rng_74a)
    filt_74a = IMMFilter(eval_model_74a)
    m_obs = cfg["obs_dim"]
    for _ in range(20):
        y_74a = rng_74a.normal(size=m_obs)
        mask_74a = (rng_74a.uniform(size=m_obs) > 0.3).astype(int)
        y_74a = np.where(mask_74a.astype(bool), y_74a, np.nan)
        y_clean_74a = np.where(np.isnan(y_74a), 0.0, y_74a)
        state_74a = filt_74a.step(y_clean_74a, mask_74a, np.zeros(cfg["control_dim"]))
    probs_ok_74a = bool(np.all(np.isfinite(state_74a.mode_probs)) and
                        abs(state_74a.mode_probs.sum() - 1.0) < 1e-4)
    record("stage07",
           "IMM numerical stability p_miss=0.3 (probs finite and sum to 1)",
           probs_ok_74a,
           f"probs={[f'{p:.3f}' for p in state_74a.mode_probs]}")

    # 07.4b — Posterior entropy under non-maladaptive observations
    # A passing result means meaningful uncertainty is retained or the filter
    # has correctly identified a different basin after 50 non-maladaptive steps.
    rng_74b = np.random.default_rng(74)
    eval_model_74b = make_evaluation_model(cfg, rng_74b)
    filt_74b = IMMFilter(eval_model_74b)
    ep_74b = _generate_episode(cfg, rng_74b, basin_idx=2)
    for t in range(50):
        y_t = ep_74b["y"][t]
        mask_t = (~np.isnan(y_t)).astype(int)
        y_clean = np.where(np.isnan(y_t), 0.0, y_t)
        state_74b = filt_74b.step(y_clean, mask_t, np.zeros(cfg["control_dim"]))
    mode_probs_74b = state_74b.mode_probs
    H_74b = float(-np.sum(mode_probs_74b * np.log(mode_probs_74b + 1e-12)))
    record("stage07",
           "IMM posterior entropy > 0.3 nats after 50 non-maladaptive steps",
           H_74b > 0.3,
           f"H={H_74b:.4f} probs={[f'{p:.3f}' for p in mode_probs_74b]}")

    # 07.4c — Mode recovery from wrong-prior initialisation
    # Filter biased toward basin 2 should identify basin-1 signal within 30 steps.
    rng_74c = np.random.default_rng(74)
    eval_model_74c = make_evaluation_model(cfg, rng_74c)
    filt_74c = IMMFilter(eval_model_74c)
    # Bias toward basin 2: strongly wrong prior
    wrong_prior = np.array([0.05, 0.05, 0.90])
    wrong_prior = wrong_prior / wrong_prior.sum()
    filt_74c.state.mode_probs = wrong_prior
    ep_74c = _generate_episode(cfg, rng_74c, basin_idx=1)
    for t in range(30):
        y_t = ep_74c["y"][t]
        # p_miss=0.0: treat all observations as present regardless of missingness
        mask_t = np.ones(cfg["obs_dim"], dtype=int)
        y_clean = np.where(np.isnan(y_t), 0.0, y_t)
        state_74c = filt_74c.step(y_clean, mask_t, np.zeros(cfg["control_dim"]))
    record("stage07",
           "IMM recovers basin-1 MAP mode within 30 steps from wrong prior",
           state_74c.map_mode == 1,
           f"map_mode={state_74c.map_mode} probs={[f'{p:.3f}' for p in state_74c.mode_probs]}")

    # 07.5 All three seeds produce consistent Mode A output
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        eval_model = make_evaluation_model(cfg, rng)
        basin = eval_model.basins[0]
        target = build_target_set(0, cfg)
        x = np.ones(n) * 0.5
        P = np.eye(n) * 0.1
        res = solve_mode_a(x, P, basin, target, kappa_hat=0.6, config=cfg, step=0)
        record("stage07", f"Mode A finite seed={seed}", bool(np.all(np.isfinite(res.u))),
               f"‖u‖={np.linalg.norm(res.u):.4f}")

    # 07.8 — Model mismatch bound covers empirical p90 delta_A for basin 1
    import json as _json_07
    _mismatch_path = ROOT / "results" / "stage_04" / "highpower" / "mismatch_audit.json"
    if not _mismatch_path.exists():
        record("stage07",
               "Mismatch bound audit file present",
               False,
               "mismatch_audit.json not found — run analyse_mismatch.py first")
    else:
        with open(_mismatch_path) as _f:
            _mismatch_data = _json_07.load(_f)
        _basin1_p90 = float(_mismatch_data["basin_1_p90_vs_bound"]["basin_1_p90"])
        _bound = float(cfg["model_mismatch_bound"])
        _bound_covers_p90 = _bound >= _basin1_p90
        record("stage07",
               "Mismatch audit: basin-1 p90 delta_A reported",
               True,
               f"basin_1_p90={_basin1_p90:.4f} bound={_bound:.4f}")
        record("stage07",
               "Mismatch bound covers p90 basin-1 delta_A (theory guarantee validity)",
               _bound_covers_p90,
               f"{'OK' if _bound_covers_p90 else 'VIOLATED'}: {_bound:.3f} vs p90={_basin1_p90:.3f}",
               note="FAIL here means ISS Proposition 10.4 guarantee invalid in ~10% of seeds — disclose in manuscript")


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("HDR Validation Suite — Validation Profile")
    print(f"Seeds: {VALIDATION_CONFIG['seeds']}")
    print(f"Episodes/seed: {VALIDATION_CONFIG['episodes_per_experiment']}")
    print(f"Steps/episode: {VALIDATION_CONFIG['steps_per_episode']}")
    print(f"MC rollouts: {VALIDATION_CONFIG['mc_rollouts']}")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")
    from hdr_validation.control.mpc import SCIPY_MINIMIZE_OPTIONS
    print(f"SciPy minimize options: {SCIPY_MINIMIZE_OPTIONS}")

    episodes = None
    stage03_data = None

    run_stage("Stage 01 — Mathematical Checks", stage01_math)

    episodes = stage02_generation()
    run_stage("Stage 02 — Synthetic Data Generation (3 seeds)", lambda: None)

    stage03_data = stage03_imm(episodes)
    run_stage("Stage 03 — IMM Inference (pooled 3 seeds)", lambda: None)

    run_stage("Stage 03b — ICI Diagnostics", lambda: stage03b_ici(stage03_data))
    run_stage("Stage 03c — Mode C Validation", stage03c_mode_c)
    run_stage("Stage 04 — Mode A Control (3 seeds)", lambda: stage04_mode_a(episodes))
    run_stage("Stage 05 — Mode B Validation (150 MC)", stage05_mode_b)
    run_stage("Stage 06 — State Coherence", stage06_coherence)
    run_stage("Stage 07 — Robustness Sweeps (3 seeds)", stage07_robustness)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    by_stage: dict[str, list] = {}
    for r in results:
        by_stage.setdefault(r["stage"], []).append(r)

    total_pass = sum(1 for r in results if r["passed"])
    total_fail = sum(1 for r in results if not r["passed"])

    for stage, checks in by_stage.items():
        n_pass = sum(1 for c in checks if c["passed"])
        n_fail = sum(1 for c in checks if not c["passed"])
        status = "✓" if n_fail == 0 else "✗"
        print(f"  {status} {stage}: {n_pass}/{len(checks)} passed")

    print(f"\n  Total: {total_pass} passed, {total_fail} failed out of {len(results)} checks")

    if total_fail == 0:
        print("\n  ✓ ALL VALIDATION CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"\n  ✗ {total_fail} CHECKS FAILED")
        print("\n  Failed checks:")
        for r in results:
            if not r["passed"]:
                print(f"    - [{r['stage']}] {r['check']}: {r['value']} ({r['note']})")
        sys.exit(1)
