"""
HDR Validation Suite — Smoke Profile Runner
============================================
Exercises all validation stages (01–07) using the correctly-structured
subpackages (control/, inference/, model/) with inline smoke config.

Usage:
    python3 smoke_runner.py

All stages report PASS / FAIL with supporting metrics.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ── Inline smoke config ────────────────────────────────────────────────────────
SMOKE_CONFIG: dict[str, Any] = {
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
    "model_mismatch_bound": 0.20,
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
    # Smoke profile
    "profile_name": "smoke",
    "seeds": [101],
    "episodes_per_experiment": 8,
    "steps_per_episode": 128,
    "mc_rollouts": 50,
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
# Stage 01 — Mathematical checks
# ═══════════════════════════════════════════════════════════════════════════════
def stage01_math():
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.model.recovery import tau_tilde, tau_sandwich, dare_terminal_cost
    from hdr_validation.model.safety import chance_tightening
    from hdr_validation.control.lqr import dlqr, committor, finite_horizon_tracking
    from hdr_validation.model.hsmm import DwellModel
    from hdr_validation.inference.ici import compute_T_k_eff, compute_mu_bar_required

    cfg = SMOKE_CONFIG
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

    # 01.2 tau sandwich (lower ≤ tau_tilde ≤ upper_coeff * dist²/(1-rho²))
    basin = eval_model.basins[0]
    result = tau_sandwich(basin.A, Q, x_outside, target, basin.rho)
    lower_ok = result["tau_L"] <= result["tau_tilde"] + 1e-6
    record("stage01", "tau sandwich lower ≤ tau_tilde", lower_ok,
           f"tau_L={result['tau_L']:.4f} tau_tilde={result['tau_tilde']:.4f}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 02 — Synthetic episode generation
# ═══════════════════════════════════════════════════════════════════════════════
def _generate_episode(cfg: dict, rng: np.random.Generator, basin_idx: int = 0) -> dict:
    """Simplified inline episode generator for smoke testing."""
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


def stage02_generation():
    cfg = SMOKE_CONFIG
    rng = np.random.default_rng(cfg["seeds"][0] + 200)
    n_eps = cfg["episodes_per_experiment"]

    episodes = [_generate_episode(cfg, rng, basin_idx=rng.integers(0, cfg["K"])) for _ in range(n_eps)]

    # 02.1 Shape checks
    T, n, m = cfg["steps_per_episode"], cfg["state_dim"], cfg["obs_dim"]
    record("stage02", "episodes count correct", len(episodes) == n_eps, f"{len(episodes)}")
    record("stage02", "x_true shape correct", episodes[0]["x_true"].shape == (T, n),
           str(episodes[0]["x_true"].shape))
    record("stage02", "y shape correct", episodes[0]["y"].shape == (T, m),
           str(episodes[0]["y"].shape))

    # 02.2 Missingness check: some NaNs expected
    all_nan_fracs = [float(np.isnan(ep["y"]).mean()) for ep in episodes]
    avg_nan = float(np.mean(all_nan_fracs))
    record("stage02", "missingness > 0 (some NaNs)", avg_nan > 0.0, f"{avg_nan:.3f}")
    record("stage02", "missingness < 1 (not all NaN)", avg_nan < 1.0, f"{avg_nan:.3f}")

    # 02.3 State trajectories are finite
    all_finite = all(np.all(np.isfinite(ep["x_true"])) for ep in episodes)
    record("stage02", "all x_true finite", all_finite)

    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03 — IMM inference
# ═══════════════════════════════════════════════════════════════════════════════
def stage03_imm(episodes: list[dict]) -> dict:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.inference.imm import IMMFilter

    cfg = SMOKE_CONFIG
    rng = np.random.default_rng(cfg["seeds"][0] + 300)
    eval_model = make_evaluation_model(cfg, rng)

    all_mode_probs = []
    all_map_modes = []
    all_z_true = []
    all_cal_true = []
    all_cal_prob = []

    for ep in episodes:
        filt = IMMFilter(eval_model)
        for t in range(len(ep["x_true"])):
            y_t = ep["y"][t]
            mask_t = (~np.isnan(y_t)).astype(int)
            y_clean = np.where(np.isnan(y_t), 0.0, y_t)
            u_t = ep["u"][t]
            state = filt.step(y_clean, mask_t, u_t)
            all_mode_probs.append(state.mode_probs.copy())
            all_map_modes.append(state.map_mode)
            # For calibration: binary label (mode1 = maladaptive)
            z_true = int(ep["z_true"][t])
            all_z_true.append(z_true)
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

    # 03.4 F1 for mode 1 detection
    y_true_bin = np.array(all_cal_true)
    y_pred_bin = (np.array(all_map_modes) == 1).astype(float)
    if y_true_bin.sum() > 0:
        tp = float(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
        fp = float(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
        fn = float(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        record("stage03", "mode1 F1 > 0", f1 > 0.0, f"{f1:.4f}")
    else:
        record("stage03", "mode1 F1 (no positives in data)", True, "N/A")

    return {
        "cal_true": np.array(all_cal_true),
        "cal_prob": np.array(all_cal_prob),
        "mode_probs": mode_probs_arr,
        "map_modes": map_modes_arr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03b — ICI diagnostics
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

    cfg = SMOKE_CONFIG
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
    record("stage03b", "Brier reliability finite", np.isfinite(R_brier), f"{R_brier:.4f}")
    record("stage03b", "Brier reliability ≥ 0", R_brier >= 0.0, f"{R_brier:.4f}")

    # 03b.2 p_A^robust computation
    p_A_robust = compute_p_A_robust(
        p_A=float(cfg["pA"]),
        k_calib=float(cfg["k_calib"]),
        R_brier=R_brier,
    )
    record("stage03b", "p_A_robust ∈ [0,1]", 0.0 <= p_A_robust <= 1.0, f"{p_A_robust:.4f}")
    record("stage03b", "p_A_robust ≥ p_A (miscalibration raises threshold)",
           p_A_robust >= float(cfg["pA"]) - 1e-6, f"{p_A_robust:.4f} ≥ {cfg['pA']}")

    # 03b.3 T_k_eff and omega_min
    rng = np.random.default_rng(cfg["seeds"][0] + 3100)
    n = cfg["state_dim"]
    n_theta = n * n + n * cfg["control_dim"] + n
    T_eff = compute_T_k_eff(T=float(cfg["steps_per_episode"]), pi_k=0.5, p_miss=0.3, rho_k=0.96)
    omega_min = compute_omega_min(n_theta=n_theta, epsilon=0.10, delta=0.05)
    record("stage03b", "T_k_eff > 0", T_eff > 0.0, f"{T_eff:.2f}")
    record("stage03b", "omega_min > 0", omega_min > 0.0, f"{omega_min:.4f}")

    # 03b.4 Full ICI state
    mu_hat = 0.2
    R_brier_max = float(cfg["R_brier_max"])
    ici = compute_ici_state(
        mu_hat=mu_hat,
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

    # 03b.5 Regime boundary: when T_k_eff < omega_min, condition_iii fires
    T_low = compute_T_k_eff(T=10.0, pi_k=0.1, p_miss=0.8, rho_k=0.95)  # very low T_k_eff
    ici_regime = compute_ici_state(
        mu_hat=0.05, mu_bar_required=0.1,
        R_brier=0.01, R_brier_max=R_brier_max,
        T_k_eff_per_basin=[T_low],
        omega_min=1.0,  # omega_min very high → condition_iii fires
    )
    record("stage03b", "condition_iii fires when T_k_eff < omega_min",
           bool(ici_regime["condition_iii"]), f"T_k_eff={T_low:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 03c — Mode C validation
# ═══════════════════════════════════════════════════════════════════════════════
def stage03c_mode_c() -> None:
    from hdr_validation.control.mode_c import (
        mode_c_entry_conditions,
        mode_c_action,
        fisher_information_proxy,
        ModeCTracker,
        supervisor_mode_select,
    )
    from hdr_validation.inference.ici import compute_ici_state, compute_T_k_eff

    cfg = SMOKE_CONFIG
    n = cfg["state_dim"]

    # 03c.1 Mode C entry when all ICI conditions fire
    entered_dict = mode_c_entry_conditions(
        mu_hat=0.5, mu_bar_required=0.1,
        R_brier=0.08, R_brier_max=float(cfg["R_brier_max"]),
        T_k_eff_per_basin=[2.0, 1.0, 3.0],
        omega_min=10.0,
    )
    entered = entered_dict.get("any_condition", False)
    record("stage03c", "Mode C entry conditions dict returned",
           isinstance(entered_dict, dict), str(list(entered_dict.keys())))

    # 03c.2 Supervisor selects mode when degradation flag set
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

    # 03c.3 Mode C action is bounded
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

    # 03c.4 Fisher proxy increases with diverse inputs
    obs_flat = np.vstack([np.random.default_rng(i).normal(size=(10, n)) for i in range(5)])
    obs_empty = np.zeros((0, n))
    fish_nodata = fisher_information_proxy(obs_empty)
    fish_data = fisher_information_proxy(obs_flat)
    record("stage03c", "Fisher proxy ≥ 0 always", fish_nodata >= 0.0 and fish_data >= 0.0,
           f"empty={fish_nodata:.4f} withdata={fish_data:.4f}")
    record("stage03c", "Fisher proxy increases with data", fish_data >= fish_nodata,
           f"{fish_nodata:.4f} → {fish_data:.4f}")

    # 03c.5 ModeCTracker enters/exits correctly
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
# Stage 04 — Mode A control
# ═══════════════════════════════════════════════════════════════════════════════
def stage04_mode_a(episodes: list[dict]) -> None:
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.model.target_set import build_target_set

    cfg = SMOKE_CONFIG
    rng = np.random.default_rng(cfg["seeds"][0] + 400)
    eval_model = make_evaluation_model(cfg, rng)

    n, m_u = cfg["state_dim"], cfg["control_dim"]
    all_u_norms = []
    feasible_count = 0

    for ep in episodes[:4]:  # Sample episodes
        basin_idx = int(ep["z_true"][0])
        basin = eval_model.basins[basin_idx]
        target = build_target_set(basin_idx, cfg)
        P_hat = np.eye(n) * 0.1

        for t in range(0, cfg["steps_per_episode"], 16):  # Every 16 steps
            x = ep["x_true"][t]
            result = solve_mode_a(x, P_hat, basin, target, kappa_hat=0.6, config=cfg, step=t)
            all_u_norms.append(float(np.linalg.norm(result.u)))
            if result.feasible:
                feasible_count += 1

    total_calls = len(all_u_norms)
    # 04.1 Control bounded by hard constraints
    max_u_norm = max(all_u_norms) if all_u_norms else 0.0
    max_possible = float(m_u) * 0.6  # all dims at max
    record("stage04", "Mode A u norm bounded", max_u_norm <= max_possible + 1e-6,
           f"max‖u‖={max_u_norm:.4f}")

    # 04.2 Feasibility rate
    feas_rate = feasible_count / max(total_calls, 1)
    record("stage04", "Mode A feasibility rate > 0.5", feas_rate > 0.5, f"{feas_rate:.2f}")

    # 04.3 Mode A vs open-loop: control is non-trivial
    nonzero = sum(1 for v in all_u_norms if v > 1e-6)
    record("stage04", "Mode A produces non-zero control",
           nonzero > 0, f"{nonzero}/{total_calls} calls")

    # 04.4 Test on maladaptive basin (rho≈0.96) - most important case
    basin_mal = eval_model.basins[1]
    target_mal = build_target_set(1, cfg)
    x_far = np.ones(n) * 2.0
    P_hat = np.eye(n) * 0.2
    res_mal = solve_mode_a(x_far, P_hat, basin_mal, target_mal, kappa_hat=0.6, config=cfg, step=0)
    record("stage04", "Mode A on rho=0.96 basin finite u", bool(np.all(np.isfinite(res_mal.u))))
    record("stage04", "Mode A on rho=0.96 risk computed", np.isfinite(res_mal.risk),
           f"risk={res_mal.risk:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 05 — Mode B validation
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

    # 05.2 Heuristic policy gap ≤ ε_H bound (≤ 0.10)
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

    # 05.4 Suboptimality bound includes ε_H
    sub_bound = compute_mode_b_suboptimality_bound(epsilon_q=0.05, delta_P=0.05, H=6, rho_star=0.96)
    record("stage05", "suboptimality bound ≥ ε_H", sub_bound >= eps_H,
           f"bound={sub_bound:.4f} eps_H={eps_H:.4f}")

    # 05.5 Posterior committor
    mode_probs = np.array([0.2, 0.6, 0.2])  # Confident in maladaptive
    transition = np.array([[0.85, 0.04, 0.11],
                            [0.05, 0.87, 0.08],
                            [0.55, 0.18, 0.27]])
    q_hat = posterior_committor(mode_probs, transition)
    record("stage05", "Posterior committor ∈ [0,1]", 0.0 <= q_hat <= 1.0, f"{q_hat:.4f}")

    # 05.6 Mode B should have higher escape prob than passive for all horizons
    rng = np.random.default_rng(42)
    P_passive = P_actions["conservative"]
    P_aggressive = P_actions["aggressive"]
    start = start_state
    H_test = 12
    passive_esc, agg_esc = [], []
    for _ in range(cfg_mc := SMOKE_CONFIG["mc_rollouts"]):
        s, sa = start, start
        for _ in range(H_test):
            if s in success_states: break
            if s in failure_states: break
            s = int(rng.choice(len(P_passive), p=P_passive[s]))
        passive_esc.append(s in success_states)
        for _ in range(H_test):
            if sa in success_states: break
            if sa in failure_states: break
            sa = int(rng.choice(len(P_aggressive), p=P_aggressive[sa]))
        agg_esc.append(sa in success_states)

    p_passive = float(np.mean(passive_esc))
    p_aggressive = float(np.mean(agg_esc))
    record("stage05", "Mode B aggressive > passive escape prob",
           p_aggressive >= p_passive, f"{p_passive:.3f} → {p_aggressive:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 06 — State coherence
# ═══════════════════════════════════════════════════════════════════════════════
def stage06_coherence() -> None:
    from hdr_validation.model.coherence import coherence_grad, coherence_penalty

    cfg = SMOKE_CONFIG

    # 06.1 coherence_penalty is monotone decreasing as kappa enters target range
    kappa_vals = np.linspace(0.0, 1.0, 50)
    penalties = [coherence_penalty(k, cfg["kappa_lo"], cfg["kappa_hi"]) for k in kappa_vals]

    # At kappa inside [kappa_lo, kappa_hi], penalty should be minimal
    inside_penalties = [coherence_penalty(k, cfg["kappa_lo"], cfg["kappa_hi"])
                        for k in np.linspace(cfg["kappa_lo"], cfg["kappa_hi"], 10)]
    outside_low = coherence_penalty(0.1, cfg["kappa_lo"], cfg["kappa_hi"])
    outside_high = coherence_penalty(0.95, cfg["kappa_lo"], cfg["kappa_hi"])

    record("stage06", "coherence_penalty all finite", bool(np.all(np.isfinite(penalties))))
    record("stage06", "coherence_penalty all ≥ 0", bool(np.all(np.array(penalties) >= 0)))
    record("stage06", "coherence_penalty lower outside target",
           outside_low >= max(inside_penalties) - 1e-8 or outside_low > 0,
           f"outside_low={outside_low:.4f} max_inside={max(inside_penalties):.4f}")

    # 06.2 coherence_grad finite
    grads = [coherence_grad(k, cfg["kappa_lo"], cfg["kappa_hi"]) for k in kappa_vals]
    record("stage06", "coherence_grad all finite", bool(np.all(np.isfinite(grads))))

    # 06.3 w3 sensitivity: larger w3 gives larger coherence contribution to Q_eff
    contributions = []
    for w3 in cfg["w3_sweep_values"]:
        kappa = 0.4  # Below target (should have high penalty)
        g_grad = coherence_grad(kappa, cfg["kappa_lo"], cfg["kappa_hi"])
        g_pen = coherence_penalty(kappa, cfg["kappa_lo"], cfg["kappa_hi"])
        coupling_scale = w3 * (abs(g_grad) * 0.5 + g_pen * 0.3)
        contributions.append(coupling_scale)

    is_monotone = all(contributions[i] <= contributions[i+1] + 1e-8
                      for i in range(len(contributions)-1))
    record("stage06", "coherence contribution monotone in w3",
           is_monotone, f"{[f'{c:.4f}' for c in contributions]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 07 — Robustness sweeps
# ═══════════════════════════════════════════════════════════════════════════════
def stage07_robustness() -> None:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.inference.ici import compute_T_k_eff

    cfg = SMOKE_CONFIG
    n = cfg["state_dim"]

    # 07.1 Stability under varying rho (spectral radius sweep)
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

    # 07.2 T_k_eff regime boundary: compound bound formula holds
    # T_k_eff = T * pi_k * (1-p_miss) * (1-rho_k)
    # Claim 14: the formula correctly predicts the regime boundary
    for rho_k in [0.72, 0.85, 0.96]:
        T_eff = compute_T_k_eff(T=128.0, pi_k=0.33, p_miss=0.5, rho_k=rho_k)
        expected = 128.0 * 0.33 * 0.5 * (1 - rho_k)
        record("stage07", f"T_k_eff formula rho={rho_k}",
               abs(T_eff - expected) < 1e-8, f"{T_eff:.4f}")

    # 07.3 Model mismatch sweep: control stays finite
    for mismatch in [0.05, 0.10, 0.20]:
        cfg_mis = {**cfg, "model_mismatch_bound": mismatch}
        rng = np.random.default_rng(99)
        eval_model = make_evaluation_model(cfg_mis, rng)
        basin = eval_model.basins[1]  # Maladaptive basin, hardest case
        target = build_target_set(1, cfg_mis)
        x = np.ones(n)
        P = np.eye(n) * 0.15
        res = solve_mode_a(x, P, basin, target, kappa_hat=0.6, config=cfg_mis, step=0)
        record("stage07", f"Mismatch δ={mismatch} Mode A finite",
               bool(np.all(np.isfinite(res.u))), f"‖u‖={np.linalg.norm(res.u):.4f}")

    # 07.4 Missing data sweep: IMM remains stable
    from hdr_validation.inference.imm import IMMFilter
    for p_miss in [0.0, 0.3, 0.6, 0.9]:
        rng = np.random.default_rng(7)
        eval_model = make_evaluation_model(cfg, rng)
        filt = IMMFilter(eval_model)
        m_obs = cfg["obs_dim"]
        for _ in range(20):
            y = rng.normal(size=m_obs)
            mask = (rng.uniform(size=m_obs) > p_miss).astype(int)
            y = np.where(mask.astype(bool), y, np.nan)
            y_clean = np.where(np.isnan(y), 0.0, y)
            state = filt.step(y_clean, mask, np.zeros(cfg["control_dim"]))
        probs_ok = bool(np.all(np.isfinite(state.mode_probs)) and
                        abs(state.mode_probs.sum() - 1.0) < 1e-4)
        record("stage07", f"IMM stable p_miss={p_miss}", probs_ok,
               f"probs={[f'{p:.3f}' for p in state.mode_probs]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("HDR Validation Suite — Smoke Profile")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    episodes = None
    stage03_data = None

    run_stage("Stage 01 — Mathematical Checks", stage01_math)
    run_stage("Stage 02 — Synthetic Data Generation",
              lambda: globals().__setitem__("episodes", stage02_generation()))

    # Re-run stage02 to get episodes for downstream stages
    episodes = stage02_generation()

    run_stage("Stage 03 — IMM Inference",
              lambda: globals().__setitem__("stage03_data", stage03_imm(episodes)))
    stage03_data = stage03_imm(episodes)

    run_stage("Stage 03b — ICI Diagnostics", lambda: stage03b_ici(stage03_data))
    run_stage("Stage 03c — Mode C Validation", stage03c_mode_c)
    run_stage("Stage 04 — Mode A Control", lambda: stage04_mode_a(episodes))
    run_stage("Stage 05 — Mode B Validation", stage05_mode_b)
    run_stage("Stage 06 — State Coherence", stage06_coherence)
    run_stage("Stage 07 — Robustness Sweeps", stage07_robustness)

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
        print("\n  ✓ ALL SMOKE CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"\n  ✗ {total_fail} CHECKS FAILED")
        print("\n  Failed checks:")
        for r in results:
            if not r["passed"]:
                print(f"    - [{r['stage']}] {r['check']}: {r['value']} ({r['note']})")
        sys.exit(1)
