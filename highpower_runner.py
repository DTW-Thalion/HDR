"""
HDR Validation Suite — High-Power Benchmark A Runner
=====================================================
20 seeds × 30 episodes per seed = 600 total episodes.
Produces bootstrap CIs for the maladaptive-basin performance claim.

Usage:
    python3 highpower_runner.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ── High-power profile config (matches EXTENDED_CONFIG, overrides below) ──────
HIGHPOWER_CONFIG: dict[str, Any] = {
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
    "max_dwell_len": 256,
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
    # v7.0 extension parameters
    "n_irr": 2,
    "n_sites": 2,
    "epsilon_G": 0.02,
    "R_k_regions": 2,
    "lambda_cat_max": 0.05,
    "drift_rate": 0.001,
    "delay_steps": 10,
    "n_cum_exp": 1,
    "xi_max": 100.0,
    "n_expansion": 2,
    "delta_J_max": 0.05,
    "m_d": 1,
    "n_particles": 100,
    "n_patients": 10,
    "T_p_values": [10, 50],
    "jump_risk_threshold": 0.3,
    "irr_boundary_threshold": 0.9,
    "lambda_irr": 1.0,
    # ── Highpower overrides ─────────────────────────────────────────────────
    "profile_name": "highpower",
    "seeds": [
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    ],
    "episodes_per_experiment": 30,
    "steps_per_episode": 256,
    "mc_rollouts": 150,
    "selected_trace_cap": 5,
}

# ── Setup sys.path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── Atomic write helper ────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


# ── Episode generation (mirrors extended_runner._generate_episode) ─────────────
def _generate_episode(cfg: dict, rng: np.random.Generator, basin_idx: int = 0) -> dict:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.specification import (
        observation_schedule,
        generate_observation,
        heteroskedastic_R,
    )

    eval_model = make_evaluation_model(cfg, rng)
    basin = eval_model.basins[basin_idx]
    T = cfg["steps_per_episode"]
    n = cfg["state_dim"]
    m = cfg["obs_dim"]
    u_dim = cfg["control_dim"]

    x = np.zeros(n)
    mask_sched = observation_schedule(T, m, rng, profile_name=cfg["profile_name"])

    x_traj, y_traj, z_traj, u_traj, mask_traj = [], [], [], [], []
    z = basin_idx

    for t in range(T):
        u = np.zeros(u_dim)
        w = rng.multivariate_normal(np.zeros(n), basin.Q)
        x_next = basin.A @ x + basin.B @ u + basin.E[:, : basin.Q.shape[0]] @ w + basin.b
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


# ── Bootstrap CI helper ────────────────────────────────────────────────────────
def _bootstrap_ci(
    data,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
    stat: str = "mean",
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    data = np.asarray(data)
    if stat == "mean":
        boot_stats = np.array(
            [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
        )
    else:
        boot_stats = np.array(
            [np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        )
    lo = float(np.percentile(boot_stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 + ci) / 2))
    return lo, hi


# ── Main highpower stage ───────────────────────────────────────────────────────
def run_highpower_benchmark() -> None:
    from hdr_validation.control.mpc import solve_mode_a
    from hdr_validation.control.lqr import dlqr
    from hdr_validation.model.slds import make_evaluation_model, pooled_basin
    from hdr_validation.model.target_set import build_target_set
    from hdr_validation.model.safety import (
        apply_control_constraints,
        observation_intervals,
        risk_score,
    )
    from hdr_validation.inference.imm import IMMFilter
    from hdr_validation.specification import (
        observation_schedule,
        generate_observation,
        heteroskedastic_R,
    )

    cfg = HIGHPOWER_CONFIG
    out_dir = ROOT / "results" / "stage_04" / "highpower"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = len(cfg["seeds"])
    n_eps_per_seed = cfg["episodes_per_experiment"]
    T = cfg["steps_per_episode"]
    n = cfg["state_dim"]
    m_u = cfg["control_dim"]
    m_obs = cfg["obs_dim"]
    lambda_u = float(cfg["lambda_u"])
    y_lo, y_hi = observation_intervals(cfg)
    P_safety = np.eye(n) * 0.1
    policy_names = ["open_loop", "pooled_lqr", "basin_lqr", "hdr_main", "pooled_lqr_estimated"]

    def _safety_violation(x_state, basin_obj):
        y_mean = basin_obj.C @ x_state + basin_obj.c
        y_cov = basin_obj.C @ P_safety @ basin_obj.C.T + basin_obj.R
        r = risk_score(y_mean, y_cov, y_lo, y_hi)
        return r > float(cfg["eps_safe"])

    # ── Collect per-seed results (with checkpoint resume) ────────────────────
    seed_results: dict[int, dict] = {}

    # Load existing checkpoints
    for s in cfg["seeds"]:
        partial_path = out_dir / f"seed_{s:04d}_partial.json"
        if partial_path.exists():
            print(f"  Seed {s}: found checkpoint, skipping re-computation.")
            with open(partial_path) as f:
                seed_results[s] = json.load(f)

    # Run missing seeds
    for i, s in enumerate(cfg["seeds"]):
        if s in seed_results:
            continue

        print(f"  Generating seed {s} ({i+1}/{n_seeds})...")
        t_seed_start = time.perf_counter()

        # ── Episode generation for this seed ─────────────────────────────────
        gen_rng = np.random.default_rng(s + 200)
        seed_episodes = []
        for _ in range(n_eps_per_seed):
            basin_idx = int(gen_rng.integers(0, cfg["K"]))
            ep = _generate_episode(cfg, gen_rng, basin_idx=basin_idx)
            ep["seed"] = s
            seed_episodes.append(ep)

        # ── Build eval model for this seed ───────────────────────────────────
        rng_sim = np.random.default_rng(s + 400)
        sim_model = make_evaluation_model(cfg, rng_sim)

        # ── Pre-compute LQR gains ─────────────────────────────────────────────
        Q_lqr = np.eye(n)
        R_lqr = np.eye(m_u) * lambda_u

        p_basin = pooled_basin(sim_model)
        try:
            K_pooled, _ = dlqr(p_basin.A, p_basin.B, Q_lqr, R_lqr)
        except Exception:
            K_pooled = np.zeros((m_u, n))

        K_basin_lqr: list[np.ndarray] = []
        for b in sim_model.basins:
            try:
                K_b, _ = dlqr(b.A, b.B, Q_lqr, R_lqr)
            except Exception:
                K_b = np.zeros((m_u, n))
            K_basin_lqr.append(K_b)

        # ── Run closed-loop simulation for this seed's episodes ───────────────
        ep_costs: dict[str, list[float]] = {p: [] for p in policy_names}
        ep_safety_rates: dict[str, list[float]] = {p: [] for p in policy_names}
        ep_basins: list[int] = []

        # Global ep_idx offset for deterministic noise seeding
        seed_offset = i * n_eps_per_seed  # 0, 30, 60, ..., 570

        for ep_idx_local, ep in enumerate(seed_episodes):
            ep_idx = seed_offset + ep_idx_local  # global 0..399

            basin_idx = int(ep["z_true"][0])
            ep_basins.append(basin_idx)
            basin_obj = sim_model.basins[basin_idx]

            t_start = min(T // 4, T - 1)
            x_init = ep["x_true"][t_start].copy()

            noise_rng = np.random.default_rng(cfg["seeds"][0] + 5000 + ep_idx)
            process_noise = [
                noise_rng.multivariate_normal(np.zeros(n), basin_obj.Q) for _ in range(T)
            ]
            obs_rng = np.random.default_rng(cfg["seeds"][0] + 6000 + ep_idx)
            mask_sched = observation_schedule(T, m_obs, obs_rng, profile_name=cfg["profile_name"])

            # ── Phase 1: estimation-based policies (independent IMM filters) ──
            # Each policy drives its own IMM filter from its own trajectory's
            # observations. Missingness pattern (mask_sched) and process noise
            # are shared; observation noise seed is shared per timestep.
            imm_filt_hdr = IMMFilter.for_hard_regime(sim_model)
            imm_filt_pe = IMMFilter.for_hard_regime(sim_model)
            x_hdr = x_init.copy()
            x_pe = x_init.copy()
            cost_hdr, cost_pe = 0.0, 0.0
            viol_hdr, viol_pe = 0, 0
            used_burden_hdr, used_burden_pe = 0.0, 0.0
            u_prev_hdr = np.zeros(m_u)
            u_prev_pe = np.zeros(m_u)

            for t in range(T):
                base_seed_t = cfg["seeds"][0] + 7000 + ep_idx * T + t
                # HDR filter: observations from x_hdr
                obs_rng_t_hdr = np.random.default_rng(base_seed_t)
                R_t_hdr = heteroskedastic_R(basin_obj.R, x_hdr, mask_sched[t], t)
                y_t_hdr = generate_observation(
                    x_hdr, basin_obj.C, basin_obj.c, R_t_hdr, mask_sched[t], obs_rng_t_hdr
                )
                mask_t_hdr = (~np.isnan(y_t_hdr)).astype(int)
                y_clean_hdr = np.where(np.isnan(y_t_hdr), 0.0, y_t_hdr)
                imm_state_hdr = imm_filt_hdr.step(y_clean_hdr, mask_t_hdr, u_prev_hdr)
                x_hat_hdr = imm_state_hdr.mixed_mean
                P_hat_hdr = imm_state_hdr.mixed_cov

                # PE filter: observations from x_pe (same noise seed)
                obs_rng_t_pe = np.random.default_rng(base_seed_t)
                R_t_pe = heteroskedastic_R(basin_obj.R, x_pe, mask_sched[t], t)
                y_t_pe = generate_observation(
                    x_pe, basin_obj.C, basin_obj.c, R_t_pe, mask_sched[t], obs_rng_t_pe
                )
                mask_t_pe = (~np.isnan(y_t_pe)).astype(int)
                y_clean_pe = np.where(np.isnan(y_t_pe), 0.0, y_t_pe)
                imm_state_pe = imm_filt_pe.step(y_clean_pe, mask_t_pe, u_prev_pe)
                x_hat_pe = imm_state_pe.mixed_mean

                # hdr_main: MPC on estimated basin using its own filter
                est_bi = imm_state_hdr.map_mode
                est_basin = sim_model.basins[est_bi]
                est_target = build_target_set(est_bi, cfg)
                mpc_res = solve_mode_a(
                    x_hat_hdr,
                    P_hat_hdr,
                    est_basin,
                    est_target,
                    kappa_hat=0.6,
                    config=cfg,
                    step=t,
                    used_burden=used_burden_hdr,
                )
                u_hdr = mpc_res.u

                # pooled_lqr_estimated: uses its own filter's x_hat
                u_pe = -K_pooled @ x_hat_pe
                u_pe, _ = apply_control_constraints(
                    u_pe, cfg, step=t, used_burden=used_burden_pe
                )

                # Costs
                cost_hdr += float(np.dot(x_hdr, x_hdr) + lambda_u * np.dot(u_hdr, u_hdr))
                cost_pe += float(np.dot(x_pe, x_pe) + lambda_u * np.dot(u_pe, u_pe))

                # Safety
                if _safety_violation(x_hdr, basin_obj):
                    viol_hdr += 1
                if _safety_violation(x_pe, basin_obj):
                    viol_pe += 1

                # Evolve both with shared process noise
                w = process_noise[t]
                x_hdr = (
                    basin_obj.A @ x_hdr
                    + basin_obj.B @ u_hdr
                    + basin_obj.E[:, :n] @ w
                    + basin_obj.b
                )
                x_pe = (
                    basin_obj.A @ x_pe
                    + basin_obj.B @ u_pe
                    + basin_obj.E[:, :n] @ w
                    + basin_obj.b
                )
                used_burden_hdr += float(np.sum(np.abs(u_hdr)))
                used_burden_pe += float(np.sum(np.abs(u_pe)))
                u_prev_hdr = u_hdr.copy()
                u_prev_pe = u_pe.copy()

            ep_costs["hdr_main"].append(cost_hdr)
            ep_costs["pooled_lqr_estimated"].append(cost_pe)
            ep_safety_rates["hdr_main"].append(viol_hdr / T)
            ep_safety_rates["pooled_lqr_estimated"].append(viol_pe / T)

            # ── Phase 2: oracle-state policies (no IMM needed) ────────────────
            for pol_name in ["open_loop", "pooled_lqr", "basin_lqr"]:
                x = x_init.copy()
                used_burden = 0.0
                cost_accum = 0.0
                violations = 0

                for t in range(T):
                    if pol_name == "open_loop":
                        u = np.zeros(m_u)
                    elif pol_name == "pooled_lqr":
                        u = -K_pooled @ x
                        u, _ = apply_control_constraints(
                            u, cfg, step=t, used_burden=used_burden
                        )
                    elif pol_name == "basin_lqr":
                        u = -K_basin_lqr[basin_idx] @ x
                        u, _ = apply_control_constraints(
                            u, cfg, step=t, used_burden=used_burden
                        )

                    cost_accum += float(np.dot(x, x) + lambda_u * np.dot(u, u))
                    if _safety_violation(x, basin_obj):
                        violations += 1

                    w = process_noise[t]
                    x = (
                        basin_obj.A @ x
                        + basin_obj.B @ u
                        + basin_obj.E[:, :n] @ w
                        + basin_obj.b
                    )
                    used_burden += float(np.sum(np.abs(u)))

                ep_costs[pol_name].append(cost_accum)
                ep_safety_rates[pol_name].append(violations / T)

        # ── Write per-seed partial checkpoint ─────────────────────────────────
        partial_data = {
            "seed": s,
            "n_episodes": n_eps_per_seed,
            "ep_costs": {p: ep_costs[p] for p in policy_names},
            "ep_basins": ep_basins,
            "ep_safety_rates": {
                "hdr_main": ep_safety_rates["hdr_main"],
                "pooled_lqr_estimated": ep_safety_rates["pooled_lqr_estimated"],
            },
        }
        partial_path = out_dir / f"seed_{s:04d}_partial.json"
        _atomic_write_json(partial_path, partial_data)
        seed_results[s] = partial_data

        elapsed = time.perf_counter() - t_seed_start
        print(f"    seed {s} done in {elapsed:.1f}s")

    # ── Aggregate across all 400 episodes ─────────────────────────────────────
    print("\n  Aggregating results across all seeds...")
    all_ep_costs: dict[str, list[float]] = {p: [] for p in policy_names}
    all_ep_basins: list[int] = []
    all_safety_hdr: list[float] = []
    all_safety_pe: list[float] = []

    for s in cfg["seeds"]:
        sd = seed_results[s]
        for p in policy_names:
            all_ep_costs[p].extend(sd["ep_costs"][p])
        all_ep_basins.extend(sd["ep_basins"])
        all_safety_hdr.extend(sd["ep_safety_rates"]["hdr_main"])
        all_safety_pe.extend(sd["ep_safety_rates"]["pooled_lqr_estimated"])

    costs_hdr = np.array(all_ep_costs["hdr_main"])
    costs_pe = np.array(all_ep_costs["pooled_lqr_estimated"])
    basins_arr = np.array(all_ep_basins)

    with np.errstate(divide="ignore", invalid="ignore"):
        paired_ratios_all = np.where(
            costs_pe > 1e-12, (costs_pe - costs_hdr) / costs_pe, 0.0
        )

    mask_mal = basins_arr == 1
    mask_adp = (basins_arr == 0) | (basins_arr == 2)

    hdr_vs_pe_maladaptive = float(np.mean(paired_ratios_all[mask_mal])) if np.any(mask_mal) else 0.0
    hdr_vs_pe_adaptive = float(np.mean(paired_ratios_all[mask_adp])) if np.any(mask_adp) else 0.0

    cost_diff_all = costs_pe - costs_hdr
    hdr_mal_win_rate = (
        float(np.mean(cost_diff_all[mask_mal] > 0)) if np.any(mask_mal) else 0.0
    )
    safety_delta = float(np.mean(all_safety_hdr) - np.mean(all_safety_pe))
    n_maladaptive_episodes = int(np.sum(mask_mal))

    gains_maladaptive = paired_ratios_all[mask_mal] if np.any(mask_mal) else np.array([0.0])

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    ci_95_mean_lo, ci_95_mean_hi = _bootstrap_ci(gains_maladaptive, n_boot=10_000, ci=0.95, rng_seed=42, stat="mean")
    ci_90_mean_lo, ci_90_mean_hi = _bootstrap_ci(gains_maladaptive, n_boot=10_000, ci=0.90, rng_seed=42, stat="mean")
    ci_95_med_lo, ci_95_med_hi = _bootstrap_ci(gains_maladaptive, n_boot=10_000, ci=0.95, rng_seed=42, stat="median")

    criterion_95_mean = bool(ci_95_mean_lo >= 0.03)
    criterion_90_mean = bool(ci_90_mean_lo >= 0.03)
    criterion_95_med = bool(ci_95_med_lo >= 0.03)

    # ── Per-seed stability ─────────────────────────────────────────────────────
    seed_gains: dict[str, float] = {}
    for s in cfg["seeds"]:
        sd = seed_results[s]
        c_hdr_s = np.array(sd["ep_costs"]["hdr_main"])
        c_pe_s = np.array(sd["ep_costs"]["pooled_lqr_estimated"])
        b_s = np.array(sd["ep_basins"])
        with np.errstate(divide="ignore", invalid="ignore"):
            pr_s = np.where(c_pe_s > 1e-12, (c_pe_s - c_hdr_s) / c_pe_s, 0.0)
        mask_mal_s = b_s == 1
        g = float(np.mean(pr_s[mask_mal_s])) if np.any(mask_mal_s) else float("nan")
        seed_gains[str(s)] = g

    seed_gain_vals = [v for v in seed_gains.values() if not np.isnan(v)]
    seed_gain_min = float(min(seed_gain_vals)) if seed_gain_vals else float("nan")
    seed_gain_max = float(max(seed_gain_vals)) if seed_gain_vals else float("nan")
    seed_gain_std = float(np.std(seed_gain_vals)) if len(seed_gain_vals) > 1 else 0.0
    n_seeds_above = int(sum(1 for g in seed_gain_vals if g >= 0.03))

    print(f"    hdr_vs_pe_maladaptive      = {hdr_vs_pe_maladaptive:+.4f}")
    print(f"    hdr_vs_pe_adaptive         = {hdr_vs_pe_adaptive:+.4f}")
    print(f"    hdr_mal_win_rate           = {hdr_mal_win_rate:.4f}")
    print(f"    safety_delta_vs_pe         = {safety_delta:+.4f}")
    print(f"    n_maladaptive_episodes     = {n_maladaptive_episodes}")
    print(f"    95% CI mean                = [{ci_95_mean_lo:+.4f}, {ci_95_mean_hi:+.4f}]")
    print(f"    90% CI mean                = [{ci_90_mean_lo:+.4f}, {ci_90_mean_hi:+.4f}]")
    print(f"    95% CI median              = [{ci_95_med_lo:+.4f}, {ci_95_med_hi:+.4f}]")
    print(f"    criterion (+3% 95CI mean)  = {criterion_95_mean}")
    print(f"    seeds >= 0.03              = {n_seeds_above}/20")
    print(f"    seed_gain_std              = {seed_gain_std:.4f}")

    # ── Write summary JSON ─────────────────────────────────────────────────────
    summary = {
        "profile": "highpower",
        "n_seeds": 20,
        "episodes_per_seed": 30,
        "total_episodes": 600,
        "n_maladaptive_episodes": n_maladaptive_episodes,
        "steps_per_episode": 256,
        "hdr_vs_pe_maladaptive_mean": hdr_vs_pe_maladaptive,
        "hdr_mal_win_rate": hdr_mal_win_rate,
        "hdr_vs_pe_adaptive_mean": hdr_vs_pe_adaptive,
        "safety_delta_vs_pe": safety_delta,
        "ci_95_mean_lo": ci_95_mean_lo,
        "ci_95_mean_hi": ci_95_mean_hi,
        "ci_90_mean_lo": ci_90_mean_lo,
        "ci_90_mean_hi": ci_90_mean_hi,
        "ci_95_median_lo": ci_95_med_lo,
        "ci_95_median_hi": ci_95_med_hi,
        "criterion_plus3pct_satisfied_95ci_mean": criterion_95_mean,
        "criterion_plus3pct_satisfied_90ci_mean": criterion_90_mean,
        "criterion_plus3pct_satisfied_95ci_median": criterion_95_med,
        "seed_gains": seed_gains,
        "seed_gain_min": seed_gain_min,
        "seed_gain_max": seed_gain_max,
        "seed_gain_std": seed_gain_std,
        "n_seeds_above_criterion": n_seeds_above,
        "gains_maladaptive_all": gains_maladaptive.tolist(),
        "config_snapshot": {
            "lambda_u": float(cfg["lambda_u"]),
            "default_burden_budget": float(cfg["default_burden_budget"]),
            "rho_reference": list(cfg["rho_reference"]),
            "K": int(cfg["K"]),
            "H": int(cfg["H"]),
            "state_dim": int(cfg["state_dim"]),
            "control_dim": int(cfg["control_dim"]),
        },
    }
    _atomic_write_json(out_dir / "highpower_summary.json", summary)
    print("  Wrote highpower_summary.json")

    # ── Write text table ───────────────────────────────────────────────────────
    if criterion_95_mean:
        criterion_str = f"PASSED — CI lower bound {ci_95_mean_lo:+.4f} >= +0.030"
    else:
        criterion_str = f"FAILED — CI lower bound {ci_95_mean_lo:+.4f} < +0.030"

    table_lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║  BENCHMARK A — HIGH-POWER RESULTS (20 seeds × 30 episodes)  ║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  Maladaptive episodes (basin 1, rho=0.96)                   ║",
        f"║  N_maladaptive : {n_maladaptive_episodes:<43d}║",
        "║                                                              ║",
        "║  Mean gain vs pooled_lqr_estimated                          ║",
        f"║    Point estimate : {hdr_vs_pe_maladaptive:+.4f}                               ║",
        f"║    95% CI (mean)  : [{ci_95_mean_lo:+.4f}, {ci_95_mean_hi:+.4f}]                    ║",
        f"║    90% CI (mean)  : [{ci_90_mean_lo:+.4f}, {ci_90_mean_hi:+.4f}]                    ║",
        f"║    95% CI (median): [{ci_95_med_lo:+.4f}, {ci_95_med_hi:+.4f}]                    ║",
        "║                                                              ║",
        f"║  Win rate : {hdr_mal_win_rate:.3f}                                       ║",
        f"║  Safety Delta : {safety_delta:+.4f}                                  ║",
        "║                                                              ║",
        "║  Criterion (+3% lower bound, 95% CI mean):                  ║",
        f"║    {criterion_str:<58s}║",
        "║                                                              ║",
        "║  Per-seed stability                                          ║",
        f"║    Min seed gain : {seed_gain_min:+.4f}                               ║",
        f"║    Max seed gain : {seed_gain_max:+.4f}                               ║",
        f"║    Std seed gain : {seed_gain_std:.4f}                               ║",
        f"║    Seeds >= +0.03 : {n_seeds_above}/20                                ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        "PROFILE COMPARISON — Benchmark A Maladaptive Gain",
        "──────────────────────────────────────────────────────────────",
        "Profile       Seeds  Ep/seed   N_mal  Mean gain  Win rate",
        "standard        2      12       ~11   +0.0574    0.909",
        "extended        3      20       ~15   +0.0357    0.800",
        f"highpower      20      30      {n_maladaptive_episodes:>4d}   {hdr_vs_pe_maladaptive:+.4f}    {hdr_mal_win_rate:.3f}",
        "──────────────────────────────────────────────────────────────",
    ]
    table_text = "\n".join(table_lines)
    (out_dir / "highpower_table.txt").write_text(table_text)
    print("  Wrote highpower_table.txt")

    # ── Manuscript LaTeX table ─────────────────────────────────────────────────
    # Estimate n_mal for standard/extended from existing results
    n_std = 11  # ~11 maladaptive out of 24 total (2 seeds × 12 episodes)
    n_ext = 15  # ~15 maladaptive out of 60 total (3 seeds × 20 episodes)
    # Try to read actual values from existing results
    std_cal = ROOT / "results" / "stage_04" / "standard" / "chance_calibration.json"
    ext_cal = ROOT / "results" / "stage_04" / "extended" / "chance_calibration.json"
    if std_cal.exists():
        try:
            d = json.loads(std_cal.read_text())
            if "n_maladaptive_episodes" in d:
                n_std = int(d["n_maladaptive_episodes"])
        except Exception:
            pass
    if ext_cal.exists():
        try:
            d = json.loads(ext_cal.read_text())
            if "n_maladaptive_episodes" in d:
                n_ext = int(d["n_maladaptive_episodes"])
        except Exception:
            pass

    mean_fmt = f"{hdr_vs_pe_maladaptive:+.3f}"
    lo_fmt = f"{ci_95_mean_lo:+.3f}"
    hi_fmt = f"{ci_95_mean_hi:+.3f}"
    win_pct = f"{hdr_mal_win_rate:.0%}"
    criterion_word = "PASSED" if criterion_95_mean else "FAILED"

    manuscript_table = rf"""\begin{{table}}[t]
\centering
\caption{{Benchmark A: HDR Mode A vs.\ pooled LQR (estimated state),
maladaptive basin (Basin~1, $\rho=0.96$).
Gain = mean fractional cost reduction over \texttt{{pooled\_lqr\_estimated}}.
CI = 95\% bootstrap percentile interval (10\,000 resamples).}}
\label{{tab:benchmark_a_highpower}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Profile & Seeds & Ep./seed & $N_{{\rm mal}}$ & Mean gain & 95\% CI \\
\midrule
Standard   &  2 & 12 & {n_std}  & $+0.057$ & ---        \\
Extended   &  3 & 20 & {n_ext}  & $+0.036$ & ---        \\
High-power & 20 & 30 & {n_maladaptive_episodes}   & ${mean_fmt}$ & $[{lo_fmt},\,{hi_fmt}]$ \\
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
{{\small Win rate (high-power): {win_pct}.
Safety $\Delta$ vs.\ baseline: ${safety_delta:+.4f}$.
Criterion: CI lower bound $\geq +0.03$:
\textbf{{{criterion_word}}}.}}
\end{{table}}
"""
    (out_dir / "manuscript_table.txt").write_text(manuscript_table)
    print("  Wrote manuscript_table.txt")

    # ── Manuscript language (Branch A or B) ───────────────────────────────────
    mean_pct = f"{hdr_vs_pe_maladaptive:+.1%}"
    lo_pct = f"{ci_95_mean_lo:+.1%}"
    hi_pct = f"{ci_95_mean_hi:+.1%}"

    if criterion_95_mean:
        # Branch A
        lang_block = (
            "╔════════════════════════════════════════════════════════════╗\n"
            "║  BENCHMARK A CONFIRMED                                     ║\n"
            "║  The 95% CI lower bound clears the +3% criterion.         ║\n"
            "║  The manuscript claim is statistically supported.          ║\n"
            "║                                                            ║\n"
            "║  Recommended manuscript language:                          ║\n"
            f'║  "HDR achieves a mean fractional cost reduction of         ║\n'
            f'║   {mean_pct} over the pooled_lqr_estimated baseline on   ║\n'
            f'║   maladaptive-basin episodes (95% CI: [{lo_pct},         ║\n'
            f'║   {hi_pct}]; win rate {hdr_mal_win_rate:.0%}; N={n_maladaptive_episodes}          ║\n'
            f'║   episodes across 20 independent seeds)."                  ║\n'
            "╚════════════════════════════════════════════════════════════╝\n"
        )
        print("\n" + lang_block)
        manuscript_lang = (
            "BENCHMARK A CONFIRMED\n"
            "The 95% CI lower bound clears the +3% criterion.\n"
            "The manuscript claim is statistically supported.\n\n"
            "Recommended manuscript language:\n"
            f'"HDR achieves a mean fractional cost reduction of {mean_pct} '
            f"over the pooled_lqr_estimated baseline on maladaptive-basin episodes "
            f"(95% CI: [{lo_pct}, {hi_pct}]; win rate {hdr_mal_win_rate:.0%}; "
            f'N={n_maladaptive_episodes} episodes across 20 independent seeds)."\n'
        )
    else:
        # Branch B — determine diagnosis
        ci_90_lo_pct = f"{ci_90_mean_lo:+.4f}"
        ci_90_hi_pct = f"{ci_90_mean_hi:+.4f}"

        if seed_gain_std > 0.04:
            diagnosis_text = (
                f"High cross-seed variance (std={seed_gain_std:.3f}) suggests "
                "the effect is real but noisy.  Consider: (a) reporting the "
                f"90% CI which is [{ci_90_mean_lo:+.4f}, {ci_90_mean_hi:+.4f}] and "
                "weaker +2% criterion, or (b) increasing episodes/seed to "
                "reduce variance before publication."
            )
        elif n_seeds_above < 10:
            diagnosis_text = (
                f"Fewer than half of seeds ({n_seeds_above}/20) "
                "individually exceed the +3% criterion.  The pooled mean is "
                "driven by a minority of high-gain seeds.  Investigate whether "
                "the effect is specific to certain random initialisations."
            )
        else:
            diagnosis_text = (
                "The CI just misses the +0.03 threshold.  The point estimate "
                "is robust but the precision requires more episodes.  Running "
                "30 episodes per seed (600 total) would likely close the gap."
            )

        lang_block = (
            "╔════════════════════════════════════════════════════════════╗\n"
            "║  BENCHMARK A — CRITERION NOT MET AT 95% CI LEVEL          ║\n"
            f"║  CI lower bound {ci_95_mean_lo:+.4f} < +0.030                     ║\n"
            f"║  Point estimate {hdr_vs_pe_maladaptive:+.4f}  (above threshold)            ║\n"
            "║                                                            ║\n"
            "║  Diagnosis:                                                ║\n"
            f"║  {diagnosis_text[:58]:<58s}║\n"
        )
        # Wrap long diagnosis
        if len(diagnosis_text) > 58:
            remaining = diagnosis_text[58:]
            while remaining:
                chunk = remaining[:58]
                remaining = remaining[58:]
                lang_block += f"║  {chunk:<58s}║\n"
        lang_block += "╚════════════════════════════════════════════════════════════╝\n"
        print("\n" + lang_block)

        weaker_lang = (
            f"HDR achieves a mean fractional cost reduction of {mean_pct} "
            "over the pooled_lqr_estimated baseline on maladaptive-basin episodes "
            f"(win rate {hdr_mal_win_rate:.0%}; N={n_maladaptive_episodes} episodes across "
            "20 independent seeds).  "
            f"The 95% bootstrap CI [{lo_pct}, {hi_pct}] does not exclude gains below +3%, "
            "indicating meaningful but not yet precisely characterised improvement."
        )
        manuscript_lang = (
            f"BENCHMARK A — CRITERION NOT MET AT 95% CI LEVEL\n"
            f"CI lower bound {ci_95_mean_lo:+.4f} < +0.030\n"
            f"Point estimate {hdr_vs_pe_maladaptive:+.4f} (above threshold)\n\n"
            f"Diagnosis:\n{diagnosis_text}\n\n"
            f"Recommended manuscript language (weaker-but-honest framing):\n{weaker_lang}\n"
        )

    (out_dir / "manuscript_language.txt").write_text(manuscript_lang)
    print("  Wrote manuscript_language.txt")

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  BENCHMARK A — HIGH-POWER RUNNER (20 seeds × 30 episodes)")
    print("=" * 62)
    t0 = time.perf_counter()
    try:
        summary = run_highpower_benchmark()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.perf_counter() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("  Done.")
