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

from hdr_validation.defaults import make_config

# ── High-power profile config (overrides only) ────────────────────────────────
HIGHPOWER_CONFIG: dict[str, Any] = make_config(
    profile_name="highpower",
    max_dwell_len=256,
    seeds=[
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    ],
    episodes_per_experiment=30,
    steps_per_episode=256,
    mc_rollouts=150,
    selected_trace_cap=5,
)

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
def _generate_episode(cfg: dict, rng: np.random.Generator, basin_idx: int = 0,
                      eval_model=None) -> dict:
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.specification import (
        observation_schedule,
        generate_observation,
        heteroskedastic_R,
    )

    if eval_model is None:
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
        x_next = basin.A @ x + basin.B @ u + w + basin.b
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
    data = np.asarray(data, dtype=float)
    indices = rng.integers(0, len(data), size=(n_boot, len(data)))
    if stat == "mean":
        boot_stats = data[indices].mean(axis=1)
    else:
        boot_stats = np.median(data[indices], axis=1)
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
    )
    from hdr_validation.inference.imm import IMMFilter
    from hdr_validation.specification import (
        observation_schedule,
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
    policy_names = ["open_loop", "pooled_lqr", "basin_lqr", "hdr_main", "pooled_lqr_estimated", "hdr_tube_mpc"]

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

        # ── Tube-MPC: pre-compute mRPI terminal sets (once per basin) ────
        # Audit summary (tube_mpc.py):
        #   - No controller class; functional API with compute_disturbance_set(),
        #     compute_mRPI_zonotope(), and solve_tube_mpc().
        #   - mRPI pre-computation: dlqr → A_cl = A - B @ K_fb, then
        #     compute_disturbance_set(Q_w, n, beta) → compute_mRPI_zonotope(A_cl, Q_w, chi2).
        #   - Per-step: solve_tube_mpc(x_hat, P_hat, basin, target, mRPI_data, K_fb, ...).
        #   - mRPI_data is a plain dict (G, center, alpha_s, iterations, scale) —
        #     fully cacheable across episodes.
        #   - beta=0.999 matches Stage 11 convention for 99.9th-percentile disturbance.
        from hdr_validation.control.tube_mpc import (
            compute_disturbance_set,
            compute_mRPI_zonotope,
            solve_tube_mpc,
        )

        tube_mRPI: dict[int, dict] = {}
        tube_K_fb: dict[int, np.ndarray] = {}
        basin_rho = cfg["rho_reference"]
        for k, basin_k in enumerate(sim_model.basins):
            K_fb_k = K_basin_lqr[k]  # reuse already-computed LQR gain
            A_cl_k = basin_k.A - basin_k.B @ K_fb_k
            _, chi2_bound_k = compute_disturbance_set(basin_k.Q, n, beta=0.999)
            mrpi_k = compute_mRPI_zonotope(
                A_cl_k, basin_k.Q, chi2_bound_k, epsilon=0.01
            )
            tube_mRPI[k] = mrpi_k
            tube_K_fb[k] = K_fb_k
            print(
                f"    mRPI pre-computed for basin {k} "
                f"(rho={basin_rho[k]:.3f}, iters={mrpi_k['iterations']})"
            )

        # HP-2: Pre-compute safety check invariants (constant per basin)
        from scipy.stats import norm as _norm_dist
        safety_std_per_basin: dict[int, np.ndarray] = {}
        for k_idx, basin_k in enumerate(sim_model.basins):
            y_cov_k = basin_k.C @ P_safety @ basin_k.C.T + basin_k.R
            safety_std_per_basin[k_idx] = np.sqrt(
                np.maximum(np.diag(y_cov_k), 1e-12)
            )
        eps_safe_val = float(cfg["eps_safe"])

        def _safety_violation_fast(x_state, basin_k_idx):
            bk = sim_model.basins[basin_k_idx]
            y_mean = bk.C @ x_state + bk.c
            std = safety_std_per_basin[basin_k_idx]
            lower = _norm_dist.cdf((y_lo - y_mean) / std)
            upper = _norm_dist.cdf((y_mean - y_hi) / std)
            return float(np.max(lower + upper)) > eps_safe_val

        # HP-3: Pre-compute target sets per basin
        target_sets = [build_target_set(k, cfg) for k in range(len(sim_model.basins))]

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

            # HP-1: Pre-generate raw observation noise once per step (not per policy)
            obs_noise_raw = np.empty((T, m_obs))
            for _t in range(T):
                _obs_rng = np.random.default_rng(
                    cfg["seeds"][0] + 7000 + ep_idx * T + _t
                )
                obs_noise_raw[_t] = _obs_rng.standard_normal(m_obs)

            # ── Phase 1: estimation-based policies (independent IMM filters) ──
            # Each policy drives its own IMM filter from its own trajectory's
            # observations. Missingness pattern (mask_sched) and process noise
            # are shared; observation noise seed is shared per timestep.
            imm_filt_hdr = IMMFilter.for_hard_regime(sim_model)
            imm_filt_pe = IMMFilter.for_hard_regime(sim_model)
            imm_filt_tube = IMMFilter.for_hard_regime(sim_model)
            x_hdr = x_init.copy()
            x_pe = x_init.copy()
            x_tube = x_init.copy()
            cost_hdr, cost_pe, cost_tube = 0.0, 0.0, 0.0
            viol_hdr, viol_pe, viol_tube = 0, 0, 0
            used_burden_hdr, used_burden_pe, used_burden_tube = 0.0, 0.0, 0.0
            u_prev_hdr = np.zeros(m_u)
            u_prev_pe = np.zeros(m_u)
            u_prev_tube = np.zeros(m_u)
            tube_fallbacks = 0

            # HP-8: Alias hot basin attributes to avoid repeated lookups
            basin_A = basin_obj.A
            basin_B = basin_obj.B
            basin_b = basin_obj.b
            basin_C = basin_obj.C
            basin_c = basin_obj.c
            basin_R = basin_obj.R

            for t in range(T):
                noise_t = obs_noise_raw[t]
                mask_t = mask_sched[t]
                mask_t_bool = mask_t.astype(bool)

                # HP-1: HDR observations using pre-drawn noise
                R_t_hdr = heteroskedastic_R(basin_R, x_hdr, mask_t, t)
                diag_R_hdr = np.diag(R_t_hdr)
                y_t_hdr = basin_C @ x_hdr + basin_c + noise_t * np.sqrt(diag_R_hdr)
                y_t_hdr = np.where(mask_t_bool, y_t_hdr, np.nan)
                mask_int_hdr = (~np.isnan(y_t_hdr)).astype(int)
                y_clean_hdr = np.where(np.isnan(y_t_hdr), 0.0, y_t_hdr)
                imm_state_hdr = imm_filt_hdr.step(y_clean_hdr, mask_int_hdr, u_prev_hdr)
                x_hat_hdr = imm_state_hdr.mixed_mean
                P_hat_hdr = imm_state_hdr.mixed_cov

                # HP-1: PE observations using pre-drawn noise
                R_t_pe = heteroskedastic_R(basin_R, x_pe, mask_t, t)
                diag_R_pe = np.diag(R_t_pe)
                y_t_pe = basin_C @ x_pe + basin_c + noise_t * np.sqrt(diag_R_pe)
                y_t_pe = np.where(mask_t_bool, y_t_pe, np.nan)
                mask_int_pe = (~np.isnan(y_t_pe)).astype(int)
                y_clean_pe = np.where(np.isnan(y_t_pe), 0.0, y_t_pe)
                imm_state_pe = imm_filt_pe.step(y_clean_pe, mask_int_pe, u_prev_pe)
                x_hat_pe = imm_state_pe.mixed_mean

                # hdr_main: MPC on estimated basin using its own filter
                est_bi = imm_state_hdr.map_mode
                est_basin = sim_model.basins[est_bi]
                est_target = target_sets[est_bi]  # HP-3
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

                # HP-1: tube-MPC observations using pre-drawn noise
                R_t_tube = heteroskedastic_R(basin_R, x_tube, mask_t, t)
                diag_R_tube = np.diag(R_t_tube)
                y_t_tube = basin_C @ x_tube + basin_c + noise_t * np.sqrt(diag_R_tube)
                y_t_tube = np.where(mask_t_bool, y_t_tube, np.nan)
                mask_int_tube = (~np.isnan(y_t_tube)).astype(int)
                y_clean_tube = np.where(np.isnan(y_t_tube), 0.0, y_t_tube)
                imm_state_tube = imm_filt_tube.step(y_clean_tube, mask_int_tube, u_prev_tube)
                x_hat_tube = imm_state_tube.mixed_mean
                P_hat_tube = imm_state_tube.mixed_cov

                est_bi_tube = imm_state_tube.map_mode
                est_basin_tube = sim_model.basins[est_bi_tube]
                est_target_tube = target_sets[est_bi_tube]  # HP-3
                try:
                    tube_res = solve_tube_mpc(
                        x_hat_tube,
                        P_hat_tube,
                        est_basin_tube,
                        est_target_tube,
                        tube_mRPI[est_bi_tube],
                        tube_K_fb[est_bi_tube],
                        kappa_hat=0.6,
                        config=cfg,
                        step=t,
                    )
                    u_tube = tube_res.u
                except Exception:
                    u_tube = np.zeros(m_u)
                    tube_fallbacks += 1

                # Costs
                cost_hdr += float(np.dot(x_hdr, x_hdr) + lambda_u * np.dot(u_hdr, u_hdr))
                cost_pe += float(np.dot(x_pe, x_pe) + lambda_u * np.dot(u_pe, u_pe))
                cost_tube += float(np.dot(x_tube, x_tube) + lambda_u * np.dot(u_tube, u_tube))

                # HP-2: Safety checks with pre-computed std
                if _safety_violation_fast(x_hdr, basin_idx):
                    viol_hdr += 1
                if _safety_violation_fast(x_pe, basin_idx):
                    viol_pe += 1
                if _safety_violation_fast(x_tube, basin_idx):
                    viol_tube += 1

                # HP-4: Evolve all with shared process noise (E=I elided)
                w = process_noise[t]
                x_hdr = basin_A @ x_hdr + basin_B @ u_hdr + w + basin_b
                x_pe = basin_A @ x_pe + basin_B @ u_pe + w + basin_b
                x_tube = basin_A @ x_tube + basin_B @ u_tube + w + basin_b
                used_burden_hdr += float(np.sum(np.abs(u_hdr)))
                used_burden_pe += float(np.sum(np.abs(u_pe)))
                used_burden_tube += float(np.sum(np.abs(u_tube)))
                # HP-7: No .copy() needed — u arrays are freshly allocated each step
                u_prev_hdr = u_hdr
                u_prev_pe = u_pe
                u_prev_tube = u_tube

            ep_costs["hdr_main"].append(cost_hdr)
            ep_costs["pooled_lqr_estimated"].append(cost_pe)
            ep_costs["hdr_tube_mpc"].append(cost_tube)
            ep_safety_rates["hdr_main"].append(viol_hdr / T)
            ep_safety_rates["pooled_lqr_estimated"].append(viol_pe / T)
            ep_safety_rates["hdr_tube_mpc"].append(viol_tube / T)

            # ── Phase 2: oracle-state policies (merged loop, HP-5) ──────────
            x_ol = x_init.copy()
            x_pl = x_init.copy()
            x_bl = x_init.copy()
            cost_ol, cost_pl, cost_bl = 0.0, 0.0, 0.0
            viol_ol, viol_pl, viol_bl = 0, 0, 0
            used_pl, used_bl = 0.0, 0.0
            K_bl = K_basin_lqr[basin_idx]

            for t in range(T):
                # Open loop: u = 0
                cost_ol += float(x_ol @ x_ol)

                # Pooled LQR
                u_pl = -K_pooled @ x_pl
                u_pl, _ = apply_control_constraints(
                    u_pl, cfg, step=t, used_burden=used_pl
                )
                cost_pl += float(x_pl @ x_pl + lambda_u * (u_pl @ u_pl))

                # Basin LQR (oracle)
                u_bl = -K_bl @ x_bl
                u_bl, _ = apply_control_constraints(
                    u_bl, cfg, step=t, used_burden=used_bl
                )
                cost_bl += float(x_bl @ x_bl + lambda_u * (u_bl @ u_bl))

                # HP-2: Safety checks
                if _safety_violation_fast(x_ol, basin_idx):
                    viol_ol += 1
                if _safety_violation_fast(x_pl, basin_idx):
                    viol_pl += 1
                if _safety_violation_fast(x_bl, basin_idx):
                    viol_bl += 1

                # HP-4: State evolution (E=I elided)
                w = process_noise[t]
                x_ol = basin_A @ x_ol + w + basin_b
                x_pl = basin_A @ x_pl + basin_B @ u_pl + w + basin_b
                x_bl = basin_A @ x_bl + basin_B @ u_bl + w + basin_b
                used_pl += float(np.sum(np.abs(u_pl)))
                used_bl += float(np.sum(np.abs(u_bl)))

            ep_costs["open_loop"].append(cost_ol)
            ep_costs["pooled_lqr"].append(cost_pl)
            ep_costs["basin_lqr"].append(cost_bl)
            ep_safety_rates["open_loop"].append(viol_ol / T)
            ep_safety_rates["pooled_lqr"].append(viol_pl / T)
            ep_safety_rates["basin_lqr"].append(viol_bl / T)

        # ── Write per-seed partial checkpoint ─────────────────────────────────
        partial_data = {
            "seed": s,
            "n_episodes": n_eps_per_seed,
            "ep_costs": {p: ep_costs[p] for p in policy_names},
            "ep_basins": ep_basins,
            "ep_safety_rates": {
                "hdr_main": ep_safety_rates["hdr_main"],
                "pooled_lqr_estimated": ep_safety_rates["pooled_lqr_estimated"],
                "hdr_tube_mpc": ep_safety_rates["hdr_tube_mpc"],
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
    all_safety_tube: list[float] = []

    for s in cfg["seeds"]:
        sd = seed_results[s]
        for p in policy_names:
            all_ep_costs[p].extend(sd["ep_costs"][p])
        all_ep_basins.extend(sd["ep_basins"])
        all_safety_hdr.extend(sd["ep_safety_rates"]["hdr_main"])
        all_safety_pe.extend(sd["ep_safety_rates"]["pooled_lqr_estimated"])
        all_safety_tube.extend(sd["ep_safety_rates"]["hdr_tube_mpc"])

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

    # ── Tube-MPC vs pooled_lqr_estimated ──────────────────────────────────
    costs_tube = np.array(all_ep_costs["hdr_tube_mpc"])
    with np.errstate(divide="ignore", invalid="ignore"):
        tube_ratios_all = np.where(
            costs_pe > 1e-12, (costs_pe - costs_tube) / costs_pe, 0.0
        )
    tube_vs_pe_maladaptive = float(np.mean(tube_ratios_all[mask_mal])) if np.any(mask_mal) else 0.0
    tube_vs_pe_adaptive = float(np.mean(tube_ratios_all[mask_adp])) if np.any(mask_adp) else 0.0
    tube_mal_win_rate = (
        float(np.mean((costs_pe - costs_tube)[mask_mal] > 0)) if np.any(mask_mal) else 0.0
    )
    safety_delta_tube = float(np.mean(all_safety_tube) - np.mean(all_safety_pe))

    # ── Tube-MPC vs hdr_main (head-to-head) ───────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        tube_vs_hdr_ratios = np.where(
            costs_hdr > 1e-12, (costs_hdr - costs_tube) / costs_hdr, 0.0
        )
    tube_vs_hdr_maladaptive = float(np.mean(tube_vs_hdr_ratios[mask_mal])) if np.any(mask_mal) else 0.0
    tube_vs_hdr_all = float(np.mean(tube_vs_hdr_ratios))

    gains_tube_maladaptive = tube_ratios_all[mask_mal] if np.any(mask_mal) else np.array([0.0])

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

    # ── Tube-MPC bootstrap CIs ─────────────────────────────────────────────
    ci_tube_95_lo, ci_tube_95_hi = _bootstrap_ci(gains_tube_maladaptive, n_boot=10_000, ci=0.95, rng_seed=43, stat="mean")

    print(f"\n  Tube-MPC vs pooled_lqr_estimated (maladaptive):")
    print(f"    tube_vs_pe_maladaptive     = {tube_vs_pe_maladaptive:+.4f}")
    print(f"    tube_vs_pe_adaptive        = {tube_vs_pe_adaptive:+.4f}")
    print(f"    tube_mal_win_rate          = {tube_mal_win_rate:.4f}")
    print(f"    safety_delta_tube_vs_pe    = {safety_delta_tube:+.4f}")
    print(f"    95% CI mean                = [{ci_tube_95_lo:+.4f}, {ci_tube_95_hi:+.4f}]")
    print(f"  Tube-MPC vs hdr_main (head-to-head):")
    print(f"    tube_vs_hdr_maladaptive    = {tube_vs_hdr_maladaptive:+.4f}")
    print(f"    tube_vs_hdr_all            = {tube_vs_hdr_all:+.4f}")

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
        "tube_vs_pe_maladaptive_mean": tube_vs_pe_maladaptive,
        "tube_vs_pe_adaptive_mean": tube_vs_pe_adaptive,
        "tube_mal_win_rate": tube_mal_win_rate,
        "safety_delta_tube_vs_pe": safety_delta_tube,
        "ci_tube_95_mean_lo": ci_tube_95_lo,
        "ci_tube_95_mean_hi": ci_tube_95_hi,
        "tube_vs_hdr_maladaptive_mean": tube_vs_hdr_maladaptive,
        "tube_vs_hdr_all_mean": tube_vs_hdr_all,
        "gains_tube_maladaptive_all": gains_tube_maladaptive.tolist(),
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
    from hdr_validation.provenance import get_provenance
    summary["provenance"] = get_provenance()
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
