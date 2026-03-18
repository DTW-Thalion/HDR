"""
WP-2.3: Cluster-Aware Bootstrap CI Analysis for Stage 04
=========================================================
Runs Stage 04 (Mode A MPC benchmark) with 100 seeds × 30 episodes/seed
= 3000 total episodes. Computes:

  a. Mean gain (HDR vs baseline) across all episodes
  b. Episode-level bootstrap 95% CI (10,000 resamples, percentile method)
  c. Seed-cluster bootstrap 95% CI (resample seeds as clusters)
  d. Intraclass correlation coefficient (ICC, one-way random effects)
  e. Design effect (DEFF = 1 + (n_per_cluster - 1) × ICC)
  f. Effective N (total episodes / DEFF)

Also runs multi-seed Stage 10 and Stage 15 for uncertainty bars.

Usage:
    python cluster_bootstrap_runner.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hdr_validation.defaults import make_config


# ── 100-seed configuration ────────────────────────────────────────────────────
CLUSTER_CONFIG: dict[str, Any] = make_config(
    profile_name="highpower",
    max_dwell_len=256,
    seeds=list(range(101, 101 + 100 * 101, 101)),  # 101, 202, ..., 10100
    episodes_per_experiment=30,
    steps_per_episode=256,
    mc_rollouts=150,
    selected_trace_cap=5,
)


# ── Atomic write helper ──────────────────────────────────────────────────────
def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


# ── Bootstrap helpers ─────────────────────────────────────────────────────────
def _episode_bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Standard (episode-level) bootstrap CI — resample individual episodes."""
    rng = np.random.default_rng(rng_seed)
    n = len(data)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = data[indices].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return lo, hi


def _cluster_bootstrap_ci(
    seed_labels: np.ndarray,
    gains: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 43,
) -> tuple[float, float]:
    """Seed-cluster bootstrap CI — resample seeds, keep all episodes per seed."""
    rng = np.random.default_rng(rng_seed)
    unique_seeds = np.unique(seed_labels)
    n_seeds = len(unique_seeds)

    # Build index of episodes per seed
    seed_to_idx: dict[int, np.ndarray] = {}
    for s in unique_seeds:
        seed_to_idx[int(s)] = np.where(seed_labels == s)[0]

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        resampled_seeds = rng.choice(unique_seeds, size=n_seeds, replace=True)
        all_gains = np.concatenate([gains[seed_to_idx[int(s)]] for s in resampled_seeds])
        boot_means[b] = all_gains.mean()

    lo = float(np.percentile(boot_means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 * (1 + ci) / 2))
    return lo, hi


def _compute_icc(seed_labels: np.ndarray, values: np.ndarray) -> float:
    """One-way random effects ICC (ICC(1,1)).

    ICC = (MS_between - MS_within) / (MS_between + (k-1) * MS_within)
    where k = number of observations per cluster.
    """
    unique_seeds = np.unique(seed_labels)
    n_seeds = len(unique_seeds)

    # Group by seed
    groups = [values[seed_labels == s] for s in unique_seeds]
    group_sizes = np.array([len(g) for g in groups])
    k_mean = float(np.mean(group_sizes))  # mean cluster size

    grand_mean = float(np.mean(values))
    group_means = np.array([float(np.mean(g)) for g in groups])

    # Between-group sum of squares
    ss_between = sum(len(g) * (gm - grand_mean) ** 2
                     for g, gm in zip(groups, group_means))
    # Within-group sum of squares
    ss_within = sum(float(np.sum((g - gm) ** 2))
                    for g, gm in zip(groups, group_means))

    df_between = n_seeds - 1
    df_within = int(np.sum(group_sizes)) - n_seeds

    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)

    # ICC(1,1) with unequal group sizes uses k0
    # k0 = (1/(a-1)) * (N - sum(n_i^2)/N) where a = n_seeds, N = total
    N = int(np.sum(group_sizes))
    k0 = (N - float(np.sum(group_sizes ** 2)) / N) / max(n_seeds - 1, 1)

    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
    return float(np.clip(icc, 0.0, 1.0))


# ── Main Stage 04 runner with 100 seeds ──────────────────────────────────────
def run_stage_04_cluster() -> dict:
    """Run Stage 04 with 100 seeds, compute cluster-aware statistics."""
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
        generate_observation,
        heteroskedastic_R,
    )
    from hdr_validation.control.tube_mpc import (
        compute_disturbance_set,
        compute_mRPI_zonotope,
        solve_tube_mpc,
    )
    from scipy.stats import norm as _norm_dist

    cfg = CLUSTER_CONFIG
    out_dir = ROOT / "results" / "stage_04" / "cluster_bootstrap"
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

    print(f"\n  Stage 04 Cluster Bootstrap: {n_seeds} seeds × {n_eps_per_seed} episodes = {n_seeds * n_eps_per_seed} total")

    # ── Collect per-seed results (with checkpoint resume) ────────────────────
    seed_results: dict[int, dict] = {}

    # Load existing checkpoints
    for s in cfg["seeds"]:
        partial_path = out_dir / f"seed_{s:05d}_partial.json"
        if partial_path.exists():
            try:
                with open(partial_path) as f:
                    seed_results[s] = json.load(f)
                print(f"  Seed {s}: found checkpoint, skipping.")
            except Exception:
                pass

    # Run missing seeds
    for i, s in enumerate(cfg["seeds"]):
        if s in seed_results:
            continue

        print(f"  Seed {s} ({i+1}/{n_seeds})...", end="", flush=True)
        t_seed_start = time.perf_counter()

        # Episode generation
        gen_rng = np.random.default_rng(s + 200)
        seed_episodes = []
        for _ in range(n_eps_per_seed):
            basin_idx = int(gen_rng.integers(0, cfg["K"]))
            # Inline episode generation
            eval_model_gen = make_evaluation_model(cfg, gen_rng)
            basin = eval_model_gen.basins[basin_idx]
            x = np.zeros(n)
            mask_sched = observation_schedule(T, m_obs, gen_rng, profile_name=cfg["profile_name"])
            x_traj, y_traj, z_traj, u_traj, mask_traj = [], [], [], [], []
            for t_step in range(T):
                u = np.zeros(m_u)
                w = gen_rng.multivariate_normal(np.zeros(n), basin.Q)
                x_next = basin.A @ x + basin.B @ u + w + basin.b
                R_t = heteroskedastic_R(basin.R, x, mask_sched[t_step], t_step)
                y = generate_observation(x, basin.C, basin.c, R_t, mask_sched[t_step], gen_rng)
                x_traj.append(x.copy())
                y_traj.append(y)
                z_traj.append(basin_idx)
                u_traj.append(u)
                mask_traj.append(mask_sched[t_step])
                x = x_next
            ep = {
                "x_true": np.array(x_traj),
                "z_true": np.array(z_traj),
                "y": np.array(y_traj),
                "mask": np.array(mask_traj),
                "u": np.array(u_traj),
                "seed": s,
                "basin_idx": basin_idx,
            }
            seed_episodes.append(ep)

        # Build eval model for this seed
        rng_sim = np.random.default_rng(s + 400)
        sim_model = make_evaluation_model(cfg, rng_sim)

        # Pre-compute LQR gains
        Q_lqr = np.eye(n)
        R_lqr = np.eye(m_u) * lambda_u
        p_basin = pooled_basin(sim_model)
        try:
            K_pooled, _ = dlqr(p_basin.A, p_basin.B, Q_lqr, R_lqr)
        except Exception:
            K_pooled = np.zeros((m_u, n))

        K_basin_lqr = []
        for b in sim_model.basins:
            try:
                K_b, _ = dlqr(b.A, b.B, Q_lqr, R_lqr)
            except Exception:
                K_b = np.zeros((m_u, n))
            K_basin_lqr.append(K_b)

        # Safety check invariants
        safety_std_per_basin = {}
        for k_idx, basin_k in enumerate(sim_model.basins):
            y_cov_k = basin_k.C @ P_safety @ basin_k.C.T + basin_k.R
            safety_std_per_basin[k_idx] = np.sqrt(np.maximum(np.diag(y_cov_k), 1e-12))
        eps_safe_val = float(cfg["eps_safe"])

        def _safety_violation(x_state, basin_k_idx):
            bk = sim_model.basins[basin_k_idx]
            y_mean = bk.C @ x_state + bk.c
            std = safety_std_per_basin[basin_k_idx]
            lower = _norm_dist.cdf((y_lo - y_mean) / std)
            upper = _norm_dist.cdf((y_mean - y_hi) / std)
            return float(np.max(lower + upper)) > eps_safe_val

        target_sets = [build_target_set(k, cfg) for k in range(len(sim_model.basins))]

        # Run simulation
        ep_costs_hdr = []
        ep_costs_pe = []
        ep_basins = []
        seed_offset = i * n_eps_per_seed

        for ep_idx_local, ep in enumerate(seed_episodes):
            ep_idx = seed_offset + ep_idx_local
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
            obs_noise_raw = np.empty((T, m_obs))
            for _t in range(T):
                _obs_rng = np.random.default_rng(cfg["seeds"][0] + 7000 + ep_idx * T + _t)
                obs_noise_raw[_t] = _obs_rng.standard_normal(m_obs)

            imm_filt_hdr = IMMFilter.for_hard_regime(sim_model)
            imm_filt_pe = IMMFilter.for_hard_regime(sim_model)
            x_hdr = x_init.copy()
            x_pe = x_init.copy()
            cost_hdr, cost_pe = 0.0, 0.0
            used_burden_hdr, used_burden_pe = 0.0, 0.0
            u_prev_hdr = np.zeros(m_u)
            u_prev_pe = np.zeros(m_u)

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

                # HDR observations
                R_t_hdr = heteroskedastic_R(basin_R, x_hdr, mask_t, t)
                diag_R_hdr = np.diag(R_t_hdr)
                y_t_hdr = basin_C @ x_hdr + basin_c + noise_t * np.sqrt(diag_R_hdr)
                y_t_hdr = np.where(mask_t_bool, y_t_hdr, np.nan)
                mask_int_hdr = (~np.isnan(y_t_hdr)).astype(int)
                y_clean_hdr = np.where(np.isnan(y_t_hdr), 0.0, y_t_hdr)
                imm_state_hdr = imm_filt_hdr.step(y_clean_hdr, mask_int_hdr, u_prev_hdr)
                x_hat_hdr = imm_state_hdr.mixed_mean
                P_hat_hdr = imm_state_hdr.mixed_cov

                # PE observations
                R_t_pe = heteroskedastic_R(basin_R, x_pe, mask_t, t)
                diag_R_pe = np.diag(R_t_pe)
                y_t_pe = basin_C @ x_pe + basin_c + noise_t * np.sqrt(diag_R_pe)
                y_t_pe = np.where(mask_t_bool, y_t_pe, np.nan)
                mask_int_pe = (~np.isnan(y_t_pe)).astype(int)
                y_clean_pe = np.where(np.isnan(y_t_pe), 0.0, y_t_pe)
                imm_state_pe = imm_filt_pe.step(y_clean_pe, mask_int_pe, u_prev_pe)
                x_hat_pe = imm_state_pe.mixed_mean

                # HDR: MPC
                est_bi = imm_state_hdr.map_mode
                est_basin = sim_model.basins[est_bi]
                est_target = target_sets[est_bi]
                mpc_res = solve_mode_a(
                    x_hat_hdr, P_hat_hdr, est_basin, est_target,
                    kappa_hat=0.6, config=cfg, step=t, used_burden=used_burden_hdr,
                )
                u_hdr = mpc_res.u

                # PE: pooled LQR
                u_pe = -K_pooled @ x_hat_pe
                u_pe, _ = apply_control_constraints(u_pe, cfg, step=t, used_burden=used_burden_pe)

                # Costs
                cost_hdr += float(np.dot(x_hdr, x_hdr) + lambda_u * np.dot(u_hdr, u_hdr))
                cost_pe += float(np.dot(x_pe, x_pe) + lambda_u * np.dot(u_pe, u_pe))

                # Evolve
                w = process_noise[t]
                x_hdr = basin_A @ x_hdr + basin_B @ u_hdr + w + basin_b
                x_pe = basin_A @ x_pe + basin_B @ u_pe + w + basin_b
                used_burden_hdr += float(np.sum(np.abs(u_hdr)))
                used_burden_pe += float(np.sum(np.abs(u_pe)))
                u_prev_hdr = u_hdr
                u_prev_pe = u_pe

            ep_costs_hdr.append(cost_hdr)
            ep_costs_pe.append(cost_pe)

        # Write per-seed checkpoint
        partial_data = {
            "seed": s,
            "n_episodes": n_eps_per_seed,
            "ep_costs_hdr": ep_costs_hdr,
            "ep_costs_pe": ep_costs_pe,
            "ep_basins": ep_basins,
        }
        _atomic_write_json(out_dir / f"seed_{s:05d}_partial.json", partial_data)
        seed_results[s] = partial_data

        elapsed = time.perf_counter() - t_seed_start
        print(f" done ({elapsed:.1f}s)")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n  Aggregating results across all seeds...")
    all_hdr, all_pe, all_basins, all_seed_labels = [], [], [], []
    for s in cfg["seeds"]:
        sd = seed_results[s]
        all_hdr.extend(sd["ep_costs_hdr"])
        all_pe.extend(sd["ep_costs_pe"])
        all_basins.extend(sd["ep_basins"])
        all_seed_labels.extend([s] * sd["n_episodes"])

    costs_hdr = np.array(all_hdr)
    costs_pe = np.array(all_pe)
    basins_arr = np.array(all_basins)
    seed_labels = np.array(all_seed_labels)

    with np.errstate(divide="ignore", invalid="ignore"):
        gains_all = np.where(costs_pe > 1e-12, (costs_pe - costs_hdr) / costs_pe, 0.0)

    mask_mal = basins_arr == 1
    gains_mal = gains_all[mask_mal]
    seeds_mal = seed_labels[mask_mal]

    mean_gain = float(np.mean(gains_mal)) if len(gains_mal) > 0 else 0.0
    n_total = len(gains_mal)

    # a. Mean gain
    print(f"    Mean gain (maladaptive): {mean_gain:+.4f}")
    print(f"    N maladaptive episodes: {n_total}")

    # b. Episode-level bootstrap CI
    ci_episode = _episode_bootstrap_ci(gains_mal, n_boot=10_000, ci=0.95, rng_seed=42)
    print(f"    Episode-level 95% CI: [{ci_episode[0]:+.4f}, {ci_episode[1]:+.4f}]")

    # c. Seed-cluster bootstrap CI
    ci_cluster = _cluster_bootstrap_ci(seeds_mal, gains_mal, n_boot=10_000, ci=0.95, rng_seed=43)
    print(f"    Cluster 95% CI:       [{ci_cluster[0]:+.4f}, {ci_cluster[1]:+.4f}]")

    # d. ICC
    icc = _compute_icc(seeds_mal, gains_mal)
    print(f"    ICC: {icc:.4f}")

    # e. DEFF
    unique_seeds_mal = np.unique(seeds_mal)
    n_per_cluster = n_total / max(len(unique_seeds_mal), 1)
    deff = 1.0 + (n_per_cluster - 1) * icc
    print(f"    DEFF: {deff:.4f} (n_per_cluster={n_per_cluster:.1f})")

    # f. Effective N
    effective_n = n_total / deff
    print(f"    Effective N: {effective_n:.1f}")

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "mean_gain": mean_gain,
        "ci_episode": list(ci_episode),
        "ci_cluster": list(ci_cluster),
        "icc": icc,
        "deff": deff,
        "effective_n": effective_n,
        "n_seeds": n_seeds,
        "n_episodes_per_seed": n_eps_per_seed,
        "n_total_episodes": n_total,
        "n_total_all_basins": len(gains_all),
        "bootstrap_n_resamples": 10_000,
        "ci_episode_width": ci_episode[1] - ci_episode[0],
        "ci_cluster_width": ci_cluster[1] - ci_cluster[0],
        "ci_widening_factor": (ci_cluster[1] - ci_cluster[0]) / max(ci_episode[1] - ci_episode[0], 1e-12),
        "n_per_cluster_mean": n_per_cluster,
    }

    report_dir = ROOT / "results" / "stage_04"
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(report_dir / "cluster_ci_report.json", report)
    print(f"\n  Wrote cluster_ci_report.json")

    return report


# ── Stage 10 multi-seed wrapper ───────────────────────────────────────────────
def run_stage_10_multiseed(n_seeds: int = 10) -> dict:
    """Run Stage 10 with multiple seeds to generate per-seed uncertainty bars."""
    from hdr_validation.stages.stage_10_mode_b_sweep import (
        _BENCHMARK_B_TRANSITION,
        _simulate_trajectory,
        _true_maladaptive_prob,
        inject_miscalibration,
        _MALADAPTIVE_STATES,
    )

    print(f"\n  Stage 10 multi-seed: {n_seeds} seeds")
    seeds = list(range(42, 42 + n_seeds))
    N_sim = 5000
    T = 50
    p_A_base = 0.70
    k_calib = 1.0
    q_min = 0.15
    R_Brier_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
    P = _BENCHMARK_B_TRANSITION.copy()

    all_seed_results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        trajectories = [_simulate_trajectory(T, rng, P) for _ in range(N_sim)]

        seed_sweep = []
        for R_target in R_Brier_levels:
            threshold_fixed = p_A_base
            threshold_robust = p_A_base + k_calib * R_target
            fp_fixed, fp_robust = 0, 0
            fn_fixed, fn_robust = 0, 0
            fn_denom, fp_denom = 0, 0

            rng_calib = np.random.default_rng(int(1000 * (1 + R_target) * 10) + seed)
            for traj in trajectories:
                for state in traj:
                    is_mal = state in _MALADAPTIVE_STATES
                    p_true = _true_maladaptive_prob(state)
                    p_hat = inject_miscalibration(p_true, R_target, rng_calib)

                    if is_mal:
                        q_spont = 0.08 + rng_calib.random() * 0.05
                        is_q_below = True
                    else:
                        q_spont = 0.20 + rng_calib.random() * 0.40
                        is_q_below = q_spont < q_min

                    if not is_mal:
                        fp_denom += 1
                        if p_hat >= threshold_fixed:
                            fp_fixed += 1
                        if p_hat >= threshold_robust:
                            fp_robust += 1

                    if is_mal and is_q_below:
                        fn_denom += 1
                        if p_hat < threshold_fixed:
                            fn_fixed += 1
                        if p_hat < threshold_robust:
                            fn_robust += 1

            seed_sweep.append({
                "R_Brier_target": R_target,
                "FP_rate_fixed": round(fp_fixed / max(fp_denom, 1), 4),
                "FP_rate_robust": round(fp_robust / max(fp_denom, 1), 4),
                "FN_rate_fixed": round(fn_fixed / max(fn_denom, 1), 4),
                "FN_rate_robust": round(fn_robust / max(fn_denom, 1), 4),
            })

        all_seed_results.append({"seed": seed, "sweep": seed_sweep})
        print(f"    Seed {seed} done")

    out_dir = ROOT / "results" / "stage_10"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "multiseed_sweep.json", {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "per_seed_results": all_seed_results,
    })
    print(f"  Wrote stage_10/multiseed_sweep.json")
    return {"n_seeds": n_seeds, "per_seed_results": all_seed_results}


# ── Stage 15 multi-seed wrapper ───────────────────────────────────────────────
def run_stage_15_multiseed(n_seeds: int = 10) -> dict:
    """Run Stage 15 with multiple seeds to generate per-seed uncertainty bars."""
    from hdr_validation.model.slds import make_evaluation_model
    from hdr_validation.defaults import DEFAULTS

    print(f"\n  Stage 15 multi-seed: {n_seeds} seeds")
    seeds = list(range(101, 101 + n_seeds))
    n_scenarios = 5
    sigma_values = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    T = 50

    all_seed_results = []

    for seed in seeds:
        cfg = dict(DEFAULTS)
        cfg["max_dwell_len"] = 64
        rng = np.random.default_rng(seed)
        model = make_evaluation_model(cfg, rng)

        rmse_values = []
        for sigma_proxy in sigma_values:
            rmse_per_scenario = []
            for sc in range(n_scenarios):
                basin = model.basins[0]
                x_true = rng.normal(size=cfg["state_dim"]) * 0.1
                errors_sq = []
                for t_step in range(T):
                    x_true = basin.A @ x_true + rng.normal(scale=0.1, size=cfg["state_dim"])
                    y_direct = basin.C @ x_true + rng.normal(scale=0.1, size=cfg["obs_dim"])
                    y_proxy = y_direct + rng.normal(scale=sigma_proxy, size=cfg["obs_dim"])
                    try:
                        x_hat = np.linalg.lstsq(basin.C, y_proxy - basin.c, rcond=None)[0]
                    except Exception:
                        x_hat = np.zeros(cfg["state_dim"])
                    errors_sq.append(float(np.sum((x_hat - x_true)**2)))
                rmse_per_scenario.append(float(np.sqrt(np.mean(errors_sq))))
            rmse_values.append(float(np.mean(rmse_per_scenario)))

        all_seed_results.append({
            "seed": seed,
            "sigma_values": sigma_values,
            "rmse_values": rmse_values,
        })
        print(f"    Seed {seed} done (RMSE@0.5={rmse_values[3]:.3f})")

    out_dir = ROOT / "results" / "stage_15"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "multiseed_results.json", {
        "n_seeds": n_seeds,
        "seeds": seeds,
        "sigma_values": sigma_values,
        "per_seed_results": all_seed_results,
    })
    print(f"  Wrote stage_15/multiseed_results.json")
    return {"n_seeds": n_seeds, "per_seed_results": all_seed_results}


# ── Threshold claims audit ────────────────────────────────────────────────────
def audit_threshold_claims(report: dict) -> str:
    """Check whether any claims depend on CI lower bound >= 0.03."""
    lines = [
        "Threshold Claims Audit — Stage 04 Cluster-Aware CI Analysis",
        "=" * 62,
        "",
        f"n_seeds: {report['n_seeds']}",
        f"n_episodes_per_seed: {report['n_episodes_per_seed']}",
        f"n_total_maladaptive_episodes: {report['n_total_episodes']}",
        "",
        "Episode-level bootstrap 95% CI:",
        f"  [{report['ci_episode'][0]:+.4f}, {report['ci_episode'][1]:+.4f}]",
        f"  Width: {report['ci_episode_width']:.4f}",
        "",
        "Seed-cluster bootstrap 95% CI:",
        f"  [{report['ci_cluster'][0]:+.4f}, {report['ci_cluster'][1]:+.4f}]",
        f"  Width: {report['ci_cluster_width']:.4f}",
        f"  Widening factor: {report['ci_widening_factor']:.2f}x",
        "",
        f"ICC: {report['icc']:.4f}",
        f"DEFF: {report['deff']:.4f}",
        f"Effective N: {report['effective_n']:.1f}",
        "",
        "-" * 62,
        "CLAIMS DEPENDING ON CI LOWER BOUND >= 0.03:",
        "-" * 62,
        "",
    ]

    # Check Claim 2 (Mode A improvement)
    ep_lo = report["ci_episode"][0]
    cl_lo = report["ci_cluster"][0]

    lines.append("Claim 2 (CLAIM_MATRIX.md):")
    lines.append("  Criterion: 95% CI lower bound >= +0.03")
    lines.append(f"  Episode-level CI lower: {ep_lo:+.4f} {'PASS' if ep_lo >= 0.03 else 'FAIL'}")
    lines.append(f"  Cluster-level CI lower: {cl_lo:+.4f} {'PASS' if cl_lo >= 0.03 else 'FAIL'}")

    if ep_lo >= 0.03 and cl_lo < 0.03:
        lines.append("")
        lines.append("  ** WARNING: Episode-level CI passes but cluster-level CI fails!")
        lines.append("     The original claim relied on episode-level bootstrap which")
        lines.append("     does not account for within-seed correlation (ICC={:.4f}).".format(report["icc"]))
        lines.append("     The cluster-aware CI is the correct inference procedure.")
        lines.append("     RECOMMENDATION: Update claim to use cluster bootstrap CI,")
        lines.append("     or increase n_seeds further to narrow the cluster CI.")
    elif cl_lo >= 0.03:
        lines.append("")
        lines.append("  Claim is ROBUST to cluster-aware resampling.")
    else:
        lines.append("")
        lines.append("  ** Both CI types fail the 0.03 threshold.")
        lines.append("     Consider weakening the claim or running more seeds.")

    lines.append("")
    lines.append("-" * 62)
    lines.append("HIGHPOWER SUMMARY (CLAIM_MATRIX.md line 115):")
    lines.append("  States: gain >= +0.03; CI lower >= +0.03")
    lines.append(f"  Mean gain: {report['mean_gain']:+.4f} {'PASS' if report['mean_gain'] >= 0.03 else 'FAIL'}")
    lines.append(f"  Cluster CI lower: {cl_lo:+.4f} {'PASS' if cl_lo >= 0.03 else 'FAIL'}")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  WP-2.3: CLUSTER-AWARE BOOTSTRAP CI ANALYSIS")
    print("=" * 62)
    t0 = time.perf_counter()

    try:
        # Task 1-3: Stage 04 with 100 seeds + cluster CI
        report = run_stage_04_cluster()

        # Task 4: Stage 10 and 15 with 10 seeds each
        run_stage_10_multiseed(n_seeds=10)
        run_stage_15_multiseed(n_seeds=10)

        # Task 5: cluster_ci_report.json already written above

        # Task 6: Threshold claims audit
        audit_text = audit_threshold_claims(report)
        report_dir = ROOT / "results" / "stage_04"
        (report_dir / "threshold_claims_audit.txt").write_text(audit_text)
        print(f"\n  Wrote threshold_claims_audit.txt")
        print("\n" + audit_text)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("  Done.")
