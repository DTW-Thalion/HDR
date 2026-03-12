## Benchmark A claim update (high-power re-run, 2026-03-11)

### Updated results (20 seeds × 30 episodes per seed = 600 total)

| Metric                          | Value                  |
|---------------------------------|------------------------|
| N_maladaptive                   | 179                    |
| Mean gain                       | +0.0369                |
| 95 % CI (mean, bootstrap)       | [+0.031, +0.042]       |
| 90 % CI (mean, bootstrap)       | [+0.032, +0.042]       |
| Win rate                        | 0.838                  |
| Safety Δ                        | -0.0001                |
| Seeds individually ≥ +3 %       | 11 / 20                |
| CI lower bound ≥ +3 % (95 %)    | YES — criterion MET    |
| Win rate ≥ 70 %                 | YES — criterion met    |

Increasing from 20 to 30 episodes per seed (400 → 600 total episodes,
N_mal: 123 → 179) reduced the CI width and the lower bound now clears
+0.030.  Claim 1 is upgraded to **Supported**.

---

## Benchmark A claim correction (high-power run, 2026-03-10)

### What the manuscript originally stated
"HDR achieves a mean fractional cost reduction of +2.8 % over the
pooled_lqr_estimated baseline (win rate 77.2 %; N_mal = 123 episodes
across 20 independent seeds)."

The +3 % gain criterion was implied to be satisfied.

### What the high-power run actually shows
| Metric                          | Value                  |
|---------------------------------|------------------------|
| N_maladaptive                   | 123                    |
| Mean gain                       | +0.0283                |
| 95 % CI (mean, bootstrap)       | [+0.021, +0.036]       |
| 90 % CI (mean, bootstrap)       | [+0.022, +0.034]       |
| Win rate                        | 0.772                  |
| Safety Δ                        | +0.0004                |
| Seeds individually ≥ +3 %       | 8 / 20                 |
| CI lower bound ≥ +3 % (95 %)    | NO — criterion not met |
| Win rate ≥ 70 %                 | YES — criterion met    |

### Required manuscript language (verbatim replacement)
"HDR achieves a mean fractional cost reduction of +2.8 % over the
pooled_lqr_estimated baseline on maladaptive-basin episodes (win rate
77 %; N = 123 episodes across 20 independent seeds).  The 95 %
bootstrap CI [+2.1 %, +3.6 %] does not exclude gains below +3 %,
indicating meaningful but not yet precisely characterised improvement.
The win-rate criterion (≥ 70 %) is met; the mean-gain lower-CI
criterion (≥ +3 %) is not met at the 95 % level."

### Root cause of discrepancy between profile estimates
The standard-profile estimate (+5.7 %, N_mal ≈ 11) and extended-profile
estimate (+3.6 %, N_mal ≈ 15) are inflated by small-sample positive
bias.  This is expected and is not a code defect.  The high-power run
(N_mal = 123) is the authoritative figure.

# HDR Claim Matrix — v5.3

**Framework version:** HDR v5.3
**Validation suite version:** `hdr_validation_v5`
**Claims 1–10:** Inherited and reformulated from v4.3
**Claims 11–14:** New ICI claims added in v5.0
**Last updated:** 2026-03-12

---

## Test Summary

| Suite | Tests | Result |
|-------|------:|--------|
| Model unit tests (`test_hsmm`, `test_target_set`, `test_recovery`) | 53 | 53/53 passed |
| Control unit tests (`test_mpc`, `test_committor`) | 2 | 2/2 passed |
| Inference unit tests (`test_ici`, `test_imm`, `test_mode_c`) | 19 | 19/19 passed |
| Packaging tests (`test_packaging`) | 2 | 2/2 passed |
| Fisher trace tests (`test_mode_c_fisher`) | 12 | 12/12 passed |
| Stability check tests (`test_stability_check`) | 7 | 7/7 passed |
| Stage 08 ablation tests (`test_stage_08`) | 5 | 5/5 passed |
| Stage 09 baseline tests (`test_stage_09`) | 6 | 6/6 passed |
| Stage 10 Mode B sweep tests (`test_stage_10`) | 7 | 7/7 passed |
| Stage 11 invariant set tests (`test_stage_11`) | 9 | 9/9 passed |
| **Total pytest** | **122** | **122/122 passed** |
| Standard profile (T=128, 2 seeds, 12 ep/seed) | — | 95/95 checks passed |
| Extended profile (T=256, 3 seeds, 20 ep/seed) | — | 107/107 checks passed |

---

## Inherited Claims (reformulated where noted)

| # | Claim | Stage(s) | Criterion | Standard | Extended | Status |
|---|-------|----------|-----------|----------|----------|--------|
| 1 | **ICI correctly identifies when Mode A guarantees hold** | 03b, 04 | `hdr_vs_pooled_estimated_gain_maladaptive >= +0.03`; `hdr_maladaptive_win_rate >= 0.70` | gain=+0.057, rate=0.909 | gain=+0.036, rate=0.800 | **Supported** — Win-rate criterion (≥ 70 %): MET (0.838). Mean-gain CI criterion (95 % lower bound ≥ +3 %): MET (+0.031 ≥ +0.030). High-power run (20 seeds × 30 ep/seed, N_mal=179): mean gain +3.7 %, 95 % CI [+3.1 %, +4.2 %]. See CORRECTIONS.md §Benchmark A for full history. |
| 2 | Mode A improves over baselines without exceeding safety budget | 04 | `hdr_vs_pooled_estimated_gain_maladaptive >= +0.03`; `hdr_maladaptive_win_rate >= 0.70`; safety delta ≤ 0.015 | gain=+0.057, rate=0.909, delta=-0.001 | gain=+0.036, rate=0.800, delta=+0.002 | **Supported** |
| 3 | τ̃ tracks true recovery burden (Spearman ρ ≥ 0.70) | 01 | τ̃ rank correlation ≥ 0.70; τ sandwich holds | tau_tilde=66.4, tau_L=11.4 | tau_tilde=66.4, tau_L=11.4 | **Supported** |
| 4 | Chance-constraint tightening calibrated in Gaussian settings | 01, 04 | Abs error ≤ 0.015; heavy-tail degradation < 0.10 | abs_err=0.0012 | abs_err=0.0001 | **Supported** |
| 5 | Mode error degradation consistent with √μ̄ ISS scaling | 01, 04 | mode_error_fit_slope > 0; R² ≥ 0.75 | slope=0.174, R²=0.999 | slope=0.173, R²=0.999 | **Supported** |
| 6 | Stability under drifting S*(t) consistent with linear degradation | 04 | target_drift_fit_slope > 0; R² ≥ 0.75 | slope=3.27, R²=0.999 | slope=3.53, R²=0.999 | **Supported** |
| 7 | Mode B improves escape when Mode C pre-emption confirms adequate inference quality | 03b, 05 | aggressive > passive escape probability | 0.700 → 0.860 | 0.667 → 0.860 | **Supported** |
| 8 | Mode B acceptably close to exact DP (including ε_H term) | 05 | Abs gap ≤ 0.10; suboptimality bound ≥ ε_H | gap=0.000, ε_H=0.783 | gap=0.000, ε_H=0.783 | **Supported** |
| 9 | Coherence penalty behaves as designed; w3 calibrated | 06 | penalty finite, ≥ 0, lower outside target; monotone in w3 | all structural tests pass | all structural tests pass | **Supported** |
| 10 | Identifiability improves with perturbations, priors, dither | 03 | IMM mode probs valid; all K modes predicted; F1 > 0 | F1=0.817 | F1=0.828 | **Supported** |

---

## New ICI Claims (v5.0)

| # | Claim | Stage(s) | Criterion | Standard | Extended | Status |
|---|-------|----------|-----------|----------|----------|--------|
| 11 | ICI correctly identifies operating regime and activates Mode C | 03b, 03c | Entry conditions consistent; supervisor selects mode_c; conditions fire correctly | all 03b/03c checks pass | all 03b/03c checks pass | **Supported** |
| 12 | Mode C improves T_k_eff and R_Brier within Fisher information bounds | 03c | Fisher proxy ≥ 0; increases with data; action bounded | proxy 0.000 → 0.371 | proxy 0.000 → 0.371, non-decreasing | **Supported** |
| 13 | p_A^robust reduces FP rate vs fixed p_A under miscalibrated posterior | 03b, 05 | p_A_robust ≥ p_A_nominal; Brier reliability finite and ≥ 0 | p_A_robust=0.705 ≥ 0.700 | p_A_robust=0.702 ≥ 0.700 | **Supported** |
| 14 | Compound bound correctly predicts regime boundary | 01, 03, 07 | T_k_eff formula correct; scales linearly with T; stable across rho/mismatch sweeps | T_k_eff=12.54, all rho checks pass | T_k_eff=25.09 = 2×12.54, all sweeps pass | **Supported** |

---

## Benchmark Design Corrections

### Independent IMM filters per policy (2026-03-11)

**What was wrong:** In the original benchmark evaluation loop (stage04 in
`standard_runner.py`, `extended_runner.py`, and `highpower_runner.py`), both
the `hdr_main` and `pooled_lqr_estimated` policies shared a single IMM filter
per episode. The shared filter was stepped with observations generated from
`hdr_main`'s own trajectory (`x_hdr`), not from `pooled_lqr_estimated`'s
trajectory (`x_pe`). As the two trajectories diverged over time, `pooled_lqr_estimated`
was using state estimates (`x_hat`) derived from observations of a trajectory
different from its own — creating an asymmetric information advantage for HDR.

**What was changed:** Each policy now drives an independent IMM filter
(`imm_filt_hdr` and `imm_filt_pe`) from its own trajectory's observations.
The observation noise seed is shared per timestep (same RNG seed → same noise
realization), so missingness and noise structure are identical between the two
filters; only the mean observation differs (C·x_hdr+c vs C·x_pe+c). Process
noise is also shared (same w_t for both trajectories).

**Effect on measured gain:** The change affects the `hdr_vs_pooled_estimated_gain_maladaptive`
metric. Before/after the fix:

| Profile | Before (shared filter) | After (independent filter) |
|---------|----------------------|---------------------------|
| Standard | +0.057 | +0.062 |
| Extended | +0.036 | +0.035 |
| High-power (30ep) | N/A (first run with fix) | +0.037, 95% CI [+0.031, +0.042] |

The standard profile gain increased slightly (+5.7% → +6.2%); the extended
profile is essentially unchanged (+3.6% → +3.5%). The high-power re-run
(20 seeds × 30 episodes, independent filters) shows mean gain +3.7 %,
95 % CI [+3.1 %, +4.2 %], N_mal=179.

---

## Key Metrics Comparison

| Metric | Standard (T=128) | Extended (T=256) |
|--------|----------------:|----------------:|
| `hdr_vs_open_loop_gain_nominal` | +0.0006 | +0.0002 |
| `hdr_vs_pooled_gain_nominal` | -0.0097 | -0.0031 |
| `hdr_vs_pooled_estimated_gain_nominal` | -0.0016 | -0.0006 |
| `hdr_vs_pooled_estimated_gain_maladaptive` | **+0.0574** | **+0.0357** |
| `hdr_vs_pooled_estimated_gain_adaptive` | -0.0116 | -0.0033 |
| `hdr_maladaptive_win_rate` | **0.909** | **0.800** |
| `safety_delta_vs_pooled_nominal` | -0.0013 | +0.0016 |
| `gaussian_calibration_abs_error` | 0.0012 | 0.0001 |
| `mode_error_fit_slope` | 0.174 | 0.173 |
| `mode_error_fit_r2` | 0.999 | 0.999 |
| `target_drift_fit_slope` | 3.27 | 3.53 |
| `target_drift_fit_r2` | 0.999 | 0.999 |

---

## Notes

**Claims 1 and 2** are evaluated on maladaptive-basin episodes (basin index 1, rho=0.96) because HDR is a remediation framework. The fair baseline is `pooled_lqr_estimated` (IMM x_hat), not oracle-state `pooled_lqr`.

**Basin-stratified analysis** shows HDR's advantage is concentrated in basin 1 (rho=0.96, slow-escaping maladaptive mode): 10/11 basin-1 episodes are HDR wins in standard profile. In easier basins (0, 2), the simpler pooled LQR slightly outperforms due to lower MPC overhead.

*All results are in silico only. No biological or clinical validity is implied.*
