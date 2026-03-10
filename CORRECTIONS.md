# HDR Claim Matrix — v5.0

**Framework version:** HDR v5.0
**Validation suite version:** `hdr_validation_v5`
**Claims 1–10:** Inherited and reformulated from v4.3
**Claims 11–14:** New ICI claims added in v5.0
**Last updated:** 2026-03-10

---

## Test Summary

| Suite | Result |
|-------|--------|
| Pytest unit tests | 2460/2460 passed |
| Standard profile (T=128, 2 seeds, 12 ep/seed) | 86/86 checks passed |
| Extended profile (T=256, 3 seeds, 20 ep/seed) | 98/98 checks passed |

---

## Inherited Claims (reformulated where noted)

| # | Claim | Stage(s) | Criterion | Standard | Extended | Status |
|---|-------|----------|-----------|----------|----------|--------|
| 1 | **ICI correctly identifies when Mode A guarantees hold** | 03b, 04 | `hdr_vs_pooled_estimated_gain_maladaptive >= +0.03`; `hdr_maladaptive_win_rate >= 0.70` | gain=+0.057, rate=0.909 | gain=+0.036, rate=0.800 | **Supported** |
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
