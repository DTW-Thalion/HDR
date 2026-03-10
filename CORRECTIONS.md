# HDR Claim Matrix — v5.0

**Framework version:** HDR v5.0  
**Validation suite version:** `hdr_validation_v5`  
**Claims 1–10:** Inherited and reformulated from v4.3  
**Claims 11–14:** New ICI claims added in v5.0

---

## Inherited Claims (reformulated where noted)

| # | Claim | Stage(s) | Notes |
|---|-------|----------|-------|
| 1 | **ICI correctly identifies when Mode A guarantees hold** (v4.3: "Non-oracle inference adequate for control") | 03b, 04 | `hdr_vs_pooled_estimated_gain_maladaptive >= +0.03`; `hdr_maladaptive_win_rate >= 0.70` |
| 2 | Mode A improves over baselines without exceeding safety budget | 04 | `hdr_vs_pooled_estimated_gain_maladaptive >= +0.03`; `hdr_maladaptive_win_rate >= 0.70`; safety delta within ISS bound |
| 3 | τ̃ tracks true recovery burden (Spearman ρ ≥ 0.70) | 01 | τ̃ rank correlation ≥ 0.70 |
| 4 | Chance-constraint tightening calibrated in Gaussian settings | 01, 04 | Abs error ≤ 0.015; heavy-tail degradation < 0.05 |
| 5 | Mode error degradation consistent with √μ̄ ISS scaling (v4.3: "√μ̄ degradation") | 01, 04 | Continuous μ̄ sweep; γ(μ̄) monotone |
| 6 | Stability under drifting S*(t) consistent with linear degradation | 04 | Slope > 0, R² ≥ 0.75 |
| 7 | Mode B improves escape when Mode C pre-emption confirms adequate inference quality | 03b, 05 | safe_trigger_fraction; mode_c_should_preempt |
| 8 | Mode B acceptably close to exact DP (including ε_H term) | 05 | Abs gap ≤ 0.10; ε_H included in bound |
| 9 | Coherence penalty behaves as designed; w3 calibrated | 06 | Structural tests; Pareto-efficient w3 identified |
| 10 | Identifiability improves with perturbations, priors, dither (reinterpreted vs regime boundary) | 03 | Gains reported relative to T_k_eff / ω_min |

---

## New ICI Claims (v5.0)

| # | Claim | Stage(s) | Evidence |
|---|-------|----------|----------|
| 11 | ICI correctly identifies operating regime and activates Mode C | 03b, 03c | Entry conditions consistent; 03c.2 priority test passes |
| 12 | Mode C improves T_k_eff and R_Brier within Fisher information bounds | 03c | Fisher proxy increases (03c.3); exit after improvement (03c.4) |
| 13 | p_A^robust reduces FP rate vs fixed p_A under miscalibrated posterior | 03b, 05 | p_A_robust > p_A_nominal; safe_trigger_fraction = 0 when below ω_min |
| 14 | Compound bound correctly predicts regime boundary | 01, 03, 07 | Formula verified; HDR params flagged; boundary sweep matches |

---

---

## Notes

**Claims 1 and 2** are evaluated on maladaptive-basin episodes (basin index 1, rho=0.96) because HDR is a remediation framework. The fair baseline is `pooled_lqr_estimated` (IMM x_hat), not oracle-state `pooled_lqr`.

*All results are in silico only. No biological or clinical validity is implied.*
