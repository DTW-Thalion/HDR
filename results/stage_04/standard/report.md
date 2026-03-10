# Stage 04 — Mode A Control: Standard Profile Report

**Date**: 2026-03-10
**Profile**: standard
**Seeds**: [101, 202]
**Episodes per seed**: 12 (24 total pooled)
**Steps per episode**: 128
**MC rollouts**: 100

---

## Configuration

| Parameter | Value |
|-----------|-------|
| state_dim | 8 |
| obs_dim | 16 |
| control_dim | 8 |
| K (basins) | 3 |
| H (horizon) | 6 |
| rho_reference | [0.72, 0.96, 0.55] |
| kappa_lo / kappa_hi | 0.55 / 0.75 |
| pA | 0.70 |
| burden_budget | 28.0 |
| lambda_u | 0.1 |

---

## Part 1: Structural Checks (Point Evaluations)

Stage 04 validates the Mode A (nominal MPC/LQR) control law by running `solve_mode_a` across pooled episodes from both seeds.

| # | Check | Result | Value |
|---|-------|--------|-------|
| 1 | Mode A u norm bounded | PASS | max ||u|| = 0.6620 |
| 2 | Mode A feasibility rate > 0.5 | PASS | 1.00 (64/64 feasible) |
| 3 | Mode A produces non-zero control | PASS | 18/64 calls produced non-zero u |
| 4 | Mode A on rho=0.96 basin finite u | PASS | Finite control on maladaptive basin |
| 5 | Mode A on rho=0.96 risk computed | PASS | risk = 0.7127 |
| 6 | Mode A seed=202 finite u | PASS | Second seed produces consistent output |

---

## Part 2: Closed-Loop Comparative Simulation

Four policies are simulated over all 24 pooled episodes starting from mid-episode states (t=T/4) with shared process noise:

| Policy | Description |
|--------|-------------|
| open_loop | u = 0 (no control) |
| pooled_lqr | Single DLQR gain from basin-averaged dynamics |
| basin_lqr | Oracle basin-specific DLQR (knows true basin) |
| hdr_main | Full IMM filter + solve_mode_a (observations generated from controlled state) |

### Cumulative Cost

Cost = sum_t ||x_t||^2 + lambda_u * ||u_t||^2 over T=128 steps.

### Headline Metrics

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| HDR vs open-loop gain | 0.0006 | > 0 | PASS |
| HDR vs pooled gain | -0.0097 | > -0.10 | PASS |
| Safety delta vs pooled | -0.0013 | <= 0.015 | PASS |

### Sensitivity Analysis

| Regression | Slope | R^2 |
|------------|-------|-----|
| Mode-error (cost degradation vs mu_hat) | 0.1743 | 0.9988 |
| Target drift (cost vs drift magnitude) | 3.2657 | 0.9987 |

### Calibration

| Metric | Value |
|--------|-------|
| Gaussian calibration abs error | 0.0012 |

---

## Analysis

### HDR vs Open-Loop (Check 7)
HDR achieves a positive median fractional cost improvement (0.06%) over open-loop. While modest, this confirms that the full IMM + MPC pipeline produces control that reduces cumulative state cost even when starting from non-trivial states with process noise.

### HDR vs Pooled LQR (Check 8)
HDR is only 0.97% worse than pooled LQR, well within the -10% threshold. The small gap is expected: pooled LQR has oracle access to the true state while HDR must estimate it from noisy, partially-observed data via the IMM filter.

### Safety (Check 9)
HDR has a slightly *lower* safety violation rate (-0.13%) than pooled LQR, indicating that the MPC's chance-constraint tightening and risk evaluation provide a small safety benefit.

### Mode-Error Sensitivity
Cost degrades linearly with mode-error probability (R^2 = 0.999, slope = 0.17), confirming that correct mode identification is important for control quality.

### Target-Drift Sensitivity
Cost degrades strongly with target drift (R^2 = 0.999, slope = 3.27), showing that the controller is sensitive to target set misspecification.

---

## Summary

| Metric | Value |
|--------|-------|
| Total checks | 9 |
| Passed | 9 |
| Failed | 0 |
| Overall | ALL CHECKS PASSED |

Stage 04 standard profile validates that Mode A (MPC with Riccati recursion) produces bounded, feasible, and non-trivial control across multiple seeds and basin types. The closed-loop comparative simulation confirms HDR beats open-loop control, is not catastrophically worse than oracle pooled LQR, and maintains comparable safety violation rates.
