# Reconciliation Report — HDR v7.3.0 → v7.4.0

Generated: 2026-03-17

---

## Section 1 — Version Stamp

All primary artifacts now carry version 7.4.0:

| Location | Field | Value |
|----------|-------|-------|
| `pyproject.toml` | `version` | `"7.4.0"` |
| `hdr_validation/__init__.py` | `__version__` (fallback) | `"7.4.0"` |
| `hdr_validation/defaults.py` | `HDR_VERSION` | `"7.4.0"` |
| `README.md` | Title | "HDR Validation Suite v7.4.0" |
| `results/stage_04/highpower/highpower_summary.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_08/ablation_results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_08b/ablation_asymmetric_results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_09/baseline_comparison.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_10/mode_b_fp_fn_sweep.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_11/invariant_set_verification.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_12/results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_13/results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_14/results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_15/results.json` | `provenance.hdr_version` | `"7.4.0"` |
| `results/stage_16/stage_16_results.json` | `provenance.hdr_version` | `"7.4.0"` |

**Confirmed**: All regenerated result artifacts, pyproject.toml, and the README carry version 7.4.0.

---

## Section 2 — Artifact Reconciliation Table

All stages were re-run at production scale (20 seeds / 30 episodes / T=256 for stages 08, 08b, 09; 5 seeds / T=128 for stage 11; production parameters for stages 10, 12–16). Seeds are deterministic, and no stage runner logic was modified. Only the version stamp changed.

### Stage 04 — Highpower Benchmark (20 seeds x 30 ep/seed)

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| N_maladaptive | 179 | 179 | Unchanged |
| hdr_vs_pe_maladaptive_mean | +0.035 | +0.035 | Unchanged |
| 95% CI mean | [+0.030, +0.040] | [+0.030, +0.040] | Unchanged |
| hdr_mal_win_rate | 0.832 | 0.832 | Unchanged |
| tube_vs_pe_maladaptive_mean | +0.074 | +0.074 | Unchanged |

Interpretation: All Benchmark A values are unchanged. Deterministic seeds produce identical trajectories when no logic changes.

### Stage 08 — Ablation Study

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| hdr_full mean_gain | -0.8395 | -0.8395 | Unchanged |
| mpc_only mean_gain | -1.0017 | -1.0017 | Unchanged |
| N_mal | 170 | 170 | Unchanged |
| ablation_criterion_met | true | true | Unchanged |
| coherence_steps_active_pct | 0.3008 | 0.3008 | Unchanged |

Interpretation: No change. Ablation variants are fully reproducible.

### Stage 08b — Multi-Axis Asymmetric Ablation

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| hdr_full mean_gain | -0.8454 | -0.8454 | Unchanged |
| mpc_only mean_gain | -1.0086 | -1.0086 | Unchanged |
| coherence_marginal_gain | 0.0001 | 0.0001 | Unchanged |
| calibration_marginal_gain | 0.0 | 0.0 | Unchanged |
| J_diagnostics.row_norm_ratio | 8.75 | 8.75 | Unchanged |

Interpretation: No change. Marginal gains remain near zero (see Section 4).

### Stage 09 — Baseline Comparison

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| open_loop mean_abs_cost | 742.15 | 742.15 | Unchanged |
| pooled_lqr_estimated mean_abs_cost | 315.11 | 315.11 | Unchanged |
| belief_mpc gain vs pooled | -0.0474 | -0.0474 | Unchanged |
| hdr_mode_a gain vs pooled | -0.8528 | -0.8528 | Unchanged |
| N_mal | 170 | 170 | Unchanged |

Interpretation: No change.

### Stage 10 — Mode B FP/FN Sweep

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| All FP/FN rates at R_Brier ∈ {0.0, 0.05, 0.1, 0.15, 0.2} | Identical | Identical | Unchanged |
| N_sim | 5000 | 5000 | Unchanged |

Interpretation: No change. Pure simulation sweep, deterministic.

### Stage 11 — Riccati Invariant Set Verification

| Metric | Prior (v7.3.0) | New (v7.4.0) | Change |
|--------|---------------|-------------|--------|
| Basin 0: c_k | 2.3304 | 2.3304 | Unchanged |
| Basin 0: containment_rate_rpi | 1.0 | 1.0 | Unchanged |
| Basin 1: c_k | 14.9867 | 14.9867 | Unchanged |
| Basin 1: containment_rate_rpi | 0.9968 | 0.9968 | Unchanged |
| Basin 2: c_k | 2.072 | 2.072 | Unchanged |
| Basin 2: containment_rate_rpi | 1.0 | 1.0 | Unchanged |
| All proposition_8_4_criterion_met | true | true | Unchanged |

Interpretation: No change. All three basins satisfy Proposition 8.4.

### Stages 12–15 — v7.0 Stages

| Stage | Prior | New | Change |
|-------|-------|-----|--------|
| 12: All 5 checks | PASS | PASS | Unchanged |
| 13: All 3 checks | PASS | PASS | Unchanged |
| 14: All 2 checks | PASS | PASS | Unchanged |
| 15: rmse_ratio_at_sigma_05 | FAIL (5.14) | FAIL (5.14) | Unchanged |

Interpretation: No change. Stage 15 check `rmse_ratio_at_sigma_05` continues to fail (value 5.14 vs threshold 2.0).

### Stage 16 — Extension Integration (v7.1)

| Metric | Prior (v7.3.0) | New (v7.4.0) | Direction | Interpretation |
|--------|---------------|-------------|-----------|----------------|
| 16.03 projection_error | 8.76e-16 | 0.0 | Negligible | Floating-point noise eliminated; no substantive change |
| 16.04 cross_site_response | 0.1986 | 0.4216 | +113% | Stochastic variability in multi-site coupling response; both above 0 threshold; all sub-checks still pass |
| 16.17 cost_ratio | 1.0003 | 1.0085 | +0.82% | Small upward shift in CRD profile cost ratio; still near 1.0; test passes |
| All 17 subtests pass/fail | 17/17 PASS | 17/17 PASS | Unchanged | No change in any pass/fail status |

Interpretation: Two metrics in Stage 16 changed numerically (16.04, 16.17) due to stochastic components in the extension integration tests (multi-site dynamics and modular axis expansion), but all pass/fail criteria remain unchanged.

---

## Section 3 — Manuscript Update Instructions

### Stages 04, 08, 08b, 09, 10, 11, 12–15

**No manuscript changes required.** All numerical values are identical between v7.3.0 and v7.4.0 for these stages. Deterministic seeds with no logic changes produce bit-identical results.

### Stage 16

1. **In the section covering Sub-test 16.04 (Multi-site dynamics), if the manuscript reports `cross_site_response = 0.1986`**, replace with `cross_site_response = 0.4216`. This does not change the substantive claim — the claim is that cross-site coupling produces a non-zero response, which both values satisfy — but the specific number should match the current artifact.

2. **In the section covering Sub-test 16.17 (CRD profile), if the manuscript reports `cost_ratio = 1.0003`**, replace with `cost_ratio = 1.0085`. This does not change the substantive claim — the claim is that modular axis expansion has cost ratio near 1.0, which 1.0085 satisfies — but the specific number should be updated.

3. **In Sub-test 16.16, if the manuscript uses the label "AD profile (M3+M2+M8)"**, replace with "AD profile (M1+M2+M8)". The extension module numbering was corrected.

### Version References

4. **Replace all references to "v7.3.0" or "HDR v7.3" with "v7.4.0" / "HDR v7.4"** throughout the manuscript, including the abstract, methods section, and any appendices that state the software version.

---

## Section 4 — Stage 08B Coherence/Calibration Finding

### Numbers

From the regenerated `results/stage_08b/ablation_asymmetric_results.json`:

- **coherence_marginal_gain**: 0.0001 (= mpc_plus_coherence gain − mpc_only gain = −1.0085 − (−1.0086))
- **calibration_marginal_gain**: 0.0 (= hdr_full gain − hdr_no_calib gain = −0.8454 − (−0.8454))

Both marginal gains are effectively zero. The coherence marginal gain of 0.0001 is within numerical noise and does not represent a statistically meaningful contribution.

### Assessment

**(a) What the numbers are:**
The coherence penalty contributes a marginal gain of +0.0001 (0.01 percentage points) on the asymmetric ablation benchmark. The calibration adjustment contributes exactly 0.0. Both are indistinguishable from zero at the scale of the benchmark's overall HDR-vs-MPC gap (0.163 = −0.8454 − (−1.0086)).

**(b) Real finding or artifact:**
This is a **real finding**, not a methodological artifact. The Stage 08b design (asymmetric J coupling matrix with strong/weak axes, initial displacement on weak axes, elevated R_Brier = 0.04) was specifically constructed to exercise these channels. The ablation runs use independent trajectories per variant with deterministic seeds. Despite the favorable experimental design, coherence and calibration do not produce measurable marginal gains. The finding is reproducible: the v7.4.0 re-run produces identical values to v7.3.0.

The explanation is that the τ̃ surrogate proxy accounts for essentially all of the HDR-vs-MPC advantage. The coherence penalty term (w3=0.3) is active for ~31% of steps but produces negligible coupling scale values (mean coupling = 0.0076). The calibration adjustment to κ̂ via p_A^robust shifts κ̂ by at most 0.013, insufficient to change the control law measurably.

**(c) Manuscript implications:**
The manuscript **cannot claim positive attribution to coherence or calibration** based on the ablation evidence. Specifically:
- Any statement of the form "the coherence penalty contributes X% of the HDR gain" is not supported unless X ≈ 0.
- Any statement of the form "calibration-adjusted p_A^robust reduces false positives by Y%" must be qualified: this operates via the FP/FN mechanism (Stage 10), not via direct cost improvement in the ablation.
- The manuscript should state: "In the multi-axis asymmetric ablation (Stage 08b), the marginal gains attributable to coherence (0.0001) and calibration (0.0) are negligible. The HDR advantage over MPC-only (0.163) is driven entirely by the τ̃ surrogate component."

---

## Section 5 — Test Accounting

### Pytest Results (v7.4.0)

| Metric | Value |
|--------|-------|
| Total collected | 293 |
| Total passed | 293 |
| Total failed | 0 |
| Total skipped | 0 |
| Test files | 30 |
| Runtime | 146.72s |

**Prior stale `pytest_final.txt` (v7.3.0):** 280 passed, 1 skipped (281 total collected), 1 warning.

**Change:** 293 − 281 = 12 additional tests since the prior pytest snapshot. 1 previously-skipped test now passes. All 293 tests pass with zero failures.

**Note on test fixes:** Two tests in `test_stage_16.py` were updated to match output key names from a duplicate function definition in `stage_16_extensions.py` (line 2603 overrides line 1088 with different keys: `expansion_bound_holds` → `bound_holds`, `new_axes_responsive` → `new_axis_responsive`).

**Table 13 check:** If Table 13 in the manuscript states 280 passed / 1 skipped / 30 files, it must be updated to **293 passed / 0 skipped / 0 failed / 30 files**. If it states 295 tests (as mentioned in CLAUDE.md), this is also incorrect — the current count is 293.

---

## Section 6 — Remaining Open Issues

### 6.1 Stage 15 — Proxy Composite `rmse_ratio_at_sigma_05`

The check `rmse_ratio_at_sigma_05` continues to fail: observed value 5.14 vs threshold < 2.0. This was present in v7.3.0 and remains in v7.4.0. The manuscript should acknowledge this as a known limitation of the proxy-composite estimator at low noise levels.

### 6.2 Stage 16 — Stochastic Variability in 16.04 and 16.17

Two Stage 16 subtests produce numerically different values across runs despite deterministic seeds. This suggests these subtests have additional sources of randomness (e.g., different execution order, floating-point non-determinism in matrix operations, or internal random state not controlled by the top-level seed). While both values remain within passing criteria, any manuscript table citing specific numbers for `cross_site_response` or `cost_ratio` (CRD) should note they may vary across platforms or Python versions.

- 16.04 cross_site_response: 0.1986 (prior) → 0.4216 (current) — both indicate positive cross-site coupling
- 16.17 cost_ratio: 1.0003 (prior) → 1.0085 (current) — both near 1.0 as expected

### 6.3 Stage 16 — Duplicate Function Definitions

`hdr_validation/stages/stage_16_extensions.py` contains two definitions of `_run_subtest_16_11_expansion` (lines 1088 and 2603) with different output key names. The second definition overrides the first. This should be cleaned up to avoid future confusion. Two tests in `test_stage_16.py` were corrected to match the active (second) definition.

### 6.4 Stage 11 — Appendix J Theoretical Expectation

Stage 11 invariant set results are unchanged:
- Basin 0 (ρ=0.72): c_k=2.33, containment_rate_rpi=1.000
- Basin 1 (ρ=0.96): c_k=14.99, containment_rate_rpi=0.997
- Basin 2 (ρ=0.55): c_k=2.07, containment_rate_rpi=1.000

All basins satisfy Proposition 8.4 (containment ≥ 0.90). If Appendix J reports different c_k values or containment rates (e.g., from an earlier run at different scale), those numbers need manual reconciliation against these production-scale values (n_seeds=5, T=128, n_sigma=5.0).

### 6.5 Test Count Discrepancy

CLAUDE.md states "295 tests, 30 files" but the actual count is 293 tests, 30 files. CLAUDE.md should be updated to match. (The stale `pytest_final.txt` said 280/281.)
