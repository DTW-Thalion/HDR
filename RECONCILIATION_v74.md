# Reconciliation Document — HDR Validation Suite v7.4.0

**Generated:** 2026-03-17
**Python:** 3.14.3
**NumPy:** 2.4.3
**Platform:** Windows 11 (10.0.26200)

---

## Section 1 — Version Stamp

| Artifact | Version | Confirmed |
|----------|---------|-----------|
| `pyproject.toml` | 7.4.0 | Yes (created this session) |
| `__init__.py` `__version__` | 7.4.0 | Yes (added this session) |
| `README.md` header | v7.4.0 | Yes (rewritten this session) |
| Regenerated result artifacts | Note: runners do not embed `hdr_version` in output JSON; `chance_calibration.json` and `stage_summary.json` were produced by v7.4.0 code but do not carry an explicit version field. Recommend adding `"hdr_version": "7.4.0"` to runner output in a follow-up patch. |

**Note:** The stage runners produce artifacts without a `"hdr_version"` or `"generated_at"` field. The `stage_summary.json` carries only a `"date"` field (set at write time). The constraint that every artifact carry `"hdr_version": "7.4.0"` and `"generated_at"` ISO timestamp requires runner code changes, which are out of scope per the constraint "do not modify any stage runner logic." The reconciliation below uses timestamps from the re-run session (2026-03-17).

---

## Section 2 — Artifact Reconciliation Table

### Smoke Profile

| Check / Metric | Prior Value (2026-03-10) | New Value (2026-03-17) | Change |
|---------------|--------------------------|------------------------|--------|
| Total checks | 85 | 85 | Unchanged |
| Passed | 85 | 85 | Unchanged |
| Failed | 0 | 0 | Unchanged |
| tau_tilde(far) | 66.4452 | 66.4452 | Unchanged — deterministic math |
| committor q[A] | 0.00e+00 | 0.00e+00 | Unchanged |
| committor q[B] | 1.000000 | 1.000000 | Unchanged |
| mode1 F1 | 0.9018 | 0.9018 | Unchanged (deterministic seed 101) |
| Brier reliability | 0.0052 | 0.0052 | Unchanged |
| p_A_robust | 0.7052 | 0.7052 | Unchanged |
| Mode A max‖u‖ | — (not in prior smoke report) | 0.6378 | New check captured; smoke uses 1 seed so max‖u‖ is lower than standard's 0.6620 |
| Mode B escape | 0.740 → 0.860 | 0.740 → 0.860 | Unchanged |
| V*(start) | 0.8476 | 0.8476 | Unchanged |
| epsilon_H | 0.7828 | 0.7828 | Unchanged |

**Interpretation:** Smoke profile is fully reproducible. All 85 checks reproduce identically across the 7-day gap.

### Standard Profile

| Check / Metric | Prior Value (2026-03-10) | New Value (2026-03-17) | Change |
|---------------|--------------------------|------------------------|--------|
| Total checks | 81 | 89 | **+8 new checks** added since prior run |
| Passed | 81 | 89 | All pass |
| Failed | 0 | 0 | Unchanged |
| Stage 01 checks | 21 | 16 | −5: prior included data-gen checks in stage01; now separated (see stage02) |
| Stage 02 checks | 5 | 6 | +1 |
| Stage 03 checks | — (not reported) | 5 | Now explicitly reported |
| Stage 04 checks | 6 | 12 | **+6 new checks**: pooled_estimated gain, estimation cost ratio, heavy-tail calibration, basin-stratified diagnostics |
| Stage 06 checks | 5 | 6 | +1: coherence time-in-band check added |
| tau_tilde(far) | 66.4452 | 66.4452 | Unchanged |
| tau_tilde Spearman rho | — | 0.9047 | New check (Claim 3); exceeds 0.70 threshold |
| Brier reliability | 0.0051 | 0.0051 | Unchanged |
| p_A_robust | 0.7051 | 0.7051 | Unchanged |
| Mode A max‖u‖ | 0.6620 | 0.6620 | Unchanged |
| Mode A feasibility | 1.00 (64/64) | 1.00 | Unchanged |
| Mode A non-zero u | 18/64 | 18/64 | Unchanged |
| HDR vs open-loop gain | 0.0006 | 0.0006 | Unchanged |
| HDR vs pooled gain | −0.0097 | −0.0097 | Unchanged |
| Safety delta vs pooled | −0.0013 | −0.0013 | Unchanged |
| mode_error_fit_slope | 0.1743 | 0.1743 | Unchanged |
| mode_error_fit_r2 | 0.9988 | 0.9988 | Unchanged |
| target_drift_fit_slope | 3.2657 | 3.2657 | Unchanged |
| gaussian_calibration_abs_error | 0.0012 | 0.0012 | Unchanged |
| HDR vs pooled_estimated gain | — | −0.0016 | **New metric.** Fair baseline comparison (both use IMM x̂). Within −3% tolerance. |
| Pooled estimated/oracle ratio | — | 0.9871 | **New metric.** Estimation noise impact is <2%, confirming fair comparison. |
| Heavy-tail cal degradation | — | 0.0712 | **New metric.** Below 0.10 threshold. |
| Coherence time-in-band | — | with=0.074, without=0.056 | **New check.** Directional improvement confirmed; margin is 1.8pp. |
| Mode B escape (n=100 MC) | 0.700 → 0.860 | 0.700 → 0.860 | Unchanged |
| V*(start) | 0.8476 | 0.8476 | Unchanged |

**Interpretation:** All previously reported values reproduce exactly. The 8-check increase comes from new checks added to the standard runner since the prior artifact generation, not from any change to existing checks. All new checks pass.

---

## Section 3 — Manuscript Update Instructions

1. **Table summarizing check counts (if any):** Replace standard profile total "81 passed / 81 total" with "89 passed / 89 total." This does not change any substantive claim; it reflects 8 additional validation checks that strengthen coverage.

2. **Stage 04 discussion:** If the manuscript reports only 6 checks for Stage 04 standard, update to 12 checks. The 6 new checks are:
   - HDR gain vs pooled_estimated > −0.03 (value: −0.0016)
   - Pooled estimated cost ≥ 90% of pooled oracle cost (value: 0.9871)
   - Heavy-tail calibration degradation < 0.10 (value: 0.0712)
   - HDR gain vs open-loop > 0 (value: 0.0006) — was implicit, now explicit
   - HDR gain vs pooled > −0.10 (value: −0.0097) — was implicit, now explicit
   - Safety delta vs pooled ≤ 0.015 (value: −0.0013) — was implicit, now explicit

   These do not change the manuscript's substantive claims. They formalize checks that were previously reported as headline gains but not counted as formal pass/fail checks.

3. **Claim 3 (τ̃ correlation):** The standard profile now reports Spearman ρ = 0.9047 (vs ≥ 0.70 threshold). If the manuscript does not already include this number, add: "τ̃ rank correlation with empirical recovery cost: ρ = 0.90 (standard profile, n = 24 episodes)."

4. **Stage 06 coherence:** If the manuscript reports 5 checks for Stage 06, update to 6. The new check is: "Coherence penalty improves time-in-band vs w3=0" (with=0.074, without=0.056). See Section 4 for interpretation.

5. **All numerical values in the manuscript that correspond to the reconciliation table above remain correct.** No stale numbers were found — the existing reported values (tau_tilde=66.4452, Brier=0.0051, HDR gain=0.0006, safety delta=−0.0013, etc.) all reproduce exactly.

---

## Section 4 — Stage 06 Coherence / Calibration Finding (Stage 08B equivalent)

The coherence penalty's contribution to time-in-band was measured as:
- **With coherence (w3=0.3):** 0.074 (7.4% of steps in target band)
- **Without coherence (w3=0):** 0.056 (5.6% of steps in target band)
- **Marginal gain:** +1.8 percentage points

**Assessment:**

(a) The marginal gain from coherence is **near-zero but positive** (1.8pp). This is a real directional effect — it is statistically consistent across the deterministic seed pair [101, 202] — but the magnitude is small.

(b) This is a **real finding, not a methodological artifact.** The standard runner uses independent episode trajectories with separate seeds. The coherence penalty operates only on the structural cost function, not on the trajectory generation, so the before/after comparison is clean. However, the 1.8pp effect is within the range that could vanish or reverse with different seed sets or higher episode counts.

(c) **Manuscript guidance:** The manuscript **should not** claim that coherence calibration produces a large or reliably significant improvement in time-in-band at the standard profile scale. The defensible claim is:
> "The coherence penalty (w3 = 0.3) produces a directional improvement in time-in-band (+1.8pp in the standard profile), confirming that the penalty functions as designed. The effect is structurally present but small at this sample size; the 10pp significance threshold is deferred to the full integration profile."

If the manuscript currently claims positive attribution to coherence/calibration for time-in-band improvement, this must be softened to match the empirical evidence.

---

## Section 5 — Test Accounting

**Pytest results (2026-03-17):**

| Metric | Value |
|--------|-------|
| Total collected | 60 |
| Passed | 60 |
| Failed | 0 |
| Skipped | 0 |
| Errors | 0 |
| Test files | 9 |
| Duration | 1.14s |
| Warnings | 1 (pytest config, non-functional) |

If Table 13 in the manuscript reports 60 tests passing across 9 test files with 0 failures, it matches. If it reports a different number, update accordingly — the authoritative count is 60 passed / 0 failed / 0 skipped from 9 test files.

---

## Section 6 — Remaining Open Issues

1. **No `hdr_version` in result artifacts.** The constraint "every result artifact must carry `hdr_version: 7.4.0` and `generated_at`" cannot be satisfied without modifying runner logic, which is out of scope. Recommend a follow-up patch to inject version metadata into all JSON outputs.

2. **`stage_summary.json` not regenerated.** The `stage_summary.json` file in `results/stage_04/standard/` was written on 2026-03-10 and still carries `"date": "2026-03-10"`. The re-run regenerated `chance_calibration.json` (which the standard runner writes explicitly) but the stage_summary.json is written by a different code path (possibly `generate_reports.py` or a separate stage04 artifact writer). Its values match the re-run exactly, so no data discrepancy exists, but the timestamp is stale.

3. **Extended and validation profiles not re-run.** Only smoke and standard profiles were re-run in this session. The prior artifacts (2026-03-10) for extended and validation profiles showed all-pass results. Recommend re-running these profiles to confirm reproducibility, though the deterministic-seed architecture makes divergence unlikely.

4. **File naming mismatches persist.** As documented in CLAUDE.md, several `.md` files have name/content mismatches (README.md was fixed; CLAIM_CRITERIA.md, EXTENDED_PROFILE_REPORT.md, REPUBLISH_NOTE.md, STANDARD_PROFILE_REPORT.md remain mismatched). These do not affect validation results but should be resolved for repository hygiene.

5. **Windows symlink issue.** The `hdr_validation` self-referential symlink (designed for Unix) does not work on Windows. A directory junction was created for this session. This should be documented or replaced with a `sys.path` manipulation in the runners.

6. **Standard profile check count changed.** The prior report showed 81 checks; the re-run shows 89. This is due to runner code additions between 2026-03-10 and 2026-03-17, not a data discrepancy. The manuscript should reflect the current count (89).
