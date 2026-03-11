# VALIDATION_FAILURES.md

## Post-refactor failures (standard profile)

_Baseline run performed before Tasks 2–5 modifications._

| Stage | Check | Value |
|-------|-------|-------|
| — | (none) | All 95 checks passed |

**Decision**: No failures present. Standard profile: **95/95 passed**.

---

## Post-refactor failures (extended profile)

_Baseline run performed before Tasks 2–5 modifications._

| Stage | Check | Value |
|-------|-------|-------|
| — | (none) | All 107 checks passed |

**Decision**: No failures present. Extended profile: **107/107 passed**.

---

## Notes on Tasks 2–5 expected failures

The following checks introduced by Tasks 2–5 are **expected to fail** and are categorised as (B) ACKNOWLEDGE:

### 07.8 — Mismatch bound covers empirical p90 delta_A for basin 1
- **Category**: (B) ACKNOWLEDGE — known limitation documented in manuscript
- **Reason**: `model_mismatch_bound = 0.20` is less than the empirical p90 of basin-1 delta_A (~0.347). This means the ISS Proposition 10.4 guarantee is violated in ~10% of seeds. This is an intentional test of the current parameter values and must remain visible. The manuscript should disclose this.
- **Before/After**: no threshold change (must fail with current params).

### 07.4b — IMM posterior entropy after 50 non-maladaptive steps
- **Category**: (B) ACKNOWLEDGE if H ≤ 0.3 — filter may over-concentrate regardless of signal. Known limitation.
- **Reason**: The IMM filter with hard-regime tuning may lock to a mode after repeated non-maladaptive observations, which would reveal that the filter doesn't retain meaningful uncertainty.

### 07.4c — IMM recovers basin-1 MAP mode within 30 steps from wrong prior
- **Category**: (B) ACKNOWLEDGE if MAP mode ≠ 1 — filter with hard-regime tuning may have slow recovery from wrong prior. Known limitation.

---

## Final check counts after Tasks 1–6

All four runners verified via `generate_reports.py --profiles smoke standard extended validation` on 2026-03-11.

| Profile | Total checks | Passed | Failed | Failure(s) |
|---------|-------------|--------|--------|------------|
| smoke | 97 | 96 | 1 | 07.8 mismatch bound (expected, (B) ACKNOWLEDGE) |
| standard | 98 | 97 | 1 | 07.8 mismatch bound (expected, (B) ACKNOWLEDGE) |
| extended | 110 | 109 | 1 | 07.8 mismatch bound (expected, (B) ACKNOWLEDGE) |
| validation | 95 | 94 | 1 | 07.8 mismatch bound (expected, (B) ACKNOWLEDGE) |
| **Total** | **400** | **396** | **4** | All 4 are the same intentional 07.8 failure |

New checks introduced by Tasks 2–5 (all four runners):

| Check | Result across all profiles |
|-------|---------------------------|
| 04.3b — Mode A active fraction ≥ 0.75 from far states | PASS (active=0.90) |
| 04.13 — Adaptive-basin N documented | PASS (always-true documentation check) |
| 07.4a — IMM numerical stability at p_miss=0.3 | PASS |
| 07.4b — IMM posterior entropy > 0.3 nats after 50 steps | PASS (H≈0.68) |
| 07.4c — IMM recovers basin-1 MAP within 30 steps | PASS (map_mode=1) |
| 07.8 — Mismatch bound covers p90 basin-1 delta_A | **FAIL** (0.200 < 0.347 — intentional, (B) ACKNOWLEDGE) |

---

## Tasks completed

- **Task 1** — COMPLETE. Baseline runs of all four runners captured before modifications. No pre-existing failures found (standard: 95/95, extended: 107/107, smoke: 87/87, validation: 86/86). Documented in this file.
- **Task 2** — COMPLETE. Replaced trivial missing-data sweep (07.4) in all four runners with three meaningful inference-quality checks: 07.4a (numerical stability), 07.4b (posterior entropy > 0.3 nats), 07.4c (MAP recovery from wrong prior). All three pass in all profiles.
- **Task 3** — COMPLETE. Added check 04.3b in all four runners immediately after "04.3 Non-trivial control". Samples 20 far states (‖x‖ ∈ [1.5, 3.0]) from basin 1 and verifies active_fraction ≥ 0.75. Passes in all profiles (active=0.90).
- **Task 4** — COMPLETE. Added check 07.8 at the end of stage07_robustness in all four runners. Loads `results/stage_04/highpower/mismatch_audit.json` and verifies `model_mismatch_bound ≥ basin_1_p90`. Intentionally FAILS in all profiles (0.200 < 0.347). This reveals that ISS Proposition 10.4 is violated in ~10% of seeds and **must be disclosed in the manuscript**. No fix applied; failure preserved.
- **Task 5** — COMPLETE. Added check 04.13 at the end of stage04_mode_a in all four runners. Counts episodes not in basin 1 (`n_adaptive`) and records the value. Always passes. Emits an UNDERPOWERED note if n_adaptive < 20.
- **Task 6** — COMPLETE. Ran `generate_reports.py --profiles smoke standard extended validation`. Completed without exception. `reports/summary.md` reflects correct check counts. The mismatch-bound check (07.8) appears as FAIL in all profiles with full disclosure note. The adaptive-basin documentation check (04.13) appears as PASS in all profiles. Final state: **396/400 checks pass** across all profiles; 4 intentional failures all correspond to the known ISS Proposition 10.4 limitation.
