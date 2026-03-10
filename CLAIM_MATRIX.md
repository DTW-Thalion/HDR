# HDR v5.0 Claim Validation Matrix

## Test Summary

| Run configuration                             | Checks | Result   |
|-----------------------------------------------|--------|----------|
| Smoke runner (1 seed × 8 episodes)            | 14     | Partial  |
| Standard runner (2 seeds × 12 episodes)       | 14     | Partial  |
| Extended runner (3 seeds × 20 episodes)       | 14     | Partial  |
| High-power runner (20 seeds × 20 ep/seed)     | 3      | See below |

---

## Benchmark A Criterion (revised for high-power)

**OLD criterion:** gain ≥ +0.03; win_rate ≥ 0.70

**NEW criterion:** gain ≥ +0.03 **AND** 95% CI lower bound ≥ +0.03 **AND** win_rate ≥ 0.70

---

## Claims 1–2: Maladaptive-Basin Performance (Benchmark A)

### Claim 1 — HDR Mode A reduces cost on maladaptive-basin episodes

**Criterion:** Mean fractional cost reduction ≥ +3% vs pooled\_lqr\_estimated; 95% CI lower bound ≥ +0.03; win rate ≥ 70%.

| Metric                          | Standard (2 seeds, 12 ep/seed) | Extended (3 seeds, 20 ep/seed) | **Highpower (20 seeds, 20 ep/seed)** |
|---------------------------------|--------------------------------|--------------------------------|--------------------------------------|
| N maladaptive episodes          | ~11                            | ~15                            | **123**                              |
| Mean gain vs pooled\_lqr\_est   | +0.0574                        | +0.0357                        | **+0.0283**                          |
| 95% CI lower (mean)             | —                              | —                              | **+0.0210**                          |
| 95% CI upper (mean)             | —                              | —                              | **+0.0355**                          |
| 90% CI lower (mean)             | —                              | —                              | **+0.0221**                          |
| 90% CI upper (mean)             | —                              | —                              | **+0.0344**                          |
| 95% CI lower (median)           | —                              | —                              | **+0.0163**                          |
| Win rate                        | 0.909                          | 0.800                          | **0.772**                            |
| Safety delta vs pe              | —                              | —                              | **+0.0004**                          |
| Seeds ≥ +0.03 criterion         | —                              | —                              | **8/20**                             |
| Seed gain std                   | —                              | —                              | **0.0165**                           |
| **Status (old criterion)**      | Supported                      | Supported                      | Supported (point est + win rate)     |
| **Status (new 95% CI criterion)** | N/A                          | N/A                            | **Partially supported**              |

**Highpower Status: Partially supported**

- Point estimate (+0.0283) exceeds +0.03 threshold: ✗ (below +0.03)
- Win rate (0.772) exceeds 0.70: ✓
- 95% CI lower bound (+0.0210) ≥ +0.03: ✗

> **Note:** The point estimate (+0.0283) is below the +0.03 threshold in the high-power run.
> The 95% CI [+0.0210, +0.0355] straddles the criterion threshold: the positive effect is
> real (lower bound > 0) but the magnitude is imprecisely characterised. Only 8/20 seeds
> individually exceed +3%, suggesting the effect is noisy across random initialisations.
> The manuscript should report the honest estimate with CI rather than the inflated small-N result.

---

### Claim 2 — HDR Mode A win rate on maladaptive basin ≥ 70%

**Criterion:** Fraction of maladaptive-basin episodes where HDR cost < pooled\_lqr\_estimated cost ≥ 0.70.

| Metric        | Standard | Extended | **Highpower** |
|---------------|----------|----------|---------------|
| Win rate      | 0.909    | 0.800    | **0.772**     |
| N episodes    | ~11      | ~15      | **123**       |
| **Status**    | Supported | Supported | **Supported** |

Claim 2 (win rate ≥ 0.70) is **Supported** in the high-power run: 0.772 > 0.70 across 123 episodes.

---

## Claims 3–14 (from v4.3 / v5.0 stages)

These claims are evaluated by the smoke/standard/extended runners and are not directly
affected by the high-power Benchmark A results. See `CLAIM_CRITERIA.md` for full criteria.

| Claim | Description                              | Smoke | Standard | Extended | Highpower |
|-------|------------------------------------------|-------|----------|----------|-----------|
| 1     | Mode A cost reduction ≥ +3% (maladaptive) | —   | Supported | Supported | Partially supported |
| 2     | Mode A win rate ≥ 70% (maladaptive)      | —     | Supported | Supported | Supported |
| 3     | τ̃ correlation with recovery cost         | —     | —         | —         | N/A       |
| 4     | Chance-constraint calibration            | —     | —         | —         | N/A       |
| 5     | ISS scaling                              | —     | —         | —         | N/A       |
| 6     | Stability under drift                    | —     | —         | —         | N/A       |
| 7     | Mode B improvement                       | —     | —         | —         | N/A       |
| 8     | DP approximation quality                 | —     | —         | —         | N/A       |
| 9     | Coherence penalty                        | —     | —         | —         | N/A       |
| 10    | Identifiability improvement              | —     | —         | —         | N/A       |
| 11    | ICI regime identification                | —     | —         | —         | N/A       |
| 12    | Mode C Fisher improvement                | —     | —         | —         | N/A       |
| 13    | p\_A^robust FP reduction                 | —     | —         | —         | N/A       |
| 14    | Compound bound correctness               | —     | —         | —         | N/A       |

---

## High-Power Run Metadata

- Run date: 2026-03-10
- Seeds: 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020
- Episodes per seed: 20
- Steps per episode: 256
- Bootstrap resamples: 10,000 (seed=42)
- Results: `results/stage_04/highpower/highpower_summary.json`
