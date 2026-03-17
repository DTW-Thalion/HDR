# HDR Validation Suite — Results Summary

Generated: 2026-03-11 21:39 UTC

## Profile Overview

| Profile | Seeds | Episodes | Steps | MC | Total | Passed | Failed | Status | Time |
|---------|-------|----------|-------|----|-------|--------|--------|--------|------|
| smoke | 1 | 8 | 128 | 50 | 97 | 96 | 1 | ✗ 1 FAIL | 6.4s |
| standard | 2 | 12 | 128 | 100 | 98 | 97 | 1 | ✗ 1 FAIL | 24.8s |
| extended | 3 | 20 | 256 | 150 | 110 | 109 | 1 | ✗ 1 FAIL | 114.0s |
| validation | 3 | 12 | 128 | 150 | 95 | 94 | 1 | ✗ 1 FAIL | 6.9s |

## Stage-by-Stage Results

| Stage | Smoke | Standard | Extended | Validation |
|-------|--------|--------|--------|--------|
| stage01 | ✓ 19/19 | ✓ 25/25 | ✓ 29/29 | ✓ 28/28 |
| stage02 | ✓ 12/12 | ✓ 5/5 | ✓ 6/6 | ✓ 6/6 |
| stage03 | ✓ 8/8 | — | — | — |
| stage03b | ✓ 9/9 | ✓ 11/11 | ✓ 11/11 | ✓ 11/11 |
| stage03c | ✓ 8/8 | ✓ 8/8 | ✓ 9/9 | ✓ 8/8 |
| stage04 | ✓ 8/8 | ✓ 14/14 | ✓ 16/16 | ✓ 10/10 |
| stage05 | ✓ 9/9 | ✓ 9/9 | ✓ 10/10 | ✓ 9/9 |
| stage06 | ✓ 8/8 | ✓ 8/8 | ✓ 9/9 | ✓ 5/5 |
| stage07 | ✗ 15/16 | ✗ 17/18 | ✗ 19/20 | ✗ 17/18 |

## Key Metrics by Profile

| Metric | Smoke | Standard | Extended | Validation |
|--------|--------|--------|--------|--------|
| tau_tilde(far) > 0 | 66.4452 | 66.4452 | 66.4452 | 66.4452 |
| committor q[A]=0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| committor q[B]=1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| DARE P positive-definite | ✓ | ✓ | ✓ | ✓ |
| mode1 F1 > 0 | 0.9018 | — | — | — |
| Brier reliability finite | 0.0052 | 0.0051 | 0.0019 | 0.0246 |
| p_A_robust ∈ [0,1] | 0.7052 | 0.7051 | 0.7019 | 0.7246 |
| Mode A feasibility rate > 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mode B aggressive > passive | 0.740 → 0.860 | 0.700 → 0.860 | 0.667 → 0.860 | 0.667 → 0.860 |
| Exact DP V* ∈ [0,1] | V*(start)=0.8476 | V*(start)=0.8476 | V*(start)=0.8476 | V*(start)=0.8476 |
| epsilon_H > 0 | 0.7828 | 0.7828 | 0.7828 | 0.7828 |

## Failed Checks

- **[smoke]** `stage07` — Mismatch bound covers p90 basin-1 delta_A (theory guarantee validity): `VIOLATED: 0.200 vs p90=0.347  (FAIL here means ISS Proposition 10.4 guarantee invalid in ~10% of seeds — disclose in manuscript)`
- **[standard]** `stage07` — Mismatch bound covers p90 basin-1 delta_A (theory guarantee validity): `VIOLATED: 0.200 vs p90=0.347  (FAIL here means ISS Proposition 10.4 guarantee invalid in ~10% of seeds — disclose in manuscript)`
- **[extended]** `stage07` — Mismatch bound covers p90 basin-1 delta_A (theory guarantee validity): `VIOLATED: 0.200 vs p90=0.347  (FAIL here means ISS Proposition 10.4 guarantee invalid in ~10% of seeds — disclose in manuscript)`
- **[validation]** `stage07` — Mismatch bound covers p90 basin-1 delta_A (theory guarantee validity): `VIOLATED: 0.200 vs p90=0.347  (FAIL here means ISS Proposition 10.4 guarantee invalid in ~10% of seeds — disclose in manuscript)`
