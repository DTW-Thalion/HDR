# HDR Validation Suite — Results Summary

Generated: 2026-03-09 23:06 UTC

## Profile Overview

| Profile | Seeds | Episodes | Steps | MC | Total | Passed | Failed | Status | Time |
|---------|-------|----------|-------|----|-------|--------|--------|--------|------|
| smoke | 1 | 8 | 128 | 50 | 85 | 85 | 0 | ✓ PASS | 2.7s |
| standard | 2 | 12 | 128 | 100 | 81 | 81 | 0 | ✓ PASS | 3.9s |
| extended | 3 | 20 | 256 | 150 | 93 | 93 | 0 | ✓ PASS | 13.3s |
| validation | 3 | 12 | 128 | 150 | 88 | 88 | 0 | ✓ PASS | 4.7s |

## Stage-by-Stage Results

| Stage | Smoke | Standard | Extended | Validation |
|-------|--------|--------|--------|--------|
| stage01 | ✓ 15/15 | ✓ 21/21 | ✓ 25/25 | ✓ 24/24 |
| stage02 | ✓ 12/12 | ✓ 5/5 | ✓ 6/6 | ✓ 6/6 |
| stage03 | ✓ 8/8 | — | — | — |
| stage03b | ✓ 9/9 | ✓ 11/11 | ✓ 11/11 | ✓ 11/11 |
| stage03c | ✓ 8/8 | ✓ 8/8 | ✓ 9/9 | ✓ 8/8 |
| stage04 | ✓ 5/5 | ✓ 6/6 | ✓ 8/8 | ✓ 8/8 |
| stage05 | ✓ 9/9 | ✓ 9/9 | ✓ 10/10 | ✓ 9/9 |
| stage06 | ✓ 5/5 | ✓ 5/5 | ✓ 6/6 | ✓ 5/5 |
| stage07 | ✓ 14/14 | ✓ 16/16 | ✓ 18/18 | ✓ 17/17 |

## Key Metrics by Profile

| Metric | Smoke | Standard | Extended | Validation |
|--------|--------|--------|--------|--------|
| tau_tilde(far) > 0 | 66.4452 | 66.4452 | 66.4452 | 66.4452 |
| committor q[A]=0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| committor q[B]=1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| DARE P positive-definite | ✓ | ✓ | ✓ | ✓ |
| mode1 F1 > 0 | 0.9018 | — | — | — |
| Brier reliability finite | 0.0052 | 0.0035 | 0.0010 | 0.0246 |
| p_A_robust ∈ [0,1] | 0.7052 | 0.7035 | 0.7010 | 0.7246 |
| Mode A feasibility rate > 0.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mode B aggressive > passive | 0.740 → 0.860 | 0.700 → 0.860 | 0.667 → 0.860 | 0.667 → 0.860 |
| Exact DP V* ∈ [0,1] | V*(start)=0.8476 | V*(start)=0.8476 | V*(start)=0.8476 | V*(start)=0.8476 |
| epsilon_H > 0 | 0.7828 | 0.7828 | 0.7828 | 0.7828 |

## Failed Checks

_None — all checks passed across all profiles._
