# Standard profile report

## Completion

| stage | status | finished_at |
| --- | --- | --- |
| stage_00 | completed | 2026-03-08T21:02:31Z |
| stage_01 | completed | 2026-03-08T21:02:56Z |
| stage_02 | completed | 2026-03-08T21:02:59Z |
| stage_03 | completed | 2026-03-08T21:03:04Z |
| stage_04 | completed | 2026-03-08T21:18:03Z |
| stage_05 | completed | 2026-03-08T21:19:05Z |
| stage_06 | completed | 2026-03-08T21:21:07Z |
| stage_07 | completed | 2026-03-08T21:22:48Z |

## Key results

- Observer state RMSE: `0.27879168722386777`
- Observer mode F1: `0.47477125753779664`
- Axes below 0.9 RMSE: `8`
- HDR nominal gain vs open-loop: `-8.409669068259522`
- HDR nominal gain vs pooled LQR: `-4.7881339639446185`
- Gaussian calibration absolute error: `0.087109375`
- Mode-error slope / R²: `1.0384679277751484e-15` / `1.0`
- Target-drift slope / R²: `0.013866237561498393` / `0.9362647582680822`
- Mode B hybrid escape gain: `0.0`
- Mode B false-positive rate: `0.8125`
- Reduced-chain absolute gap: `0.10052230782514282`
- Coherence effect label: `neutral`
- Integrated time-in-band gain: `-0.018798828125`
- Robustness best / worst gain: `0.9948967598328289` / `-6.403292003795752`
- Oracle optimism gap: `0.9965698524453173`


## Interpretation

- This profile is **in silico only**.
- The generated evidence is intentionally non-oracle and includes model mismatch, missingness, delays, and robustness sweeps.
- Negative or brittle findings are preserved rather than filtered out.

