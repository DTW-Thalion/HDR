# Reproducibility

## Determinism

- All stages use explicit integer seeds.
- The run manifest records seeds and config hashes.
- Completed stages are skipped automatically when the config hash is unchanged and `--skip-done` is enabled.

## Resumability

- Every stage writes artifacts atomically.
- Partial results are flushed after every seed and chunk.
- Per-stage zips are updated immediately after stage completion or failure.
- Failures are logged to `results/logs/` and summarized in `docs/FAILURES.md`.

## Re-running

Standard end-to-end run:

```bash
python run_all.py --resume --skip-done
```

Selective re-run:

```bash
python run_all.py --profiles smoke --stages 03 04 --force
```

## No internet / no external downloads

This repository uses only synthetic data generated locally. It performs no network access and no external dataset downloads.
