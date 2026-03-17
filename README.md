# HDR Validation Suite v7.4.0

In-silico validation suite for the **Homeodynamic Remediation Framework (HDR) v5.0** — a multi-mode adaptive control system for constrained stochastic linear dynamical systems (SLDS). The suite validates mathematical properties, implementation correctness, and empirical performance across synthetic physiological scenarios using three control strategies: nominal LQR/MPC (Mode A), structured exploration (Mode B), and information-maximizing identification (Mode C).

## Repository Structure

```
├── control/                  # Control policy subpackage (LQR, MPC, Mode B, Mode C)
├── inference/                # Inference subpackage (Kalman, IMM, ICI)
├── model/                    # System model subpackage (SLDS, HSMM, target sets, safety)
├── results/                  # Per-stage result artifacts (auto-generated)
├── reports/                  # Aggregated profile reports (auto-generated)
├── smoke_runner.py           # Smoke profile runner (fastest)
├── standard_runner.py        # Standard profile runner
├── extended_runner.py        # Extended profile runner
├── extended_512_runner.py    # Extended-512 profile runner
├── validation_runner.py      # Validation profile runner
├── run_all.py                # Orchestration script (all profiles, all stages)
├── generate_reports.py       # Report generation across profiles
├── config.json               # Master configuration defaults
├── smoke.json                # Smoke profile overrides
├── standard.json             # Standard profile overrides
├── extended_512.json         # Extended-512 profile overrides
├── validation.json           # Validation profile overrides
├── test_*.py                 # Pytest test modules (9 files)
└── pyproject.toml            # Package metadata and version
```

## Running the Validation Suite

Requires Python 3.10+ with `numpy`, `scipy`, `pandas`, and `pytest`.

```bash
pip install numpy scipy pandas pytest
```

### Individual profiles

```bash
python smoke_runner.py          # ~3s, 1 seed, 8 episodes
python standard_runner.py       # ~4s, 2 seeds, 12 episodes
python extended_runner.py       # ~16s, 3 seeds, 20 episodes
python validation_runner.py     # ~5s, 3 seeds, 12 episodes
```

### Orchestrated runs

```bash
python run_all.py                                    # All profiles, all stages
python run_all.py --profiles smoke                   # Smoke only
python run_all.py --profiles smoke standard --stages 01 04
python run_all.py --resume --skip-done               # Resume, skip completed
```

### Stage IDs

| ID   | Description                          |
|------|--------------------------------------|
| 01   | Mathematical validation              |
| 02   | Synthetic dataset generation         |
| 03   | IMM inference                        |
| 03b  | ICI calibration and diagnostics      |
| 03c  | Mode C validation                    |
| 04   | Mode A control performance           |
| 05   | Mode B structured exploration        |
| 06   | State coherence                      |
| 07   | Robustness sweeps                    |

## Regenerating Result Artifacts

```bash
# Regenerate all reports (writes to reports/)
python generate_reports.py

# Regenerate specific profiles
python generate_reports.py --profiles smoke standard

# Force re-run all stages for a profile
python run_all.py --profiles standard --force
```

Result artifacts are written to `results/stage_{id}/{profile}/` and include `config.json`, `summary.json`, `metrics.csv`, and optional plots.

## Running Tests

```bash
pytest                    # All tests
pytest test_mpc.py -v     # Specific test file
pytest -q                 # Quiet summary
```

Nine test files cover inference (ICI, IMM), control (MPC, Mode C, committor), model (HSMM, target set, recovery), and packaging.

## Configuration

Configs compose from `config.json` (master defaults) plus a profile override file. Key parameters include 8-dimensional state/control, K=3 basins with spectral radii [0.72, 0.96, 0.55], and ICI thresholds for Mode C activation.

## Claim Validation

The suite validates 14 claims (Claims 1-14) covering ICI correctness, control improvement, recovery cost bounds, calibration, stability, and identifiability. A claim is marked `Supported` only when it passes in both smoke and standard profiles. See `CLAIM_MATRIX.md` for current status.
