# CLAUDE.md — HDR Validation Suite

## Project Overview

This repository is an **in-silico validation suite** for the **Homeodynamic Remediation Framework (HDR) v7.0** — a multi-mode adaptive control system for constrained stochastic linear dynamical systems (SLDS). The framework validates mathematical properties, implementation correctness, and empirical performance across synthetic physiological scenarios.

HDR models a latent physiological state (e.g., neuroendocrine system) with K discrete operating modes ("basins") and switches between three control strategies based on inference quality:

- **Mode A**: Nominal control (LQR/MPC with Riccati recursion)
- **Mode B**: Structured exploration when inference quality is adequate
- **Mode C**: Information-maximizing identification (dither injection) when inference quality is insufficient

---

## Repository Structure

```
/home/user/HDR/
├── hdr_validation/              # Python package (control, inference, model, identification)
│   ├── control/                 # Control policy subpackage
│   │   ├── lqr.py               # DLQR, committor calculations, value iteration, committor_with_jumps
│   │   ├── mode_b.py            # Structured exploration control (Mode B)
│   │   ├── mode_c.py            # Information-maximizing dither control (Mode C)
│   │   ├── mimpc.py             # Mixed-Integer MPC (v7.0)
│   │   ├── supervisor.py        # Extended 8-branch supervisor (v7.0)
│   │   └── mpc.py               # Model Predictive Control (Mode A), solve_mode_a_unstable, solve_mode_a_irr
│   ├── inference/               # Inference/estimation subpackage
│   │   ├── ici.py               # Inference-Control Interface (ICI) — core ICI conditions
│   │   ├── imm.py               # IMM filter + RegionConditionedIMM, FactoredMultiSiteIMM, MultiRateIMM (v7.0)
│   │   ├── kalman.py            # Kalman filtering
│   │   ├── particle.py          # Particle filter / SMC (v7.0)
│   │   ├── variational.py       # Variational SLDS inference (v7.0)
│   │   └── population.py        # Population-prior basin assignment (v7.0)
│   ├── model/                   # System model subpackage
│   │   ├── slds.py              # SLDS model factory (BasinModel, EvaluationModel)
│   │   ├── hsmm.py              # Hidden Semi-Markov dwell models
│   │   ├── coherence.py         # State coherence metrics
│   │   ├── recovery.py          # Recovery trajectory analysis
│   │   ├── safety.py            # Safety analysis tools
│   │   ├── target_set.py        # Target set geometry (box/ellipsoidal)
│   │   ├── extensions.py        # v7.0 structural extensions (basin classifier, PWA, jump-diffusion, etc.)
│   │   ├── adaptive.py          # Forgetting-factor RLS, drift detection (v7.0)
│   │   └── multirate.py         # Multi-rate observer, delay augmentation (v7.0)
│   ├── identification/          # v7.0 identification subpackage
│   │   ├── hierarchical.py      # Hierarchical empirical-Bayes coupling estimation
│   │   ├── boed.py              # Bayesian Optimal Experimental Design
│   │   ├── committor_recovery.py # Committor recovery via kernel regression
│   │   ├── transition_rates.py  # Transition rate estimation
│   │   ├── population_planning.py # Population-prior treatment planning
│   │   ├── tau_estimation.py    # Tau burden estimation
│   │   └── risk_information.py  # Risk-information frontier
│   └── stages/                  # Stage scripts for stages 08–15
│       ├── stage_08_ablation.py
│       ├── stage_09_baselines.py
│       ├── stage_10_mode_b_sweep.py
│       ├── stage_11_invariant_set.py
│       ├── stage_12_hierarchical.py  # v7.0
│       ├── stage_13_inference_backbone.py  # v7.0
│       ├── stage_14_population_planning.py  # v7.0
│       └── stage_15_proxy_composite.py  # v7.0
├── results/                     # Experiment outputs (auto-generated)
│   ├── stage_00/ … stage_15/    # Per-stage result artifacts
│   └── stage_03b/, stage_03c/   # Sub-stage artifacts
├── stage01_math_checks.py       # Stage 01: Mathematical validation
├── stage02_generator.py         # Stage 02: Synthetic dataset generation
├── stage03b_ici_diagnostic.py   # Stage 03b: ICI diagnostic pipeline
├── stage03c_mode_c.py           # Stage 03c: Mode C validation
├── stage05_mode_b.py            # Stage 05: Mode B validation
├── stage06_coherence.py         # Stage 06: State coherence
├── stage07_robustness.py        # Stage 07: Robustness sweeps
├── test_*.py                    # Pytest test modules (24 files)
├── checkpointing.py             # Experiment state tracking + checkpoint recovery
├── cli.py                       # Command-line argument parsing
├── common.py                    # Shared utilities: chance-constraint tightening, risk scoring
├── ground_truth.py              # Synthetic environment simulator
├── run_all.py                   # Orchestration script (stages 01–15)
├── runtime.py                   # Experiment execution framework
├── specification.py             # Configuration composition + observation scheduling
├── utils.py                     # Atomic file I/O, directory management
├── config.json                  # Master configuration
├── smoke.json                   # Smoke profile overrides
├── standard.json                # Standard profile overrides
├── extended.json                # Extended profile overrides
├── validation.json              # Validation profile overrides
├── paper_defaults.json          # Reference parameter values from paper
└── config (N).json              # Parameter sweep variants (N = 1–58)
```

**Important**: `hdr_validation/` is a proper Python package directory containing `control/`, `inference/`, `model/`, `identification/` (v7.0), `stages/`, `utils.py`, `specification.py`, and `packaging.py`. Tests and scripts import the package as `hdr_validation.control.mpc`, `hdr_validation.inference.ici`, `hdr_validation.identification.hierarchical`, etc.

---

## Package Import Conventions

All production code imports via the `hdr_validation` package namespace:

```python
from hdr_validation.control.mpc import solve_mode_a
from hdr_validation.control.lqr import dlqr, committor, committor_with_jumps
from hdr_validation.control.mimpc import solve_mixed_integer_mpc, CumulativeExposureConstraint
from hdr_validation.control.supervisor import ExtendedSupervisor
from hdr_validation.inference.ici import compute_ici_state, compute_T_k_eff
from hdr_validation.inference.imm import IMMFilter, RegionConditionedIMM
from hdr_validation.inference.particle import ParticleFilter
from hdr_validation.inference.variational import VariationalSLDS
from hdr_validation.inference.population import PopulationPriorAssignment
from hdr_validation.model.slds import make_evaluation_model, BasinModel
from hdr_validation.model.target_set import build_target_set, TargetSet
from hdr_validation.model.hsmm import DwellModel
from hdr_validation.model.extensions import BasinClassifier, JumpDiffusion, PWACoupling
from hdr_validation.model.adaptive import FFRLSEstimator, DriftDetector
from hdr_validation.identification.hierarchical import HierarchicalCouplingEstimator
from hdr_validation.identification.boed import BOEDEstimator
from hdr_validation.utils import atomic_write_json, ensure_dir, seed_everything
from hdr_validation.packaging import zip_paths
```

Root-level modules (e.g., `checkpointing.py`, `common.py`, `specification.py`) use relative imports within stage scripts:

```python
from ..inference.ici import apply_calibration, compute_ici_state
from ..model.slds import make_evaluation_model
from .common import save_experiment_bundle
```

---

## Running the Validation Pipeline

### Full pipeline

```bash
python run_all.py                          # Run all stages, all profiles
python run_all.py --resume --skip-done     # Resume, skipping completed stages
python run_all.py --profiles smoke         # Smoke profile only (fastest)
python run_all.py --profiles smoke standard extended validation
```

### Selective stage execution

```bash
python run_all.py --profiles smoke --stages 03 04 --force    # Force-rerun stages 3 & 4
python run_all.py --stages 01 03b 03c                        # Multiple stages
python run_all.py --stages 12 13 14 15                       # v7.0 stages only
```

### Stage IDs

| ID   | Script                    | Description                              |
|------|---------------------------|------------------------------------------|
| 01   | `stage01_math_checks.py`  | Mathematical validation (τ̃, committor)   |
| 02   | `stage02_generator.py`    | Synthetic dataset generation             |
| 03   | (IMM inference)           | Mode identification and calibration      |
| 03b  | `stage03b_ici_diagnostic.py` | ICI calibration and regime boundaries |
| 03c  | `stage03c_mode_c.py`      | Mode C validation                        |
| 04   | (Mode A)                  | Mode A performance vs baselines          |
| 05   | `stage05_mode_b.py`       | Mode B structured exploration            |
| 06   | `stage06_coherence.py`    | State coherence checks                   |
| 07   | `stage07_robustness.py`   | Robustness across parameter sweeps       |
| 08   | `stage_08_ablation.py`    | Ablation study                           |
| 09   | `stage_09_baselines.py`   | Baseline comparison                      |
| 10   | `stage_10_mode_b_sweep.py` | Mode B FP/FN sweep                      |
| 11   | `stage_11_invariant_set.py` | Riccati invariant set verification     |
| 12   | `stage_12_hierarchical.py` | Hierarchical coupling estimation (v7.0) |
| 13   | `stage_13_inference_backbone.py` | Inference backbone benchmark (v7.0) |
| 14   | `stage_14_population_planning.py` | Population planning (v7.0)         |
| 15   | `stage_15_proxy_composite.py` | Proxy composite estimation (v7.0)    |

---

## Configuration System

### Profile hierarchy

Configs are composed from `config.json` (master defaults) + profile override (e.g., `smoke.json`). A config hash is computed for cache-key / checkpoint purposes.

```python
from hdr_validation.specification import compose_profile_config, config_hash
cfg = compose_profile_config(project_root, profile_name="smoke")
```

### Key configuration parameters

```json
{
  "state_dim": 8,
  "obs_dim": 16,
  "control_dim": 8,
  "disturbance_dim": 8,
  "K": 3,
  "H": 6,
  "kappa_lo": 0.55,
  "kappa_hi": 0.75,
  "pA": 0.70,
  "qmin": 0.15,
  "alpha_i": 0.05,
  "eps_safe": 0.01,
  "rho_reference": [0.72, 0.96, 0.55],
  "steps_per_day": 48,
  "dt_minutes": 30,
  "default_burden_budget": 28.0,
  "R_brier_max": 0.05,
  "sigma_dither": 0.08,
  "missing_fraction_target": 0.516
}
```

### Profile sizes

| Profile    | Seeds     | Episodes | Steps/ep | MC Rollouts |
|------------|-----------|----------|----------|-------------|
| smoke      | [101]     | 8        | 128      | 50          |
| standard   | [101,202] | 12       | 128      | 100         |
| extended   | [101,202,303] | 20   | 256      | 150         |
| validation | [101,202,303] | 12   | 128      | 150         |

---

## Testing

### Run tests

```bash
# Install pytest first if needed
pip install pytest numpy scipy pandas

# Run all tests
pytest

# Run specific test file
pytest test_ici.py -v
pytest test_mpc.py test_committor.py -v
```

### Test conventions

- Test files live at the repo root: `test_*.py`
- Tests import via `hdr_validation.*` package paths
- Use `tmp_path` pytest fixture for temporary directories
- No mocking — tests perform real functional operations
- Tests are self-contained: they create minimal configs inline

Example test pattern:
```python
def test_mpc_returns_bounded_control():
    config = {"state_dim": 8, "obs_dim": 16, ...}
    rng = np.random.default_rng(2)
    model = make_evaluation_model(config, rng)
    target = build_target_set(0, config)
    res = solve_mode_a(x, P, model.basins[0], target, 0.4, config, step=0)
    assert np.all(np.abs(res.u) <= 0.6 + 1e-8)
```

### Test files

| File                      | Tests                                          |
|---------------------------|-------------------------------------------------|
| `test_packaging.py`       | Zip archive creation                            |
| `test_ici.py`             | ICI state computation and bounds                |
| `test_imm.py`             | IMM inference filter                            |
| `test_mpc.py`             | MPC/Mode A control bounds                       |
| `test_mode_c.py`          | Mode C dither injection                         |
| `test_mode_c_fisher.py`   | Mode C Fisher information proxy                 |
| `test_hsmm.py`            | HSMM dwell distribution models                  |
| `test_target_set.py`      | Target set geometry                             |
| `test_committor.py`       | Committor BVP solution                          |
| `test_recovery.py`        | Recovery trajectory analysis                    |
| `test_stability_check.py` | Basin stability classification                  |
| `test_stage_08.py`        | Stage 08 ablation study                         |
| `test_stage_09.py`        | Stage 09 baseline comparison                    |
| `test_stage_10.py`        | Stage 10 Mode B sweep                           |
| `test_stage_11.py`        | Stage 11 invariant set                          |
| `test_extensions.py`      | v7.0 model extensions (25 tests)                |
| `test_adaptive.py`        | FF-RLS and drift detection (v7.0)               |
| `test_multirate.py`       | Multi-rate observation (v7.0)                   |
| `test_mimpc.py`           | Mixed-integer MPC (v7.0)                        |
| `test_supervisor.py`      | Extended supervisor 8-branch logic (v7.0)       |
| `test_particle.py`        | Particle filter / SMC (v7.0)                    |
| `test_variational.py`     | Variational SLDS inference (v7.0)               |
| `test_identification.py`  | Identification subpackage (v7.0, 15 tests)      |
| `test_committor_jump.py`  | Committor with jumps (v7.0)                     |

---

## Core Algorithms

### System model

Discrete-time switched linear system:
```
x_{t+1} = A_k x_t + B_k u_t + E_k w_t + b_k    (state evolution in basin k)
y_t     = C_k x_t + v_t + c_k                   (observations, with possible NaN missingness)
z_t ∈ {0,...,K-1}                               (discrete basin/mode)
```

K=3 basins with spectral radii `rho_reference = [0.72, 0.96, 0.55]`. Basin 1 (rho=0.96) is the "maladaptive" slow-escaping mode.

### Inference-Control Interface (ICI)

The ICI (`inference/ici.py`) defines three conditions for activating Mode C:

1. **Condition (i)**: `μ̂_k > μ̄_required` — mode error exceeds tolerable bound
2. **Condition (ii)**: `p_A^robust < ω_min` — robust Mode A probability too low
3. **Condition (iii)**: `T_k^eff < T_C_max` — effective sample count insufficient

Key ICI quantities:
- `T_k^eff = T * π_k * (1 - p_miss) * (1 - ρ_k)` — effective sample count (Prop 9.2)
- `p_A^robust = p_A_nominal + k_calib * R_Brier` — calibration-adjusted Mode A probability
- `ω_min = omega_min_factor * T_k^eff` — minimum viable probability threshold

### Control hierarchy

```
ICI state → Mode selection → Control law
   │
   ├── All conditions False → Mode A: solve_mode_a() (MPC with Riccati recursion)
   ├── Condition (i)/(ii)/(iii) True + Mode B eligible → Mode B: excitation control
   └── Mode C pre-emption → Mode C: Fisher-information dither
```

### Safety and burden

- Control bounds: `u ∈ [-0.6, 0.6]^m` (hard clip)
- Circadian mask: some control dimensions locked at night (`steps_per_day=48`, 30min/step)
- Burden budget: `Σ|u_t| ≤ default_burden_budget` per episode
- Safety fallback: `u *= 0.5` when safety constraint active

---

## Variable Naming Conventions

| Variable | Meaning |
|----------|---------|
| `x`, `x_hat` | Continuous state (true, estimated) |
| `z`, `z_hat` | Discrete mode/basin (true, estimated) |
| `u` | Control input |
| `y` | Observation vector (may contain NaN for missing) |
| `K` | Number of basins/modes |
| `n` | State dimension (`state_dim`) |
| `m` | Observation dimension (`obs_dim`) |
| `rho_k` | Spectral radius of basin k dynamics |
| `T_k_eff` | Effective sample count for basin k |
| `mu_hat` | Estimated mode-error probability |
| `R_Brier` | Brier calibration score (reliability) |
| `p_A_robust` | Calibration-adjusted Mode A probability |
| `tau_tilde` | Recovery burden proxy (dist²/(1-ρ²)) |
| `A, B, C, E` | System, input, observation, disturbance matrices |
| `Q, R` | Process, observation noise covariance |
| `P` | Kalman/Riccati covariance matrix |
| `kappa` | Target set scale (kappa_lo to kappa_hi) |
| `w1, w2, w3` | Mode A/B/coherence loss weights |

---

## Checkpointing and Resumability

The `checkpointing.py` module provides a `RunManifest` class that tracks experiment state:

```python
manifest = RunManifest(project_root)
if manifest.should_skip(stage_id, profile_name, config_hash):
    return  # Already completed with same config

manifest.mark_running(stage_id, profile_name)
try:
    result = run_stage(...)
    manifest.mark_completed(stage_id, profile_name, result)
except Exception as e:
    manifest.mark_failed(stage_id, profile_name, str(e))
```

- Manifest stored as `run_manifest.json` in project root
- Config hash ensures cache invalidation on parameter changes
- Partial results flushed atomically after every seed/chunk
- Use `--skip-done` flag to skip already-completed stages

---

## Atomic File I/O

All file writes use atomic operations (write to temp, then rename):

```python
from hdr_validation.utils import (
    atomic_write_json,   # JSON serialization
    atomic_write_text,   # Plain text
    atomic_save_npz,     # NumPy compressed arrays
    ensure_dir,          # mkdir -p
    seed_everything,     # Set numpy + Python random seeds
)
```

NumPy episode data format:
```python
np.savez_compressed(path,
    x_true=...,    # (T, n) true latent state
    z_true=...,    # (T,) true discrete mode
    y=...,         # (T, m) observations (NaN = missing)
    mask=...,      # (T, m) bool observation mask
    u=...,         # (T, u_dim) applied controls
    kappa=...,     # target set scale per step
)
```

---

## Result Artifacts

Each stage outputs to `results/stage_{id}/{profile_name}/{component}/`:

```
results/stage_03b/smoke/ici_diagnostic/
├── config.json       # Config snapshot
├── seed.json         # Seed used
├── summary.json      # Scalar metrics
├── metrics.csv       # Tabular results
├── manifest.json     # Artifact manifest
└── plots/            # Optional visualizations
```

---

## Claim Validation

The suite validates 32 claims across v5.0 and v7.0:

- **Claims 1–4**: ICI correctness, Mode A improvement, τ̃ correlation, chance-constraint calibration
- **Claims 5–6**: ISS scaling, stability under drift
- **Claims 7–8**: Mode B improvement and DP approximation quality
- **Claims 9–10**: Coherence penalty, identifiability improvement
- **Claims 11–14**: ICI regime identification, Mode C Fisher improvement, p_A^robust FP reduction, compound bound correctness
- **Claims 15–18** (v7.0): Basin classification, reversible/irreversible partition, PWA coupling, multi-site Gershgorin bounds
- **Claims 19–22** (v7.0): Jump-diffusion, cumulative exposure, state-conditioned coupling, modular expansion
- **Claims 23–26** (v7.0): FF-RLS drift tracking, multi-rate observation, MI-MPC binary constraints, extended supervisor
- **Claims 27–28** (v7.0): Particle filter consistency, hierarchical coupling convergence
- **Claims 29–32** (v7.0): B_k sample complexity, basin boundary convergence, population planning, proxy-composite estimation

Claims 1–14 are evaluated by stages 01–11; Claims 15–32 by stages 12–15.
A claim is marked `Supported` only when it passes its criterion in both smoke and standard profiles.

See `CLAIM_CRITERIA.md` for full criterion definitions and `CLAIM_MATRIX.md` for current status.

---

## Key Constraints and Assumptions

1. **No external solvers**: `cvxpy` and `osqp` are not available. Use SciPy SLSQP for projection checks.
2. **No internet access**: All data is synthetic and generated locally.
3. **Python ≥ 3.10** required (`from __future__ import annotations` used throughout).
4. **Deterministic seeds**: All randomness uses `np.random.default_rng(seed)`.
5. **Mode A is approximate**: Uses Riccati recursion + box projection, not full robust tube MPC.
6. **EM updates are restricted**: Only `C_k`, `R_k`, and selected `A_k`, `B_k` via weighted regression.
7. **Coherence**: Evaluated from PLV-like summary of oscillatory axes, not full predictive coherence model.
8. **Control grid**: 30-minute steps (`dt_minutes=30`, `steps_per_day=48`).

---

## Dependencies

Core scientific stack (Python 3.10+):

```
numpy
scipy         # linalg (DARE, Lyapunov, QR), stats (norm, poisson, lognorm, zipf, weibull)
pandas        # tabular result aggregation
pytest        # test runner
```

Optional (for plotting):
```
matplotlib    # visualization in plotting.py
```

---

## File Naming Note

Some files at the repository root have names that do not match their content due to upload history (multiple "Add files via upload" commits). When navigating the codebase, verify file content rather than relying solely on filename. The canonical organized package is in `control/`, `inference/`, and `model/` subdirectories.
