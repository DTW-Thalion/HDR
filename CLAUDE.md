# CLAUDE.md — HDR Validation Suite

## Project Overview

This repository is an **in-silico validation suite** for the **Homeodynamic Remediation Framework (HDR) v7.3** — a multi-mode adaptive control system for constrained stochastic linear dynamical systems (SLDS). The framework validates mathematical properties, implementation correctness, and empirical performance across synthetic physiological scenarios.

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
│   │   ├── mode_c_fisher.py     # Fisher-information dither policy and proxy (Mode C)
│   │   ├── mimpc.py             # Mixed-Integer MPC (v7.0)
│   │   ├── supervisor.py        # Extended 9-branch supervisor (v7.0+)
│   │   ├── mpc.py               # Model Predictive Control (Mode A), solve_mode_a_unstable, solve_mode_a_irr
│   │   └── tube_mpc.py          # mRPI terminal set and tube-MPC (v7.1)
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
│   │   ├── adaptive.py          # Forgetting-factor RLS, drift detection, adaptive mismatch bound (v7.1)
│   │   ├── saturation.py        # Michaelis-Menten saturating dose-response (v7.1)
│   │   ├── stability_check.py   # Basin spectral radius validation
│   │   └── multirate.py         # Multi-rate observer, delay augmentation (v7.0)
│   ├── identification/          # v7.0 identification subpackage
│   │   ├── hierarchical.py      # Hierarchical empirical-Bayes coupling estimation
│   │   ├── boed.py              # Bayesian Optimal Experimental Design
│   │   ├── committor_recovery.py # Committor recovery via kernel regression
│   │   ├── transition_rates.py  # Transition rate estimation
│   │   ├── population_planning.py # Population-prior treatment planning
│   │   ├── tau_estimation.py    # Tau burden estimation
│   │   └── risk_information.py  # Risk-information frontier
│   └── stages/                  # Stage scripts for stages 08–17
│       ├── stage_08_ablation.py
│       ├── stage_08b_ablation.py # Multi-axis asymmetric ablation (coherence + calibration marginal gains)
│       ├── stage_09_baselines.py
│       ├── stage_10_mode_b_sweep.py
│       ├── stage_11_invariant_set.py
│       ├── stage_12_hierarchical.py  # v7.0
│       ├── stage_13_inference_backbone.py  # v7.0
│       ├── stage_14_population_planning.py  # v7.0
│       ├── stage_15_proxy_composite.py  # v7.0
│       ├── stage_16_extensions.py  # v7.1 — model-failure extension integration
│       └── stage_17_gompertz.py  # v7.5 — emergent Gompertz mortality & complexity collapse
├── results/                     # Experiment outputs (auto-generated)
│   └── stage_04/ … stage_17/    # Per-stage result artifacts (incl. stage_08b)
├── smoke_runner.py              # Smoke profile runner (stage functions + SMOKE_CONFIG)
├── standard_runner.py           # Standard profile runner
├── extended_runner.py           # Extended profile runner
├── extended_512_runner.py       # Extended profile with T=512
├── validation_runner.py         # Validation profile runner
├── highpower_runner.py          # High-power profile runner (20 seeds × 30 ep/seed)
├── test_*.py                    # Pytest test modules (31 files, 312 tests)
├── run_all.py                   # Orchestration script (stages 01–17, --full-validation)
├── plotting.py                  # Visualization utilities
├── analyse_highpower.py         # High-power run analysis
├── analyse_mismatch.py          # Model mismatch analysis
├── derive_criterion.py          # Criterion derivation utility
├── generate_reports.py          # Report generation utility
├── config.json                  # Master configuration
├── paper_defaults.json          # Reference parameter values from paper
└── config (N).json              # Parameter sweep variants
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
from hdr_validation.model.coherence import damping_ratio, coherence_penalty, coherence_grad
from hdr_validation.model.saturation import michaelis_menten, apply_saturation
from hdr_validation.control.tube_mpc import compute_mRPI_zonotope, solve_tube_mpc
from hdr_validation.identification.hierarchical import HierarchicalCouplingEstimator
from hdr_validation.identification.boed import BOEDEstimator
from hdr_validation.utils import ensure_dir, atomic_write_text
from hdr_validation.packaging import zip_paths
from hdr_validation.specification import observation_schedule, generate_observation
```

Stage logic for stages 01–07 lives in the profile runner modules (`smoke_runner.py`, `standard_runner.py`, etc.). Stages 08–17 (including 08b) are in `hdr_validation/stages/`.

---

## Running the Validation Pipeline

### Recommended: full validation (all 34 claims)

```bash
python run_all.py --full-validation        # Complete validation of all 34 claims
```

This is the **recommended entry point** for reviewers. It runs four phases:

| Phase | What runs | Claims covered |
|-------|-----------|----------------|
| 1 | Extended profile, stages 01–03c + 05–07 | 3–14 |
| 2 | Highpower benchmark (20 seeds × 30 ep/seed) | 1–2 (authoritative) |
| 3 | Stages 08–17 at production scale + pytest | 9, 13, 15–34 |
| 4 | Full pytest suite (312 tests, 31 files) | 15–34 (unit test layer) |

Output ends with a per-claim pass/fail summary table. Supports `--resume --skip-done`
for resuming interrupted runs, and `--force` to re-run completed stages.

### Per-profile runs

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
python run_all.py --stages 08 08b --run-tests                # Run stages then pytest
```

### Stage IDs

| ID   | Script                    | Description                              |
|------|---------------------------|------------------------------------------|
| 01   | (in profile runner)       | Mathematical validation (τ̃, committor)   |
| 02   | (in profile runner)       | Synthetic dataset generation             |
| 03   | (in profile runner)       | Mode identification and calibration      |
| 03b  | (in profile runner)       | ICI calibration and regime boundaries    |
| 03c  | (in profile runner)       | Mode C validation                        |
| 04   | (in profile runner)       | Mode A performance vs baselines          |
| 05   | (in profile runner)       | Mode B structured exploration            |
| 06   | (in profile runner)       | State coherence checks                   |
| 07   | (in profile runner)       | Robustness across parameter sweeps       |
| 08   | `stage_08_ablation.py`    | Ablation study                           |
| 08b  | `stage_08b_ablation.py`   | Multi-axis asymmetric ablation           |
| 09   | `stage_09_baselines.py`   | Baseline comparison                      |
| 10   | `stage_10_mode_b_sweep.py` | Mode B FP/FN sweep                      |
| 11   | `stage_11_invariant_set.py` | Riccati invariant set verification     |
| 12   | `stage_12_hierarchical.py` | Hierarchical coupling estimation (v7.0) |
| 13   | `stage_13_inference_backbone.py` | Inference backbone benchmark (v7.0) |
| 14   | `stage_14_population_planning.py` | Population planning (v7.0)         |
| 15   | `stage_15_proxy_composite.py` | Proxy composite estimation (v7.0)    |
| 16   | `stage_16_extensions.py` | Model-failure extension integration (v7.1) |
| 17   | `stage_17_gompertz.py` | Emergent Gompertz mortality & complexity collapse (v7.5) |

---

## Configuration System

### Profile hierarchy

Each profile runner (e.g., `smoke_runner.py`) defines its own config dict inline (e.g., `SMOKE_CONFIG`). Additional runners include `extended_512_runner.py` (T=512 variant) and `highpower_runner.py` (20-seed Benchmark A). The master defaults come from `config.json`.

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

| Profile      | Seeds           | Episodes | Steps/ep | MC Rollouts |
|--------------|-----------------|----------|----------|-------------|
| smoke        | [101]           | 8        | 128      | 50          |
| standard     | [101,202]       | 12       | 128      | 100         |
| extended     | [101,202,303]   | 20       | 256      | 150         |
| extended_512 | [101,202,303]   | 10       | 512      | 150         |
| validation   | [101,202,303]   | 12       | 128      | 150         |
| highpower    | 20 seeds        | 30/seed  | 256      | —           |

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

# Run tests filtered by profile (cumulative tiers)
pytest --profile smoke          # 94 tests  — core + stage integration
pytest --profile standard       # 188 tests — smoke + ICI/Mode-C/coherence
pytest --profile extended       # 313 tests — standard + v7.0/v7.1 extensions
pytest --profile validation     # 313 tests — all tests
```

### Profile test counts

| Profile    | Tests | Files | What it adds                                      |
|------------|-------|-------|---------------------------------------------------|
| smoke      | 94    | 12    | Core fast tests + stage integration tests          |
| standard   | 188   | 18    | + ICI, Mode C, Fisher, coherence, committor-jump   |
| extended   | 313   | 31    | + extensions, adaptive, identification, supervisor |
| validation | 313   | 31    | All tests (same as extended)                       |

Profile filtering is configured in `conftest.py` via the `--profile` CLI option.

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

31 test files at the repo root (`test_*.py`) cover all 34 claims. See `CLAIM_MATRIX.md`
for the per-claim test file mapping. Run `python check_claims.py --verbose` for automated
validation of claim criteria against test results and stage artifacts.

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

1. **Condition (i)**: `μ̂ ≥ μ̄_required` — mode error exceeds tolerable bound
2. **Condition (ii)**: `R_Brier ≥ R_Brier_max` — calibration error exceeds maximum
3. **Condition (iii)**: `any T_k^eff < ω_min` — effective sample count below threshold

Key ICI quantities:
- `T_k^eff = T * π_k * (1 - p_miss) * (1 - ρ_k)` — effective sample count (Prop 9.2)
- `p_A^robust = p_A_nominal + k_calib * R_Brier` — calibration-adjusted Mode A probability
- `ω_min` — regime boundary threshold for minimum effective sample count

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

`run_all.py` manages a JSON manifest (`run_all_manifest.json`) to track stage completion:

```python
# In run_all.py:
manifest = load_manifest()            # Load existing state
is_done(manifest, profile, stage_id)  # Check if stage completed
mark_done(manifest, profile, stage_id)  # Mark stage as done
mark_failed(manifest, profile, stage_id)  # Mark stage as failed
save_manifest(manifest)               # Persist to disk
```

- Manifest stored as `run_all_manifest.json` in project root
- Use `--skip-done` flag to skip already-completed stages
- Use `--resume` to load existing manifest state
- Use `--force` to re-run regardless of manifest

---

## Atomic File I/O

All file writes use atomic operations (write to temp, then rename):

```python
from hdr_validation.utils import (
    atomic_write_text,   # Plain text (write-to-temp + rename)
    ensure_dir,          # mkdir -p
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

Stages 01–07 (profile-runner stages) output to `results/stage_{id}/{profile_name}/{component}/`:

```
results/stage_03b/smoke/ici_diagnostic/
├── config.json       # Config snapshot
├── seed.json         # Seed used
├── summary.json      # Scalar metrics
├── metrics.csv       # Tabular results
├── manifest.json     # Artifact manifest
└── plots/            # Optional visualizations
```

Stages 08–17 (profile-independent stages) output flat JSON files directly:

```
results/stage_08/
├── ablation_results.json      # Production-scale ablation output
└── ablation_diagnosis.json    # Diagnostic/fix history
```

---

## Claim Validation

The suite validates 34 claims across v5.0, v7.0, v7.1, and v7.5:

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
- **Claims 33–34** (v7.5): Emergent Gompertz mortality law, Lipsitz–Goldberger complexity collapse

Claims 1–14 are evaluated by stages 01–11; Claims 15–32 by stages 12–16; Claims 33–34 by stage 17.
A claim is marked `Supported` only when it passes its criterion in the appropriate profile.
Claims 1–2 require the highpower runner (20 seeds × 30 ep/seed) for authoritative validation.

**To validate all 34 claims in a single run:** `python run_all.py --full-validation`

See `CLAIM_CRITERIA.md` for full criterion definitions and `CLAIM_MATRIX.md` for current status.

---

## Key Constraints and Assumptions

1. **No external solvers**: `cvxpy` and `osqp` are not available. Use SciPy SLSQP for projection checks.
2. **No internet access**: All data is synthetic and generated locally.
3. **Python ≥ 3.10** required (`from __future__ import annotations` used throughout).
4. **Deterministic seeds**: All randomness uses `np.random.default_rng(seed)`.
5. **Mode A**: Uses Riccati recursion + box projection by default; tube-MPC with mRPI terminal set available via `tube_mpc.py` (v7.1).
6. **EM updates are restricted**: Only `C_k`, `R_k`, and selected `A_k`, `B_k` via weighted regression.
7. **Coherence**: The coherence measure κ̂_t is operationalised as the damping ratio ζ = |Re(λ₁)|/|λ₁| of the least-stable eigenvalue (Remark B.1). Values near 1 indicate overdamped (healthy); declining values indicate underdamped (degraded). The `coherence_penalty` and `coherence_grad` functions take a scalar κ̂ and are operationalisation-agnostic.
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

Some production source files in `hdr_validation/` may have names that do not perfectly match their content due to upload history (multiple "Add files via upload" commits). The canonical organized package is in `control/`, `inference/`, `model/`, and `identification/` subdirectories. Test files at the repo root have been renamed to match their actual content.
