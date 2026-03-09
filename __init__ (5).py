from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, atomic_write_text, environment_snapshot, load_json, stable_hash


PACKAGE_CANDIDATES = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pytest",
    "cvxpy",
    "osqp",
]


def detect_packages() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in PACKAGE_CANDIDATES}


def load_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    config_dir = project_root / "configs"
    names = ["paper_defaults", "resource_safe_defaults", "smoke", "standard", "extended", "validation"]
    return {name: load_json(config_dir / f"{name}.json") for name in names}


def compose_profile_config(project_root: Path, profile_name: str) -> dict[str, Any]:
    cfgs = load_configs(project_root)
    merged: dict[str, Any] = {}
    for name in ["paper_defaults", "resource_safe_defaults", profile_name]:
        merged.update(cfgs[name])
    merged["profile_name"] = profile_name
    return merged


def write_environment_report(project_root: Path) -> Path:
    packages = detect_packages()
    report = environment_snapshot(packages)
    out = project_root / "results" / "environment_report.json"
    atomic_write_json(out, report)
    return out


def specification_markdown(using_paper: bool = True) -> str:
    source_note = "Primary source: attached HDR working paper v4.3." if using_paper else "Primary source: embedded prompt specification because no paper attachment was available."
    return f"""# HDR operational specification used for this build

{source_note}

## Interpretation rule

This repository uses the paper as the formal specification **for implementation targets**, but it deliberately does **not** inherit the paper's own in silico results from Section 13 nor its Discussion/Conclusion sections as evidence. Those sections are treated as prior claims to be re-tested independently.

## Core state and regimes

- Latent state dimension `n = 8`
- Axes:
  1. immunosenescence / inflammaging
  2. metabolic dysfunction
  3. epigenetic drift
  4. mitochondrial decline
  5. proteostasis stress
  6. circadian dysregulation
  7. neuroendocrine imbalance
  8. musculoskeletal / functional state
- Baseline basin count `K = 3`
  - `0`: desired / healthy
  - `1`: maladaptive / near-unit-root
  - `2`: transient / stress response
- Optional `K = 4` used only in selected sweeps

## Evaluation-model dynamics

Within basin `k`, the evaluator uses a switching linear state-space model:

- `x[t+1] = A[k] x[t] + B[k] u[t] + E[k] d[t] + b[k] + w[t]`
- `y[t]   = C[k] x[t] + c[k] + v[t]`

Default reference spectral radii used in the synthetic baseline design:

- desired: about `0.72`
- maladaptive: about `0.96`
- transient: about `0.55`

## Dwell model

- Basin process uses an HSMM-style dwell model
- Maladaptive dwell is intentionally heavy-tailed / low-hazard

## Mode A objective

Finite-horizon controller with stage cost

`w1 * dist_Q(x_t, S*(t))^2 + w2 * tau_tilde(x_t, z_t) + w3 * g(kappa_hat_t) + lambda * ||u_t||_R^2`

where

- `tau_tilde(x, z) = dist_Q(x, S*)^2 / (1 - rho(A[z])^2)`
- `g(kappa)` is zero in-band and quadratic outside `[kappa_lo, kappa_hi]`

Defaults:

- `H = 6`
- `kappa_lo = 0.55`
- `kappa_hi = 0.75`
- `pA = 0.70`
- `qmin = 0.15`
- `alpha_i = 0.05`
- `eps_safe = 0.01`
- weight ratio `w1:w2:w3 = 1.0:0.5:0.3`
- `lambda = 0.1`

## Target set

`S*(t) = X_safety(t) ∩ X_cohort(t) ∩ X_personal(t) ∩ X_function(t)`

Fallback rule: if the intersection is empty, use `X_safety`.

## Safety constraints

- control bounds
- burden budget
- circadian compatibility windows
- chance-constraint tightening based on predicted observation covariance

## Mode B trigger

Entry requires all of:

1. posterior maladaptive probability at least `pA`
2. entrenchment diagnostic positive
3. committor estimate at most `qmin`

Two implementations are required:

- exact reduced discrete controlled MDP reference
- heuristic committor-based supervisor for the hybrid system

Safety fallback is always available.

## Observation model

Evaluation stack uses a linear-factor observation model with masked heterogeneous timing:

- fast channels frequently observed
- slow channels intermittent
- sporadic channels rare with optional smoothing-like update

For resource-safe runs, a `30 minute` master control grid is used and heterogeneous updates are emulated via masks on that grid.

## Ground-truth generator requirements

The synthetic ground-truth generator must be richer than the evaluation model:

- switching multi-basin dynamics
- mild nonlinearity
- slow drift in cross-axis couplings
- bounded control delays
- heavy-tailed or mixture process noise in robustness settings
- heavy-tailed maladaptive dwell times
- control-dependent transition bias in selected scenarios
- multi-rate observations with missingness
- heteroskedastic observation noise
- optional nonlinear emission perturbation
- slowly drifting target set
- challenge library:
  - orthostatic-like pulse
  - mixed-meal-like pulse
  - exercise-like pulse
  - optional acute-stress pulse

## Validation stance

This repository is explicitly restricted to **synthetic numerical validation**.
No biological, translational, or clinical claims are made.
"""


def assumptions_markdown() -> str:
    return """# Assumptions used by this implementation

These are the smallest defensible assumptions added where the paper leaves operational degrees of freedom.

## Numerical and solver assumptions

1. `cvxpy` and `osqp` are not available in the execution environment.
2. Stage 01 diagnostics may use SciPy SLSQP for projection checks.
3. The online controller uses a conservative axis-aligned box surrogate of the full target-set intersection for speed and resumability.
4. Mode A is implemented as an approximate constrained finite-horizon tracking controller using Riccati recursion, clipping, burden scaling, deterministic tightening, and safety fallback. It is **not** a full robust tube MPC solve.

## Time-grid assumption

1. The resource-safe master control grid is `30 minutes` in standard operation, following the prompt override.
2. Smoke and extended profiles keep the same coarse grid but change horizon length through more steps per episode rather than finer real-time integration by default.

## Synthetic physiology assumptions

1. The latent state is standardized and centered so that the nominal target box is near zero.
2. Control dimension is set equal to state dimension (`m = 8`) to keep intervention channels interpretable and lightweight.
3. Observation dimension is `16`, with two channels per latent axis and heterogeneous schedules.
4. Coherence is evaluated from a PLV-like summary of selected oscillatory axes rather than from a full predictive coherence state model.
5. Mode B control-dependent transition bias is implemented as a bounded perturbation of discrete transition tendencies and early-exit hazard, not as a mechanistic intervention model.

## Identification assumptions

1. EM-style updates are restricted to `C_k` and `R_k` plus local weighted regressions for selected `A_k`, `B_k`.
2. Hierarchical-prior emulation is implemented as ridge shrinkage toward paper-default matrices.
3. Dither injection during identification is small, bounded, and synthetic only.

## Evaluation assumptions

1. Per-step full traces are saved only for a capped subset of episodes.
2. Large robustness sweeps use aggregate metrics with a small raw-trace sample.
3. Standard and extended profiles run selected coarse sweeps rather than exhaustive high-resolution sweeps over every parameter combination.

## Claiming assumptions

1. Support labels are assigned only from this repository's own synthetic experiments.
2. A claim can be marked `Supported` only when it passes its predeclared criterion in both smoke and standard runs, unless explicitly flagged otherwise.
"""


def claim_criteria_markdown() -> str:
    return """# Claim support criteria

Labels allowed in the final claim matrix:

- Supported
- Partially supported
- Not supported
- Inconclusive
- Skipped

## Rules by claim type

### 1. Non-oracle state/mode inference is adequate for control
Supported:
- standard-profile closed-loop HDR beats pooled LQR on cumulative cost by at least 10% median paired improvement,
- while non-oracle state RMSE stays below 0.9 standardized units on at least 6/8 axes,
- and mode F1 for maladaptive detection is at least 0.65.

Partially supported:
- control improvement is present but smaller than 10%, or
- RMSE / F1 passes only in some scenarios.

Not supported:
- no consistent improvement or inference quality too poor for closed-loop use.

### 2. Mode A improves over simple baselines without increasing safety violations
Supported:
- HDR beats open-loop and pooled LQR by at least 10% on cumulative cost,
- and safety-violation rate is not worse by more than 0.5 percentage points,
- and burden adherence remains at least 95%.

Partially supported:
- improvement only against weaker baselines,
- or safety is slightly worse but still within 1.5 percentage points.

### 3. tau_tilde tracks or ranks true recovery burden sufficiently
Supported:
- Spearman rank correlation between `tau_tilde` and Lyapunov or empirical recovery burden is at least 0.7 in standard runs.

Partially supported:
- correlation between 0.45 and 0.7.

### 4. Chance-constraint tightening is empirically calibrated in Gaussian settings
Supported:
- observed violation rate is within ±1.5 percentage points of nominal under Gaussian toy and control settings.

Partially supported:
- within ±3 percentage points.

### 5. Practical stability under mode error is numerically consistent with sqrt(mu)-type degradation
Supported:
- residual-vs-sqrt(mu) fit has slope > 0 and R^2 >= 0.75.

Partially supported:
- slope > 0 and R^2 between 0.45 and 0.75.

### 6. Practical stability under drifting S*(t) is numerically consistent with linear-in-drift degradation
Supported:
- residual-vs-drift fit slope > 0 and R^2 >= 0.75.

Partially supported:
- slope > 0 and R^2 between 0.45 and 0.75.

### 7. Mode B heuristic improves escape versus conservative baselines
Supported:
- escape probability improves by at least 10 percentage points in the reduced-chain and hybrid evaluations without materially increasing safety violations (>1 percentage point).

Partially supported:
- improvement is smaller or safety worsens modestly.

### 8. Mode B remains acceptably close to exact DP on reduced discrete problems
Supported:
- absolute escape-probability gap <= 0.05 and median time-to-escape gap <= 10%.

Partially supported:
- probability gap <= 0.10.

### 9. Coherence penalty behaves as designed
Supported:
- standalone tests confirm zero in-band and monotone outside-band,
- and integrated experiments improve time-in-band by at least 10 percentage points in under/over-coupled scenarios.

Partially supported:
- standalone behavior passes but integrated benefit is small or mixed.

### 10. Identifiability improves with perturbations, priors, and dither
Supported:
- parameter-recovery error improves by at least 10% median in each of the three comparisons (with perturbations vs without, with priors vs without, with dither vs without).

Partially supported:
- improvement holds for some but not all three.

## Negative labels

Not supported:
- experiment directly contradicts the claim.

Inconclusive:
- experiment exists but is too noisy, too underpowered, or conflicting.

Skipped:
- dependency or runtime limitation prevented a meaningful run.
"""


def validation_plan_markdown() -> str:
    return """# Validation plan

## Stage 00
Bootstrap environment, detect package availability, initialize manifest, write specification and assumptions docs, and package stage archives.

## Stage 01
Mathematical and component checks:
- target-set construction and fallback
- projection diagnostics
- recovery surrogate and Lyapunov comparison
- coherence penalty checks
- committor and DP correctness
- HSMM hazard and dwell sampling
- chance-constraint tightening
- DARE terminal ingredients

## Stage 02
Synthetic dataset creation:
- train / validation / test datasets
- identification-focused and challenge-library episodes
- compressed data + manifests + generator diagnostics

## Stage 03
Observer and identification validation:
- IMM + Kalman + HSMM
- local EM-style `C_k`, `R_k` updates
- weighted regressions for selected dynamics parameters
- identifiability sweeps over data length, priors, perturbations, dither, near-unit-root sensitivity

## Stage 04
Non-oracle Mode A closed-loop validation:
- nominal and mismatched runs
- missingness / heteroskedasticity / delayed-control / target-drift scenarios
- baseline comparisons and ablations
- practical-stability sweeps and chance-calibration study

## Stage 05
Mode B validation:
- exact reduced discrete MDP reference
- heuristic supervisor comparison
- hybrid continuous/discrete supervisor evaluation with false positives / negatives and fallback analysis

## Stage 06
Coherence / coupling validation:
- standalone under-coupled, in-band, over-coupled tests
- integrated closed-loop tests with and without coherence regularization

## Stage 07
Robustness, falsification, phase maps:
- structured sweeps over noise, missingness, delay, mismatch, K, H, spectral radius, coupling pattern, burden budget, pA/qmin, alpha_i
- negative-control oracle / inverse-crime optimism gap
"""


def reproducibility_markdown() -> str:
    return """# Reproducibility

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
python run_all.py --profiles smoke --stages 03,04 --force
```

## No internet / no external downloads

This repository uses only synthetic data generated locally. It performs no network access and no external dataset downloads.
"""


def initialize_docs(project_root: Path, using_paper: bool = True) -> None:
    docs = project_root / "docs"
    atomic_write_text(docs / "SPECIFICATION.md", specification_markdown(using_paper=using_paper))
    atomic_write_text(docs / "ASSUMPTIONS.md", assumptions_markdown())
    atomic_write_text(docs / "CLAIM_CRITERIA.md", claim_criteria_markdown())
    atomic_write_text(docs / "VALIDATION_PLAN.md", validation_plan_markdown())
    atomic_write_text(docs / "REPRODUCIBILITY.md", reproducibility_markdown())

    placeholders = {
        docs / "FAILURES.md": "# Failures\n",
        docs / "CLAIM_MATRIX.md": "# Claim matrix\n\n_To be populated after experiments._\n",
        docs / "VALIDATION_REPORT.md": "# Validation report\n\n_To be populated after experiments._\n",
        docs / "RESULTS_INDEX.md": "# Results index\n\n_To be populated after experiments._\n",
    }
    for path, content in placeholders.items():
        if not path.exists():
            atomic_write_text(path, content)


def config_hash(project_root: Path, profile_name: str, stage_name: str) -> str:
    cfg = compose_profile_config(project_root, profile_name)
    payload = {"stage": stage_name, "profile": profile_name, "config": cfg}
    return stable_hash(payload)
