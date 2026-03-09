# Assumptions used by this implementation

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
