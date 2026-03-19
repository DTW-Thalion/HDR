# Assumptions used by this implementation

These are the smallest defensible assumptions added where the paper leaves operational degrees of freedom.

## Numerical and solver assumptions

1. `cvxpy` and `osqp` are not available in the execution environment.
2. Stage 01 diagnostics may use SciPy SLSQP for projection checks.
3. The online controller uses a conservative axis-aligned box surrogate of the full target-set intersection for speed and resumability.
### Mode A controller — implementation vs. manuscript description

**What the manuscript calls it:** robust tube MPC with chance-constraint
tightening (Appendix B, Proposition E.2).

**What is actually implemented** (`hdr_validation/control/mpc.py`,
function `solve_mode_a`):

1. Chance-constraint tightening: observation-space margins propagated
   to state-space via C-pseudoinverse to form a tightened box S*_δ
   (approximates the tube-MPC inner set).
2. Reference projection: x_ref = clip(Π(x̂, S*), tight_low, tight_high).
   This is a box-clip, not a full SLSQP projection (SLSQP is used only
   in Stage 01 math checks, not in the online controller).
3. Effective cost matrix: Q_eff = (w1 + w2/(1-ρ²))·I plus coherence
   and constraint-tightness boosts.  Correctly implements τ̃ weighting
   from Equation (13).
4. Finite-horizon gains: initialised from DARE terminal cost with
   inflated R (mismatch-robust gain), rolled out for H steps.
5. Safety fallback: clamp-and-scale when risk score > 3·eps_safe.

**Why this is acceptable:** The approximation is consistent with the
manuscript's architectural description at the level of concepts
(tightened set, projected reference, Riccati terminal cost).  The gap
is in degree of optimality, not architectural correctness.

**v7.1 update:** A tube-MPC implementation is now available in
`hdr_validation/control/tube_mpc.py`. It computes a maximal Robust
Positively Invariant (mRPI) set via the Raković et al. (2005) algorithm
using zonotope representation, and wraps `solve_mode_a` with nominal +
ancillary feedback decomposition: u = u_bar + K_fb @ (x_hat - x_bar).
This can be activated via `use_tube_mpc=True` in Stage 11. The default
Mode A controller (`solve_mode_a`) remains the approximate version
described above.

## Time-grid assumption

1. The resource-safe master control grid is `30 minutes` in standard operation, following the prompt override.
2. Smoke and extended profiles keep the same coarse grid but change horizon length through more steps per episode rather than finer real-time integration by default.

## Synthetic physiology assumptions

1. The latent state is standardized and centered so that the nominal target box is near zero.
2. Control dimension is set equal to state dimension (`m = 8`) to keep intervention channels interpretable and lightweight.
3. Observation dimension is `16`, with two channels per latent axis and heterogeneous schedules.
4. Coherence is operationalised as the damping ratio ζ = |Re(λ₁)|/|λ₁| of the least-stable eigenvalue of the estimated basin dynamics matrix (Remark B.1). The coherence penalty and gradient functions take this scalar ζ as input and are operationalisation-agnostic.
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
