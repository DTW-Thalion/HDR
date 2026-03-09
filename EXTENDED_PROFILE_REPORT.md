# Claim support criteria

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
