from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import atomic_write_text, dict_to_markdown_table, load_json


def _load_stage_summary(project_root: Path, stage: str, profile: str) -> dict[str, Any]:
    path = project_root / "results" / stage / profile / "stage_summary.json"
    if not path.exists():
        alt = project_root / "results" / stage / profile / "summary.json"
        if alt.exists():
            return load_json(alt, {})
        return {}
    return load_json(path, {})


def _claim(label_supported: bool, label_partial: bool, missing: bool = False) -> str:
    if missing:
        return "Skipped"
    if label_supported:
        return "Supported"
    if label_partial:
        return "Partially supported"
    return "Not supported"


def build_claim_matrix(project_root: Path) -> list[dict[str, str]]:
    s01 = _load_stage_summary(project_root, "stage_01", "standard")
    s03 = _load_stage_summary(project_root, "stage_03", "standard")
    s04 = _load_stage_summary(project_root, "stage_04", "standard")
    s05 = _load_stage_summary(project_root, "stage_05", "standard")
    s06 = _load_stage_summary(project_root, "stage_06", "standard")
    claims = []
    missing = lambda d: not bool(d)

    c1_sup = (s04.get("hdr_vs_pooled_gain_nominal", -1) >= 0.10 and s03.get("axes_rmse_below_0_9", 0) >= 6 and s03.get("observer_mode_f1", 0) >= 0.65)
    c1_part = (s04.get("hdr_vs_pooled_gain_nominal", -1) > 0 and (s03.get("axes_rmse_below_0_9", 0) >= 4 or s03.get("observer_mode_f1", 0) >= 0.5))
    claims.append({
        "claim": "non-oracle state/mode inference is adequate for control",
        "label": _claim(c1_sup, c1_part, missing(s03) or missing(s04)),
        "evidence": f"gain={s04.get('hdr_vs_pooled_gain_nominal', 'na')}, axes<0.9={s03.get('axes_rmse_below_0_9', 'na')}, F1={s03.get('observer_mode_f1', 'na')}",
    })

    c2_sup = (s04.get("hdr_vs_open_loop_gain_nominal", -1) >= 0.10 and s04.get("hdr_vs_pooled_gain_nominal", -1) >= 0.10 and s04.get("safety_delta_vs_pooled_nominal", 999) <= 0.005 and s04.get("burden_adherence_hdr_nominal", 0) >= 0.95)
    c2_part = (s04.get("hdr_vs_open_loop_gain_nominal", -1) > 0 and s04.get("safety_delta_vs_pooled_nominal", 999) <= 0.015)
    claims.append({
        "claim": "Mode A improves over simple baselines without increasing safety violations",
        "label": _claim(c2_sup, c2_part, missing(s04)),
        "evidence": f"gain_open={s04.get('hdr_vs_open_loop_gain_nominal', 'na')}, gain_pooled={s04.get('hdr_vs_pooled_gain_nominal', 'na')}, safety_delta={s04.get('safety_delta_vs_pooled_nominal', 'na')}",
    })

    c3_sup = s01.get("tau_rank_corr", -1) >= 0.7
    c3_part = s01.get("tau_rank_corr", -1) >= 0.45
    claims.append({
        "claim": "tau_tilde tracks or ranks true recovery burden sufficiently",
        "label": _claim(c3_sup, c3_part, missing(s01)),
        "evidence": f"spearman={s01.get('tau_rank_corr', 'na')}",
    })

    c4_sup = s04.get("gaussian_calibration_abs_error", 999) <= 0.015
    c4_part = s04.get("gaussian_calibration_abs_error", 999) <= 0.03
    claims.append({
        "claim": "chance-constraint tightening is empirically calibrated in Gaussian settings",
        "label": _claim(c4_sup, c4_part, missing(s04)),
        "evidence": f"abs_error={s04.get('gaussian_calibration_abs_error', 'na')}, heavy_tail_degradation={s04.get('heavy_tail_calibration_degradation', 'na')}",
    })

    mode_slope = float(s04.get("mode_error_fit_slope", -1))
    c5_sup = mode_slope >= 0.01 and s04.get("mode_error_fit_r2", -1) >= 0.75
    c5_part = mode_slope >= 0.005 and s04.get("mode_error_fit_r2", -1) >= 0.45
    claims.append({
        "claim": "practical stability under mode error is numerically consistent with sqrt(mu)-type degradation",
        "label": _claim(c5_sup, c5_part, missing(s04)),
        "evidence": f"slope={s04.get('mode_error_fit_slope', 'na')}, R2={s04.get('mode_error_fit_r2', 'na')}",
    })

    c6_sup = s04.get("target_drift_fit_slope", -1) > 0 and s04.get("target_drift_fit_r2", -1) >= 0.75
    c6_part = s04.get("target_drift_fit_slope", -1) > 0 and s04.get("target_drift_fit_r2", -1) >= 0.45
    claims.append({
        "claim": "practical stability under drifting S*(t) is numerically consistent with linear-in-drift degradation",
        "label": _claim(c6_sup, c6_part, missing(s04)),
        "evidence": f"slope={s04.get('target_drift_fit_slope', 'na')}, R2={s04.get('target_drift_fit_r2', 'na')}",
    })

    c7_sup = s05.get("hybrid_escape_gain", -999) >= 0.10 and s05.get("hybrid_safety_delta", 999) <= 0.01
    c7_part = s05.get("hybrid_escape_gain", -999) > 0
    claims.append({
        "claim": "Mode B heuristic improves escape versus conservative baselines",
        "label": _claim(c7_sup, c7_part, missing(s05)),
        "evidence": f"escape_gain={s05.get('hybrid_escape_gain', 'na')}, safety_delta={s05.get('hybrid_safety_delta', 'na')}",
    })

    c8_sup = s05.get("reduced_abs_gap", 999) <= 0.05 and s05.get("reduced_time_gap_h6", 999) <= 0.10 * 6
    c8_part = s05.get("reduced_abs_gap", 999) <= 0.10
    claims.append({
        "claim": "Mode B remains acceptably close to exact DP on reduced discrete problems",
        "label": _claim(c8_sup, c8_part, missing(s05)),
        "evidence": f"abs_gap={s05.get('reduced_abs_gap', 'na')}, time_gap_h6={s05.get('reduced_time_gap_h6', 'na')}",
    })

    c9_sup = bool(s06.get("standalone_in_band_zero_penalty", 0)) and bool(s06.get("standalone_monotone_outside", 0)) and s06.get("integrated_time_in_band_gain", -999) >= 0.10
    c9_part = bool(s06.get("standalone_in_band_zero_penalty", 0)) and bool(s06.get("standalone_monotone_outside", 0))
    claims.append({
        "claim": "coherence penalty behaves as designed",
        "label": _claim(c9_sup, c9_part, missing(s06)),
        "evidence": f"time_gain={s06.get('integrated_time_in_band_gain', 'na')}, effect={s06.get('coherence_help_label', 'na')}",
    })

    c10_sup = s03.get("priors_gain", -999) >= 0.10 and s03.get("dither_gain", -999) >= 0.10 and s03.get("perturbation_gain", -999) >= 0.10
    c10_part = sum(v >= 0.10 for v in [s03.get("priors_gain", -999), s03.get("dither_gain", -999), s03.get("perturbation_gain", -999)]) >= 1
    claims.append({
        "claim": "identifiability improves with perturbations, priors, and dither",
        "label": _claim(c10_sup, c10_part, missing(s03)),
        "evidence": f"priors={s03.get('priors_gain', 'na')}, dither={s03.get('dither_gain', 'na')}, perturb={s03.get('perturbation_gain', 'na')}",
    })
    return claims


def write_claim_matrix(project_root: Path) -> None:
    claims = build_claim_matrix(project_root)
    body = "# Claim matrix\n\n" + dict_to_markdown_table(claims) + "\n"
    atomic_write_text(project_root / "docs" / "CLAIM_MATRIX.md", body)


def write_validation_report(project_root: Path) -> None:
    manifest = load_json(project_root / "results" / "run_manifest.json", default={"entries": []})
    claims = build_claim_matrix(project_root)
    entries = manifest.get("entries", [])
    stage_rows = [
        {"stage": e["stage"], "profile": e["profile"], "status": e["status"]}
        for e in entries
    ]
    s03 = _load_stage_summary(project_root, "stage_03", "standard")
    s04 = _load_stage_summary(project_root, "stage_04", "standard")
    s05 = _load_stage_summary(project_root, "stage_05", "standard")
    s06 = _load_stage_summary(project_root, "stage_06", "standard")
    s07 = _load_stage_summary(project_root, "stage_07", "standard")
    report = f"""# Validation report

## Completion

{dict_to_markdown_table(stage_rows) if stage_rows else "_No stage rows found._"}

## Headline findings

- Standard-profile non-oracle observer performance: state RMSE = {s03.get('observer_state_rmse', 'na')}, mode F1 = {s03.get('observer_mode_f1', 'na')}, axes below 0.9 RMSE = {s03.get('axes_rmse_below_0_9', 'na')}.
- Standard-profile Mode A nominal gain vs open-loop = {s04.get('hdr_vs_open_loop_gain_nominal', 'na')}, vs pooled LQR = {s04.get('hdr_vs_pooled_gain_nominal', 'na')}, safety delta vs pooled = {s04.get('safety_delta_vs_pooled_nominal', 'na')}.
- Mode B reduced-chain gap = {s05.get('reduced_abs_gap', 'na')}, hybrid escape gain = {s05.get('hybrid_escape_gain', 'na')}, hybrid safety delta = {s05.get('hybrid_safety_delta', 'na')}.
- Coherence integrated time-in-band gain = {s06.get('integrated_time_in_band_gain', 'na')} with qualitative effect `{s06.get('coherence_help_label', 'na')}`.
- Robustness negative-control oracle optimism gap = {s07.get('oracle_optimism_gap', 'na')}.

## Approximations and skips

- Online projection and controller use conservative box-surrogate target geometry because `cvxpy/osqp` are unavailable.
- Delay-LMI screen is marked as skipped because no suitable LMI solver is available in the environment.
- Optional variational SLDS/HSMM path is skipped by design and does not block the required backbone.

## Claim summary

{dict_to_markdown_table(claims)}
"""
    atomic_write_text(project_root / "docs" / "VALIDATION_REPORT.md", report)


def write_results_index(project_root: Path) -> None:
    deliverables = sorted((project_root / "deliverables").rglob("*.zip"))
    rows = [{"zip": str(p.relative_to(project_root))} for p in deliverables]
    stage_paths = []
    for stage_dir in sorted((project_root / "results").glob("stage_*")):
        for summary_path in sorted(stage_dir.rglob("stage_summary.json")):
            stage_paths.append({"stage_summary": str(summary_path.relative_to(project_root))})
    body = "# Results index\n\n## Deliverable zips\n\n"
    body += (dict_to_markdown_table(rows) if rows else "_No zip files found._")
    body += "\n\n## Stage summaries\n\n"
    body += (dict_to_markdown_table(stage_paths) if stage_paths else "_No stage summaries found._")
    atomic_write_text(project_root / "docs" / "RESULTS_INDEX.md", body)


def finalize_reports(project_root: Path) -> None:
    write_claim_matrix(project_root)
    write_validation_report(project_root)
    write_results_index(project_root)
