"""Tests for Stage 18 — Partially Observed Closed-Loop ICI Benchmark.

Tests run with reduced parameters (n_seeds=2, n_ep=3, T=64, fast_mode=True)
for speed. Production-scale validation uses 20 seeds x 30 episodes x 256 steps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_18_closed_loop_ici import (
    CONDITIONS,
    run_stage_18,
)

ROOT = Path(__file__).parent


@pytest.fixture(scope="module")
def stage_result() -> dict:
    return run_stage_18(n_seeds=2, n_ep=3, T=64, fast_mode=True)


def test_18_01_runs_without_error(stage_result: dict) -> None:
    """18.01: Stage runs and returns a valid result dict."""
    assert stage_result is not None
    assert "checks" in stage_result
    assert len(stage_result["checks"]) == 7


def test_18_02_all_conditions_present(stage_result: dict) -> None:
    """18.02: All 4 conditions present in summary."""
    for cond in CONDITIONS:
        assert cond in stage_result["summary_all"], f"Missing condition: {cond}"


def test_18_03_costs_positive_finite(stage_result: dict) -> None:
    """18.03: Costs are positive and finite for all conditions."""
    for cond in CONDITIONS:
        cost = stage_result["summary_all"][cond]["mean_cost"]
        assert np.isfinite(cost), f"{cond}: cost not finite ({cost})"
        assert cost > 0, f"{cond}: cost not positive ({cost})"


def test_18_04_ici_triggers_recorded(stage_result: dict) -> None:
    """18.04: ICI trigger rate is recorded and non-negative."""
    rate = stage_result["summary_all"].get("ici_trigger_rate", -1)
    assert rate >= 0, f"ICI trigger rate not recorded or negative: {rate}"


def test_18_05_oracle_cost_bounded(stage_result: dict) -> None:
    """18.05: Oracle cost is finite and bounded."""
    oracle = stage_result["summary_all"]["oracle_hdr"]["mean_cost"]
    assert np.isfinite(oracle), f"Oracle cost not finite: {oracle}"
    assert oracle < 1e6, f"Oracle cost too large: {oracle}"


def test_18_06_mode_error_rate_valid(stage_result: dict) -> None:
    """18.06: Mode error rates are in [0, 1]."""
    for cond in ["hdr_ici", "hdr_no_ici"]:
        rate = stage_result["summary_all"][cond]["mode_error_rate"]
        assert 0 <= rate <= 1, f"{cond}: mode error rate out of range: {rate}"


def test_18_07_output_json_saved(stage_result: dict) -> None:
    """18.07: Results JSON is saved and loadable."""
    path = ROOT / "results" / "stage_18" / "stage_18_results.json"
    assert path.exists(), f"Results file not found: {path}"
    with open(path) as f:
        data = json.load(f)
    assert "checks" in data
    assert "provenance" in data


def test_18_08_bootstrap_cis_finite(stage_result: dict) -> None:
    """18.08: Bootstrap CIs are finite with lo <= hi."""
    for cond in CONDITIONS:
        if cond == "pooled_lqr":
            continue
        entry = stage_result["summary_all"][cond]
        lo = entry["gain_ci_lo"]
        hi = entry["gain_ci_hi"]
        assert np.isfinite(lo), f"{cond}: CI lo not finite"
        assert np.isfinite(hi), f"{cond}: CI hi not finite"
        assert lo <= hi + 1e-6, f"{cond}: CI lo ({lo}) > hi ({hi})"


def test_18_09_basin_breakdown_present(stage_result: dict) -> None:
    """18.09: Basin-conditioned breakdown has maladaptive and healthy."""
    assert "summary_maladaptive" in stage_result
    assert "summary_healthy" in stage_result
    # At least maladaptive should have episodes (due to forced_mal guard)
    assert stage_result["episode_counts"]["maladaptive"] > 0


def test_18_10_no_divergence(stage_result: dict) -> None:
    """18.10: No condition diverges (max state norm < 1e6)."""
    checks = {c["check"]: c for c in stage_result["checks"]}
    assert checks["no_divergence"]["passed"], (
        f"Divergence detected: {checks['no_divergence']['value']}"
    )
