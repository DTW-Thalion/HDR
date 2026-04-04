"""Tests for Stage 18b — Sensor-degradation sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdr_validation.stages.stage_18_closed_loop_ici import run_stage_18b

ROOT = Path(__file__).parent


@pytest.fixture(scope="module")
def sweep_result() -> dict:
    return run_stage_18b(n_seeds=2, n_ep=4, T=64, fast_mode=True)


def test_18b_01_runs_without_error(sweep_result: dict) -> None:
    """18b.01: Sweep runs and returns valid result."""
    assert sweep_result is not None
    assert "checks" in sweep_result
    assert len(sweep_result["checks"]) == 5


def test_18b_02_sigma_sweep_present(sweep_result: dict) -> None:
    """18b.02: Sigma sweep has expected number of points."""
    assert "sigma_sweep" in sweep_result
    assert len(sweep_result["sigma_sweep"]) >= 5


def test_18b_03_pdrop_sweep_present(sweep_result: dict) -> None:
    """18b.03: Dropout sweep has expected number of points."""
    assert "pdrop_sweep" in sweep_result
    assert len(sweep_result["pdrop_sweep"]) >= 3


def test_18b_04_ici_trigger_rates_nonnegative(sweep_result: dict) -> None:
    """18b.04: All ICI trigger rates are non-negative."""
    for pt in sweep_result["sigma_sweep"]:
        assert pt["ici_trigger_pct"] >= 0, f"Negative trigger at sigma={pt['sigma_proxy']}"
    for pt in sweep_result["pdrop_sweep"]:
        assert pt["ici_trigger_pct"] >= 0, f"Negative trigger at p_drop={pt['p_drop']}"


def test_18b_05_mode_errors_bounded(sweep_result: dict) -> None:
    """18b.05: Mode error rates are in [0, 100]%."""
    for pt in sweep_result["sigma_sweep"] + sweep_result["pdrop_sweep"]:
        assert 0 <= pt["mode_error_pct"] <= 100


def test_18b_06_output_saved(sweep_result: dict) -> None:
    """18b.06: Results JSON saved and loadable."""
    path = ROOT / "results" / "stage_18b" / "sweep_results.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "sigma_sweep" in data


def test_18b_07_provenance_present(sweep_result: dict) -> None:
    """18b.07: Provenance metadata present."""
    assert "provenance" in sweep_result
    assert "hdr_version" in sweep_result["provenance"]
