"""Tests for Stage 16 — Model-Failure Extension Integration Validation."""
from __future__ import annotations

import numpy as np
import pytest

from hdr_validation.stages.stage_16_extensions import (
    _make_stage16_config,
    _run_subtest_16_01_pwa,
    _run_subtest_16_05_adaptive,
    _run_subtest_16_12_baseline,
    run_stage_16,
)


def _fast_cfg(n_seeds=2, T=32):
    return _make_stage16_config(n_seeds=n_seeds, T=T)


def test_stage16_pwa_numerical_stability():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_01_pwa(cfg, n_seeds=2, T=32)
    assert result["numerical_stability"] is True


def test_stage16_pwa_region_consistency():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_01_pwa(cfg, n_seeds=2, T=32)
    assert result["region_consistency_rate"] >= 0.90


def test_stage16_adaptive_drift_tracked():
    cfg = _fast_cfg(n_seeds=2, T=64)
    result = _run_subtest_16_05_adaptive(cfg, n_seeds=2, T=64)
    assert result["drift_tracked_rate"] >= 0.50


def test_stage16_baseline_equivalence():
    cfg = _fast_cfg(n_seeds=2, T=32)
    result = _run_subtest_16_12_baseline(cfg, n_seeds=2, T=32)
    assert result["backward_compatible"] is True


def test_stage16_stub_returns_not_implemented():
    result = run_stage_16(n_seeds=1, T=16, fast_mode=True,
                          subtests=["16.02"])
    assert result["16.02"]["status"] == "NOT_IMPLEMENTED"


def test_stage16_full_run_no_crash():
    result = run_stage_16(n_seeds=1, T=16, fast_mode=True)
    assert all(r.get("pass", True) or r.get("status") == "NOT_IMPLEMENTED"
               for r in result.values())
