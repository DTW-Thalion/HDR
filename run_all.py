"""
HDR Validation Suite — Orchestration Script
============================================
Runs any combination of profiles and stages by importing stage functions
directly from the per-profile runner modules.

Usage:
    python run_all.py                              # all profiles, all stages
    python run_all.py --profiles smoke             # smoke only
    python run_all.py --profiles smoke standard extended validation
    python run_all.py --profiles standard --stages 04 --force
    python run_all.py --stages 01 03b 03c          # selected stages, all profiles
    python run_all.py --resume --skip-done         # skip already-completed stages
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "run_all_manifest.json"

# ── Stage metadata ─────────────────────────────────────────────────────────────

STAGE_SEQUENCE = ["01", "02", "03", "03b", "03c", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]

STAGE_LABELS = {
    "01":  "Stage 01 — Mathematical Checks",
    "02":  "Stage 02 — Synthetic Data Generation",
    "03":  "Stage 03 — IMM Inference",
    "03b": "Stage 03b — ICI Diagnostics",
    "03c": "Stage 03c — Mode C Validation",
    "04":  "Stage 04 — Mode A Control",
    "05":  "Stage 05 — Mode B Validation",
    "06":  "Stage 06 — State Coherence",
    "07":  "Stage 07 — Robustness Sweeps",
    "08":  "Stage 08 — Ablation Study",
    "09":  "Stage 09 — Baseline Comparison",
    "10":  "Stage 10 — Mode B FP/FN Sweep",
    "11":  "Stage 11 — Riccati Invariant Set",
    "12":  "Stage 12 — Hierarchical Coupling",
    "13":  "Stage 13 — Inference Backbone",
    "14":  "Stage 14 — Population Planning",
    "15":  "Stage 15 — Proxy Composite",
}

# Stages that must run before a given stage (dependency order)
STAGE_DEPS: dict[str, list[str]] = {
    "03":  ["02"],
    "03b": ["02", "03"],
    "04":  ["02"],
}

PROFILE_MODULES = {
    "smoke":      "smoke_runner",
    "standard":   "standard_runner",
    "extended":   "extended_runner",
    "validation": "validation_runner",
}

# ── Manifest (checkpoint) ──────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def manifest_key(profile: str, stage: str) -> str:
    return f"{profile}:{stage}"


def is_done(manifest: dict, profile: str, stage: str) -> bool:
    return manifest.get(manifest_key(profile, stage)) == "done"


def mark_done(manifest: dict, profile: str, stage: str) -> None:
    manifest[manifest_key(profile, stage)] = "done"


def mark_failed(manifest: dict, profile: str, stage: str) -> None:
    manifest[manifest_key(profile, stage)] = "failed"

# ── Dependency resolution ──────────────────────────────────────────────────────

def resolve_with_deps(requested: list[str]) -> list[str]:
    """Return ordered list of stages to run, including required dependencies."""
    needed: set[str] = set(requested)
    for s in requested:
        for dep in STAGE_DEPS.get(s, []):
            needed.add(dep)
    return [s for s in STAGE_SEQUENCE if s in needed]


# ── Stage dispatch ─────────────────────────────────────────────────────────────

def _call_stage_08(fast: bool = False) -> None:
    """Run Stage 08 ablation study."""
    from hdr_validation.stages.stage_08_ablation import run_stage_08
    n_seeds = 2 if fast else 20
    n_ep = 3 if fast else 30
    T = 32 if fast else 256
    run_stage_08(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=False)


def _call_stage_09(fast: bool = False) -> None:
    """Run Stage 09 baseline comparison."""
    from hdr_validation.stages.stage_09_baselines import run_stage_09
    n_seeds = 2 if fast else 20
    n_ep = 3 if fast else 30
    T = 32 if fast else 256
    run_stage_09(n_seeds=n_seeds, n_ep=n_ep, T=T, fast_mode=False)


def _call_stage_10(fast: bool = False) -> None:
    """Run Stage 10 Mode B FP/FN sweep."""
    from hdr_validation.stages.stage_10_mode_b_sweep import run_stage_10
    N_sim = 200 if fast else 5000
    run_stage_10(N_sim=N_sim, T=50, fast_mode=False)


def _call_stage_12(fast: bool = False) -> None:
    """Run Stage 12 hierarchical coupling estimation."""
    from hdr_validation.stages.stage_12_hierarchical import run_stage_12
    n_patients = 5 if fast else 10
    T_p_values = [0, 10, 50] if fast else [0, 10, 50, 200]
    run_stage_12(n_patients=n_patients, T_p_values=T_p_values, fast_mode=fast)


def _call_stage_13(fast: bool = False) -> None:
    """Run Stage 13 inference backbone benchmark."""
    from hdr_validation.stages.stage_13_inference_backbone import run_stage_13
    n_particles = 50 if fast else 100
    n_scenarios = 3 if fast else 5
    run_stage_13(n_particles=n_particles, n_scenarios=n_scenarios, fast_mode=fast)


def _call_stage_14(fast: bool = False) -> None:
    """Run Stage 14 population planning benchmark."""
    from hdr_validation.stages.stage_14_population_planning import run_stage_14
    n_patients = 10 if fast else 20
    run_stage_14(n_patients=n_patients, fast_mode=fast)


def _call_stage_15(fast: bool = False) -> None:
    """Run Stage 15 proxy composite benchmark."""
    from hdr_validation.stages.stage_15_proxy_composite import run_stage_15
    n_scenarios = 3 if fast else 5
    run_stage_15(n_scenarios=n_scenarios, fast_mode=fast)


def _call_stage_11(fast: bool = False) -> None:
    """Run Stage 11 Riccati invariant set verification."""
    from hdr_validation.stages.stage_11_invariant_set import run_stage_11
    n_seeds = 2 if fast else 5
    T = 32 if fast else 128
    run_stage_11(n_seeds=n_seeds, T=T, fast_mode=False)


def call_stage(mod: Any, stage_id: str, state: dict) -> None:
    """Call the appropriate stage function from the runner module."""
    fast = state.get("fast_mode", False)
    if stage_id == "01":
        mod.stage01_math()
    elif stage_id == "02":
        state["episodes"] = mod.stage02_generation()
    elif stage_id == "03":
        state["stage03_data"] = mod.stage03_imm(state["episodes"])
    elif stage_id == "03b":
        mod.stage03b_ici(state["stage03_data"])
    elif stage_id == "03c":
        mod.stage03c_mode_c()
    elif stage_id == "04":
        mod.stage04_mode_a(state["episodes"])
    elif stage_id == "05":
        mod.stage05_mode_b()
    elif stage_id == "06":
        mod.stage06_coherence()
    elif stage_id == "07":
        mod.stage07_robustness()
    elif stage_id == "08":
        _call_stage_08(fast=fast)
    elif stage_id == "09":
        _call_stage_09(fast=fast)
    elif stage_id == "10":
        _call_stage_10(fast=fast)
    elif stage_id == "11":
        _call_stage_11(fast=fast)
    elif stage_id == "12":
        _call_stage_12(fast=fast)
    elif stage_id == "13":
        _call_stage_13(fast=fast)
    elif stage_id == "14":
        _call_stage_14(fast=fast)
    elif stage_id == "15":
        _call_stage_15(fast=fast)
    else:
        raise ValueError(f"Unknown stage: {stage_id!r}")


# ── Profile runner ─────────────────────────────────────────────────────────────

def run_profile(
    profile: str,
    stages_to_run: list[str],
    force: bool,
    skip_done: bool,
    manifest: dict,
    fast: bool = False,
) -> dict[str, list[dict]]:
    """
    Run the requested stages for a profile.

    Returns a mapping of stage_id -> list of check records.
    Dependency stages that aren't in stages_to_run are run silently
    (their checks are collected but marked as dependency-only).
    """
    module_name = PROFILE_MODULES[profile]
    mod = importlib.import_module(module_name)

    # Reset the module's results list for a clean run
    mod.results.clear()

    # Expand with dependencies; track which stages are "user-requested"
    full_sequence = resolve_with_deps(stages_to_run)
    requested_set = set(stages_to_run)
    dep_only_set = set(full_sequence) - requested_set

    print(f"\n{'#'*60}")
    print(f"  Profile: {profile.upper()}")
    print(f"  Stages:  {', '.join(full_sequence)}")
    print(f"{'#'*60}")

    # Stages 08-11 run with fast_mode=True for smoke profile or --fast flag
    fast_mode = fast or profile in ("smoke",)
    state: dict[str, Any] = {"episodes": None, "stage03_data": None, "fast_mode": fast_mode}
    stage_results: dict[str, list[dict]] = {}

    # Stages that are profile-independent (have their own simulation logic)
    INDEPENDENT_STAGES = {"08", "09", "10", "11", "12", "13", "14", "15"}

    for stage_id in full_sequence:
        label = STAGE_LABELS[stage_id]
        is_dep = stage_id in dep_only_set

        if skip_done and not force and is_done(manifest, profile, stage_id):
            print(f"\n  [SKIP] {label} (already done)")
            continue

        print(f"\n{'='*60}")
        if is_dep:
            print(f"  {label}  [dependency]")
        else:
            print(f"  {label}")
        print(f"{'='*60}")

        # Snapshot results count before stage to isolate this stage's records
        idx_before = len(mod.results)
        t0 = time.perf_counter()
        stage_exception = None

        try:
            call_stage(mod, stage_id, state)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.2f}s")
            mark_done(manifest, profile, stage_id)
            save_manifest(manifest)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            traceback.print_exc()
            print(f"  [ERROR] Stage {stage_id} raised an exception after {elapsed:.2f}s")
            mark_failed(manifest, profile, stage_id)
            save_manifest(manifest)
            stage_exception = exc

        # For profile-independent stages (08-11), add a synthetic result record
        # since they don't append to mod.results directly
        if stage_id in INDEPENDENT_STAGES:
            mod.results.append({
                "stage": stage_id,
                "check": "stage_execution",
                "passed": stage_exception is None,
                "value": "OK" if stage_exception is None else str(stage_exception),
                "note": f"Profile-independent stage {stage_id}",
            })

        stage_results[stage_id] = list(mod.results[idx_before:])

    return stage_results


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, dict[str, list[dict]]]) -> int:
    """Print overall summary. Returns number of failures."""
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    total_pass = total_fail = 0

    for profile, stage_map in all_results.items():
        print(f"\n  Profile: {profile}")
        for stage_id, records in stage_map.items():
            n_pass = sum(1 for r in records if r["passed"])
            n_fail = len(records) - n_pass
            status = "✓" if n_fail == 0 else "✗"
            print(f"    {status} {STAGE_LABELS[stage_id]}: {n_pass}/{len(records)} passed")
            total_pass += n_pass
            total_fail += n_fail

    print(f"\n  Total: {total_pass} passed, {total_fail} failed out of {total_pass + total_fail} checks")

    if total_fail == 0:
        print("\n  ✓ ALL CHECKS PASSED")
    else:
        print(f"\n  ✗ {total_fail} CHECKS FAILED")
        print("\n  Failed checks:")
        for profile, stage_map in all_results.items():
            for stage_id, records in stage_map.items():
                for r in records:
                    if not r["passed"]:
                        print(f"    [{profile}/{stage_id}] {r['check']}: {r['value']} ({r['note']})")

    return total_fail


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDR Validation Suite — run all stages across profiles"
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=list(PROFILE_MODULES),
        choices=list(PROFILE_MODULES),
        metavar="PROFILE",
        help="Profiles to run (default: all)",
    )
    parser.add_argument(
        "--stages", nargs="+",
        default=STAGE_SEQUENCE,
        metavar="STAGE",
        help="Stage IDs to run, e.g. 01 03b 04 (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if marked done in manifest",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing manifest (implies using checkpoint state)",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip stages already marked done in manifest",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run stages 08-11 with reduced parameters (n_seeds=2, n_ep=3, T=32) for smoke testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate stage IDs
    unknown = [s for s in args.stages if s not in STAGE_SEQUENCE]
    if unknown:
        print(f"[ERROR] Unknown stage(s): {unknown}")
        print(f"  Valid stages: {STAGE_SEQUENCE}")
        sys.exit(2)

    manifest = load_manifest() if (args.resume or args.skip_done) else {}

    print("HDR Validation Suite")
    print(f"  Profiles : {args.profiles}")
    print(f"  Stages   : {args.stages}")
    print(f"  Force    : {args.force}")
    print(f"  SkipDone : {args.skip_done}")

    all_results: dict[str, dict[str, list[dict]]] = {}

    for profile in args.profiles:
        all_results[profile] = run_profile(
            profile=profile,
            stages_to_run=args.stages,
            force=args.force,
            skip_done=args.skip_done,
            manifest=manifest,
            fast=args.fast,
        )

    n_fail = print_summary(all_results)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
