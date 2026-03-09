"""
HDR Validation Suite — Report Generator
========================================
Runs smoke, standard, extended, and validation profiles, captures all check
results, and writes structured reports to reports/:

  reports/
    smoke_results.json
    standard_results.json
    extended_results.json
    validation_results.json
    all_results.csv
    summary.md

Usage:
    python3 generate_reports.py [--profiles smoke standard extended validation]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"

RUNNERS = {
    "smoke":      ROOT / "smoke_runner.py",
    "standard":   ROOT / "standard_runner.py",
    "extended":   ROOT / "extended_runner.py",
    "validation": ROOT / "validation_runner.py",
}

PROFILE_META = {
    "smoke":      {"seeds": [101],          "episodes": 8,  "steps": 128, "mc": 50},
    "standard":   {"seeds": [101, 202],     "episodes": 12, "steps": 128, "mc": 100},
    "extended":   {"seeds": [101, 202, 303],"episodes": 20, "steps": 256, "mc": 150},
    "validation": {"seeds": [101, 202, 303],"episodes": 12, "steps": 128, "mc": 150},
}

# ── Parsing ────────────────────────────────────────────────────────────────────

_CHECK_RE = re.compile(
    r"\s+\[(PASS|FAIL)\]\s+(.+?)(?:\s+=\s+(.+))?$"
)
_STAGE_RE  = re.compile(r"Stage\s+([\w]+)\s+[—–-]")
_ELAPSED_RE = re.compile(r"Elapsed:\s+([\d.]+)s")


def parse_output(stdout: str) -> list[dict]:
    """Parse runner stdout into a list of check dicts."""
    records: list[dict] = []
    current_stage = "unknown"
    for line in stdout.splitlines():
        m_stage = _STAGE_RE.search(line)
        if m_stage:
            current_stage = "stage" + m_stage.group(1).lower().replace(" ", "")
            continue
        m_check = _CHECK_RE.match(line)
        if m_check:
            status, check, value = m_check.groups()
            records.append({
                "stage":  current_stage,
                "check":  check.strip(),
                "passed": status == "PASS",
                "value":  (value or "").strip(),
            })
    return records


# ── Runner ────────────────────────────────────────────────────────────────────

def run_profile(profile: str) -> tuple[list[dict], float, int]:
    """Run a profile runner and return (records, elapsed_seconds, returncode)."""
    runner = RUNNERS[profile]
    if not runner.exists():
        print(f"  [SKIP] {runner.name} not found")
        return [], 0.0, -1

    print(f"  Running {profile} profile ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    records = parse_output(proc.stdout)
    print(f"  Done in {elapsed:.1f}s — {sum(r['passed'] for r in records)}/"
          f"{len(records)} checks passed")
    return records, elapsed, proc.returncode


# ── Writers ───────────────────────────────────────────────────────────────────

def write_profile_json(profile: str, records: list[dict], elapsed: float,
                       returncode: int, out_dir: Path) -> Path:
    meta = PROFILE_META.get(profile, {})
    by_stage: dict[str, dict] = {}
    for r in records:
        s = r["stage"]
        if s not in by_stage:
            by_stage[s] = {"passed": 0, "failed": 0, "checks": []}
        by_stage[s]["checks"].append(r)
        if r["passed"]:
            by_stage[s]["passed"] += 1
        else:
            by_stage[s]["failed"] += 1

    total_pass = sum(r["passed"] for r in records)
    total_fail = len(records) - total_pass

    doc = {
        "profile":        profile,
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "exit_code":      returncode,
        "meta":           meta,
        "summary": {
            "total":  len(records),
            "passed": total_pass,
            "failed": total_fail,
            "all_passed": total_fail == 0,
        },
        "stages": by_stage,
    }
    path = out_dir / f"{profile}_results.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def write_all_csv(all_records: dict[str, list[dict]], out_dir: Path) -> Path:
    path = out_dir / "all_results.csv"
    fieldnames = ["profile", "stage", "check", "passed", "value"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for profile, records in all_records.items():
            for r in records:
                writer.writerow({
                    "profile": profile,
                    "stage":   r["stage"],
                    "check":   r["check"],
                    "passed":  r["passed"],
                    "value":   r["value"],
                })
    return path


def write_summary_md(all_records: dict[str, list[dict]],
                     elapsed_map: dict[str, float],
                     out_dir: Path) -> Path:
    profiles = list(all_records.keys())
    lines: list[str] = []

    lines.append("# HDR Validation Suite — Results Summary")
    lines.append(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Cross-profile overview table
    lines.append("## Profile Overview\n")
    lines.append("| Profile | Seeds | Episodes | Steps | MC | Total | Passed | Failed | Status | Time |")
    lines.append("|---------|-------|----------|-------|----|-------|--------|--------|--------|------|")
    for p in profiles:
        records = all_records[p]
        meta = PROFILE_META.get(p, {})
        total  = len(records)
        passed = sum(r["passed"] for r in records)
        failed = total - passed
        status = "✓ PASS" if failed == 0 else f"✗ {failed} FAIL"
        seeds  = len(meta.get("seeds", []))
        eps    = meta.get("episodes", "?")
        steps  = meta.get("steps", "?")
        mc     = meta.get("mc", "?")
        t      = f"{elapsed_map.get(p, 0):.1f}s"
        lines.append(f"| {p} | {seeds} | {eps} | {steps} | {mc} | {total} | {passed} | {failed} | {status} | {t} |")

    # Per-stage breakdown across profiles
    lines.append("\n## Stage-by-Stage Results\n")

    # Collect all stage names in order
    stage_order: list[str] = []
    seen: set[str] = set()
    for records in all_records.values():
        for r in records:
            if r["stage"] not in seen:
                stage_order.append(r["stage"])
                seen.add(r["stage"])

    # Header row
    header = "| Stage | " + " | ".join(p.capitalize() for p in profiles) + " |"
    sep    = "|-------|" + "|".join("--------" for _ in profiles) + "|"
    lines.append(header)
    lines.append(sep)

    for stage in stage_order:
        cells = []
        for p in profiles:
            stage_checks = [r for r in all_records[p] if r["stage"] == stage]
            if not stage_checks:
                cells.append("—")
            else:
                n = len(stage_checks)
                ok = sum(r["passed"] for r in stage_checks)
                cells.append(f"{'✓' if ok == n else '✗'} {ok}/{n}")
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")

    # Key metric table
    lines.append("\n## Key Metrics by Profile\n")
    lines.append("| Metric | " + " | ".join(p.capitalize() for p in profiles) + " |")
    lines.append("|--------|" + "|".join("--------" for _ in profiles) + "|")

    METRIC_CHECKS = [
        ("tau_tilde(far) > 0",             "stage01"),
        ("committor q[A]=0",               "stage01"),
        ("committor q[B]=1",               "stage01"),
        ("DARE P positive-definite",       "stage01"),
        ("mode1 F1 > 0",                   "stage03"),
        ("Brier reliability finite",       "stage03b"),
        ("p_A_robust ∈ [0,1]",             "stage03b"),
        ("Mode A feasibility rate > 0.5",  "stage04"),
        ("Mode B aggressive > passive",    "stage05"),
        ("Exact DP V* ∈ [0,1]",            "stage05"),
        ("epsilon_H > 0",                  "stage05"),
    ]

    for label, stage in METRIC_CHECKS:
        cells = []
        for p in profiles:
            match = next(
                (r for r in all_records[p]
                 if r["stage"] == stage and label.lower() in r["check"].lower()),
                None,
            )
            if match is None:
                cells.append("—")
            else:
                v = match["value"] if match["value"] else ("✓" if match["passed"] else "✗")
                cells.append(v)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Failed checks section
    all_failed = [
        (p, r) for p, records in all_records.items()
        for r in records if not r["passed"]
    ]
    if all_failed:
        lines.append("\n## Failed Checks\n")
        for p, r in all_failed:
            lines.append(f"- **[{p}]** `{r['stage']}` — {r['check']}: `{r['value']}`")
    else:
        lines.append("\n## Failed Checks\n\n_None — all checks passed across all profiles._")

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HDR validation reports")
    parser.add_argument(
        "--profiles", nargs="+",
        default=["smoke", "standard", "extended", "validation"],
        choices=list(RUNNERS),
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reports will be written to: {REPORTS_DIR}\n")

    all_records: dict[str, list[dict]] = {}
    elapsed_map: dict[str, float] = {}

    for profile in args.profiles:
        records, elapsed, returncode = run_profile(profile)
        all_records[profile] = records
        elapsed_map[profile] = elapsed
        json_path = write_profile_json(profile, records, elapsed, returncode, REPORTS_DIR)
        print(f"  → {json_path.relative_to(ROOT)}")

    csv_path = write_all_csv(all_records, REPORTS_DIR)
    md_path  = write_summary_md(all_records, elapsed_map, REPORTS_DIR)
    print(f"\n  → {csv_path.relative_to(ROOT)}")
    print(f"  → {md_path.relative_to(ROOT)}")

    total_pass = sum(sum(r["passed"] for r in recs) for recs in all_records.values())
    total_all  = sum(len(recs) for recs in all_records.values())
    total_fail = total_all - total_pass

    print(f"\nTotal across all profiles: {total_pass}/{total_all} passed", end="")
    if total_fail:
        print(f", {total_fail} FAILED")
        sys.exit(1)
    else:
        print(" — ALL PASSED")


if __name__ == "__main__":
    main()
