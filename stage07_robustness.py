from __future__ import annotations

from pathlib import Path

from ..packaging import update_stage_archives
from ..specification import initialize_docs, write_environment_report
from ..utils import atomic_write_json, ensure_dir, timestamp


def run(project_root: Path, profile_name: str, config: dict) -> dict:
    stage_root = ensure_dir(project_root / "results" / "stage_00" / profile_name)
    using_paper = True
    initialize_docs(project_root, using_paper=using_paper)
    env_report = write_environment_report(project_root)
    summary = {
        "stage": "stage_00",
        "profile": profile_name,
        "timestamp": timestamp(),
        "using_paper": using_paper,
        "environment_report": str(env_report.relative_to(project_root)),
        "notes": "Docs initialized from attached paper; stage/profile packaging refreshed.",
    }
    atomic_write_json(stage_root / "summary.json", summary)
    atomic_write_json(stage_root / "config.json", config)
    update_stage_archives(project_root, "stage_00")
    return summary
