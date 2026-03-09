from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, ensure_dir, list_relative_files, load_json, timestamp


@dataclass
class ManifestEntry:
    stage: str
    profile: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    random_seeds: list[int] = field(default_factory=list)
    config_hash: str | None = None
    output_files: list[str] = field(default_factory=list)
    notes: str = ""
    failure_traceback: str = ""


class RunManifest:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        ensure_dir(manifest_path.parent)
        self.data = load_json(manifest_path, default={"project": "hdr_validation", "entries": [], "updated_at": timestamp()})
        if "entries" not in self.data:
            self.data["entries"] = []

    def save(self) -> None:
        self.data["updated_at"] = timestamp()
        atomic_write_json(self.manifest_path, self.data)

    def find(self, stage: str, profile: str) -> dict[str, Any] | None:
        for entry in self.data["entries"]:
            if entry["stage"] == stage and entry["profile"] == profile:
                return entry
        return None

    def upsert(self, entry: ManifestEntry) -> dict[str, Any]:
        current = self.find(entry.stage, entry.profile)
        payload = {
            "stage": entry.stage,
            "profile": entry.profile,
            "status": entry.status,
            "started_at": entry.started_at,
            "finished_at": entry.finished_at,
            "random_seeds": entry.random_seeds,
            "config_hash": entry.config_hash,
            "output_files": entry.output_files,
            "notes": entry.notes,
            "failure_traceback": entry.failure_traceback,
        }
        if current is None:
            self.data["entries"].append(payload)
            current = payload
        else:
            current.update(payload)
        self.save()
        return current

    def should_skip(self, stage: str, profile: str, config_hash: str, skip_done: bool = True, force: bool = False) -> bool:
        if force:
            return False
        current = self.find(stage, profile)
        if not current:
            return False
        return skip_done and current.get("status") == "completed" and current.get("config_hash") == config_hash

    def mark_running(self, stage: str, profile: str, seeds: list[int], config_hash: str, notes: str = "") -> None:
        current = self.find(stage, profile)
        started_at = current.get("started_at") if current and current.get("status") == "running" else timestamp()
        self.upsert(
            ManifestEntry(
                stage=stage,
                profile=profile,
                status="running",
                started_at=started_at,
                finished_at=None,
                random_seeds=seeds,
                config_hash=config_hash,
                output_files=current.get("output_files", []) if current else [],
                notes=notes,
            )
        )

    def mark_completed(self, stage: str, profile: str, seeds: list[int], config_hash: str, output_root: Path, notes: str = "") -> None:
        self.upsert(
            ManifestEntry(
                stage=stage,
                profile=profile,
                status="completed",
                started_at=(self.find(stage, profile) or {}).get("started_at"),
                finished_at=timestamp(),
                random_seeds=seeds,
                config_hash=config_hash,
                output_files=list_relative_files(output_root),
                notes=notes,
            )
        )

    def mark_skipped(self, stage: str, profile: str, seeds: list[int], config_hash: str, notes: str = "") -> None:
        self.upsert(
            ManifestEntry(
                stage=stage,
                profile=profile,
                status="skipped",
                started_at=(self.find(stage, profile) or {}).get("started_at"),
                finished_at=timestamp(),
                random_seeds=seeds,
                config_hash=config_hash,
                output_files=(self.find(stage, profile) or {}).get("output_files", []),
                notes=notes,
            )
        )

    def mark_failed(self, stage: str, profile: str, seeds: list[int], config_hash: str, output_root: Path, exc: BaseException, notes: str = "") -> str:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.upsert(
            ManifestEntry(
                stage=stage,
                profile=profile,
                status="failed",
                started_at=(self.find(stage, profile) or {}).get("started_at"),
                finished_at=timestamp(),
                random_seeds=seeds,
                config_hash=config_hash,
                output_files=list_relative_files(output_root),
                notes=notes,
                failure_traceback=tb,
            )
        )
        return tb
