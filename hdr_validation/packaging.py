from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

from .utils import atomic_write_text, ensure_dir


def zip_paths(zip_path: Path | str, include_roots: list[tuple[Path, str]]) -> Path:
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)
    zip_real = zip_path.resolve()
    fd, tmp_name = tempfile.mkstemp(prefix=zip_path.name + ".", suffix=".tmp", dir=zip_path.parent)
    os.close(fd)
    tmp_real = Path(tmp_name).resolve()
    try:
        with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, arc_prefix in include_roots:
                root = Path(root)
                if root.is_file():
                    root_real = root.resolve()
                    if root_real == zip_real or root_real == tmp_real or root.name.endswith(".tmp"):
                        continue
                    zf.write(root, arcname=str(Path(arc_prefix) / root.name))
                    continue
                if not root.exists():
                    continue
                for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
                    file_real = file_path.resolve()
                    if file_real == zip_real or file_real == tmp_real:
                        continue
                    if file_path.name.endswith(".tmp") or file_path.suffix == ".tmp":
                        continue
                    arcname = str(Path(arc_prefix) / file_path.relative_to(root))
                    zf.write(file_path, arcname=arcname)
        os.replace(tmp_name, zip_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return zip_path


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_stage_archives(project_root: Path, stage_name: str) -> list[Path]:
    deliverables = ensure_dir(project_root / "deliverables" / "stages")
    stage_results_root = project_root / "results" / stage_name
    archives = []
    archives.append(zip_paths(deliverables / f"{stage_name}_source.zip", [
        (project_root / "src", "src"),
        (project_root / "tests", "tests"),
        (project_root / "configs", "configs"),
        (project_root / "run_all.py", ""),
        (project_root / "README.md", ""),
        (project_root / "pyproject.toml", ""),
    ]))
    archives.append(zip_paths(deliverables / f"{stage_name}_docs.zip", [(project_root / "docs", "docs")]))
    archives.append(zip_paths(deliverables / f"{stage_name}_results.zip", [
        (stage_results_root, f"results/{stage_name}"),
        (project_root / "results" / "run_manifest.json", "results"),
        (project_root / "results" / "logs", "results/logs"),
    ]))
    return archives


def update_final_archives(project_root: Path) -> list[Path]:
    deliverables = ensure_dir(project_root / "deliverables")
    archives = []
    archives.append(zip_paths(deliverables / "hdr_validation_python_source.zip", [
        (project_root / "src", "src"),
        (project_root / "tests", "tests"),
        (project_root / "configs", "configs"),
        (project_root / "run_all.py", ""),
        (project_root / "README.md", ""),
        (project_root / "pyproject.toml", ""),
    ]))
    archives.append(zip_paths(deliverables / "hdr_validation_docs.zip", [(project_root / "docs", "docs")]))
    archives.append(zip_paths(deliverables / "hdr_validation_results.zip", [
        (project_root / "results", "results"),
    ]))
    archives.append(zip_paths(deliverables / "all_deliverables_and_stage_zips.zip", [
        (project_root / "deliverables", "deliverables"),
    ]))
    return archives


def write_checksums(project_root: Path) -> Path:
    deliverables = project_root / "deliverables"
    rows = []
    for file_path in sorted(p for p in deliverables.rglob("*.zip") if p.is_file()):
        rows.append(f"{sha256_file(file_path)}  {file_path.relative_to(project_root)}")
    out = deliverables / "checksums_sha256.txt"
    atomic_write_text(out, "\n".join(rows) + ("\n" if rows else ""))
    return out
