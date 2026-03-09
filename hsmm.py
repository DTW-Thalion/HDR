from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, cls=NumpyJSONEncoder).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=True, cls=NumpyJSONEncoder))


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_save_npz(path: Path | str, **arrays: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_csv(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_dataframe_csv(path: Path | str, df: pd.DataFrame) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))


def load_json(path: Path | str, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def list_relative_files(root: Path | str) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def batched(seq: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


def setup_logger(log_path: Path | str, name: str) -> logging.Logger:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def environment_snapshot(package_presence: dict[str, bool]) -> dict[str, Any]:
    return {
        "timestamp": timestamp(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_presence": package_presence,
    }


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    vals = (c[window:] - c[:-window]) / float(window)
    pad = np.full(window - 1, vals[0] if len(vals) else np.nan)
    return np.concatenate([pad, vals]) if len(vals) else np.full_like(x, np.nan, dtype=float)


def robust_mean(x: Sequence[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return float("nan")
    lo, hi = np.quantile(arr, [0.1, 0.9])
    clipped = np.clip(arr, lo, hi)
    return float(np.mean(clipped))


def safe_inv(mat: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    eye = np.eye(mat.shape[0])
    return np.linalg.inv(mat + ridge * eye)


def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm <= 1e-12:
        return vec
    return vec * (max_norm / norm)


def linspace_int(start: int, stop: int, num: int) -> list[int]:
    vals = np.linspace(start, stop, num=num)
    return [int(round(v)) for v in vals]


def append_markdown_section(path: Path | str, title: str, body: str) -> None:
    path = Path(path)
    existing = path.read_text() if path.exists() else ""
    text = existing.rstrip() + f"\n\n## {title}\n\n{body.strip()}\n"
    atomic_write_text(path, text)


def save_text_log(path: Path | str, message: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    existing = path.read_text() if path.exists() else ""
    atomic_write_text(path, existing + message)


def dict_to_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)
