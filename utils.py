from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.rename(tmp_name, path)
    except Exception:
        os.unlink(tmp_name)
        raise
    return path