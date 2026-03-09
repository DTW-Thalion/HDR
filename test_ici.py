from pathlib import Path

from hdr_validation.packaging import zip_paths


def test_packaging_creates_zip(tmp_path: Path):
    src = tmp_path / "srcdir"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    zip_path = tmp_path / "out.zip"
    zip_paths(zip_path, [(src, "srcdir")])
    assert zip_path.exists()
