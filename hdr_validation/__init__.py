"""HDR Validation Suite — top-level package."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("hdr_validation")
except PackageNotFoundError:
    __version__ = "7.3.0"

from .defaults import HDR_VERSION as _HDR_VERSION
import warnings as _warnings
if __version__ != _HDR_VERSION:
    _warnings.warn(
        f"hdr_validation installed version ({__version__}) != "
        f"defaults.HDR_VERSION ({_HDR_VERSION}). "
        f"Run 'pip install -e .' to sync.",
        stacklevel=2,
    )

from . import control, inference, model  # noqa: F401

try:
    from . import identification  # noqa: F401 — v7.0 identification subpackage
except ImportError:
    pass
