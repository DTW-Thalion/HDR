"""HDR Validation Suite — top-level package."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("hdr_validation")
except PackageNotFoundError:
    __version__ = "7.3.0"

from . import control, inference, model  # noqa: F401

try:
    from . import identification  # noqa: F401 — v7.0 identification subpackage
except ImportError:
    pass
