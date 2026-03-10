"""HDR Validation Suite — top-level package."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("hdr_validation")
except PackageNotFoundError:
    __version__ = "5.0.0-dev"

from . import control, inference, model  # noqa: F401
