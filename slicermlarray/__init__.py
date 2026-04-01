"""Helpers for a scripted 3D Slicer MLArray IO extension."""

from importlib import metadata as _metadata

from .conversion import (
    ResolvedMlaLayout,
    SlicerGeometry,
    build_export_axis_labels,
    ijk_to_ras_matrix_from_geometry,
    mlarray_geometry_to_slicer,
    mla_array_to_slicer,
    resolve_supported_mla_layout,
    slicer_geometry_to_mlarray,
)
from .dependencies import ensure_mlarray_available, import_mlarray_module
from .exceptions import DependencyError, SlicerMLArrayError, UnsupportedLayoutError

__all__ = [
    "__version__",
    "DependencyError",
    "ResolvedMlaLayout",
    "SlicerGeometry",
    "SlicerMLArrayError",
    "UnsupportedLayoutError",
    "build_export_axis_labels",
    "ensure_mlarray_available",
    "ijk_to_ras_matrix_from_geometry",
    "import_mlarray_module",
    "mlarray_geometry_to_slicer",
    "mla_array_to_slicer",
    "resolve_supported_mla_layout",
    "slicer_geometry_to_mlarray",
]

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - during editable installs pre-build
    __version__ = "0.0.0"
