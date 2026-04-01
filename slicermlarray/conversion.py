"""Pure-Python layout and geometry mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from .exceptions import UnsupportedLayoutError

VolumeKind = Literal["scalar", "labelmap", "vector"]

SPATIAL_X = "spatial_x"
SPATIAL_Y = "spatial_y"
SPATIAL_Z = "spatial_z"

SLICER_SPATIAL_LABELS_XYZ = (SPATIAL_X, SPATIAL_Y, SPATIAL_Z)
MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX = (SPATIAL_Z, SPATIAL_Y, SPATIAL_X)
CHANNEL_AXIS_LABELS = frozenset({"channel", "components"})
SUPPORTED_AXIS_LABELS = frozenset(
    {
        SPATIAL_X,
        SPATIAL_Y,
        SPATIAL_Z,
        *CHANNEL_AXIS_LABELS,
    }
)


@dataclass(frozen=True)
class ResolvedMlaLayout:
    """Supported MLArray layout mapped onto a Slicer volume kind."""

    kind: VolumeKind
    source_shape: tuple[int, ...]
    axis_labels: tuple[str, ...] | None
    spatial_axes_zyx: tuple[int, int, int]
    metadata_spatial_labels: tuple[str, str, str]
    component_axis: int | None = None

    @property
    def slicer_shape(self) -> tuple[int, ...]:
        """Output numpy shape expected by `slicer.util.addVolumeFromArray`."""
        zyx = tuple(self.source_shape[index] for index in self.spatial_axes_zyx)
        if self.component_axis is None:
            return zyx
        return zyx + (self.source_shape[self.component_axis],)


@dataclass(frozen=True)
class SlicerGeometry:
    """Geometry in Slicer's IJK-to-RAS convention."""

    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction_xyz: np.ndarray


def normalize_axis_labels(axis_labels: Sequence[object] | None) -> tuple[str, ...] | None:
    """Normalize axis labels to a tuple of strings."""
    if axis_labels is None:
        return None
    return tuple(str(label) for label in axis_labels)


def resolve_supported_mla_layout(
    shape: Sequence[int],
    axis_labels: Sequence[object] | None,
    *,
    is_seg: bool = False,
) -> ResolvedMlaLayout:
    """Resolve a supported MLArray image layout for Slicer import."""
    normalized_shape = tuple(int(dim) for dim in shape)
    ndim = len(normalized_shape)
    labels = normalize_axis_labels(axis_labels)

    if labels is None:
        if ndim != 3:
            raise UnsupportedLayoutError(
                "Only unlabeled 3D MLArray images are supported. "
                "Add explicit axis_labels for non-scalar layouts."
            )
        return ResolvedMlaLayout(
            kind="labelmap" if is_seg else "scalar",
            source_shape=normalized_shape,
            axis_labels=None,
            spatial_axes_zyx=(0, 1, 2),
            metadata_spatial_labels=MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        )

    if len(labels) != ndim:
        raise UnsupportedLayoutError(
            f"axis_labels length {len(labels)} does not match array ndim {ndim}."
        )

    unsupported_labels = [label for label in labels if label not in SUPPORTED_AXIS_LABELS]
    if unsupported_labels:
        raise UnsupportedLayoutError(
            "Unsupported axis labels for Slicer import: "
            + ", ".join(sorted(set(unsupported_labels)))
        )

    if "spatial" in labels:
        raise UnsupportedLayoutError(
            "Generic 'spatial' axis labels are ambiguous for Slicer import. "
            "Use explicit spatial_z/spatial_y/spatial_x labels."
        )

    spatial_positions = {
        label: [index for index, axis_label in enumerate(labels) if axis_label == label]
        for label in SLICER_SPATIAL_LABELS_XYZ
    }
    missing_spatial = [label for label, positions in spatial_positions.items() if len(positions) != 1]
    if missing_spatial:
        raise UnsupportedLayoutError(
            "Slicer import requires exactly one each of "
            "spatial_x, spatial_y, and spatial_z."
        )

    component_axes = [
        index for index, label in enumerate(labels) if label in CHANNEL_AXIS_LABELS
    ]
    if len(component_axes) > 1:
        raise UnsupportedLayoutError("Only one channel/components axis is supported.")
    if is_seg and component_axes:
        raise UnsupportedLayoutError(
            "Segmentations with a channel/components axis are not supported."
        )

    spatial_axes_zyx = (
        spatial_positions[SPATIAL_Z][0],
        spatial_positions[SPATIAL_Y][0],
        spatial_positions[SPATIAL_X][0],
    )
    metadata_spatial_labels = tuple(
        label for label in labels if label in SLICER_SPATIAL_LABELS_XYZ
    )
    component_axis = component_axes[0] if component_axes else None

    if component_axis is None:
        if ndim != 3:
            raise UnsupportedLayoutError(
                "Only 3D scalar/labelmap MLArray images are supported without a "
                "channel/components axis."
            )
        kind: VolumeKind = "labelmap" if is_seg else "scalar"
        return ResolvedMlaLayout(
            kind=kind,
            source_shape=normalized_shape,
            axis_labels=labels,
            spatial_axes_zyx=spatial_axes_zyx,
            metadata_spatial_labels=metadata_spatial_labels,
        )

    if ndim != 4:
        raise UnsupportedLayoutError(
            "Vector volume import requires exactly 3 spatial axes plus one "
            "channel/components axis."
        )
    if normalized_shape[component_axis] <= 1:
        raise UnsupportedLayoutError(
            "Vector volume import requires more than one component."
        )

    return ResolvedMlaLayout(
        kind="vector",
        source_shape=normalized_shape,
        axis_labels=labels,
        spatial_axes_zyx=spatial_axes_zyx,
        metadata_spatial_labels=metadata_spatial_labels,
        component_axis=component_axis,
    )


def mla_array_to_slicer(array: np.ndarray, layout: ResolvedMlaLayout) -> np.ndarray:
    """Reorder an MLArray ndarray into Slicer's numpy volume order."""
    axes = list(layout.spatial_axes_zyx)
    if layout.component_axis is not None:
        axes.append(layout.component_axis)
    if tuple(axes) == tuple(range(array.ndim)):
        return array
    return np.transpose(array, axes=axes)


def export_scalar_like_array_to_mla(array_zyx: np.ndarray) -> np.ndarray:
    """Validate a Slicer scalar/labelmap numpy array before MLArray export."""
    if array_zyx.ndim != 3:
        raise UnsupportedLayoutError(
            f"Only 3D scalar/labelmap volumes can be exported, got shape {array_zyx.shape}."
        )
    return array_zyx


def build_export_axis_labels(kind: VolumeKind) -> list[str]:
    """Return the MLArray axis labels used for Slicer export."""
    if kind in {"scalar", "labelmap"}:
        return list(MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX)
    if kind == "vector":
        return [*MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX, "channel"]
    raise UnsupportedLayoutError(f"Unsupported export kind: {kind}")


def _normalize_vector(values: Sequence[float] | None, default: Iterable[float]) -> tuple[float, float, float]:
    if values is None:
        return tuple(float(value) for value in default)
    if len(values) != 3:
        raise UnsupportedLayoutError(f"Expected 3 values, got {len(values)}.")
    return tuple(float(value) for value in values)


def _normalize_direction(direction: Sequence[Sequence[float]] | None) -> np.ndarray:
    if direction is None:
        return np.eye(3, dtype=float)
    matrix = np.asarray(direction, dtype=float)
    if matrix.shape != (3, 3):
        raise UnsupportedLayoutError(
            f"Expected a 3x3 direction matrix, got shape {matrix.shape}."
        )
    return matrix


def _metadata_permutation_to_xyz(metadata_spatial_labels: Sequence[str]) -> tuple[int, int, int]:
    labels = tuple(metadata_spatial_labels)
    if set(labels) != set(SLICER_SPATIAL_LABELS_XYZ) or len(labels) != 3:
        raise UnsupportedLayoutError(
            "Spatial metadata must resolve to spatial_x/spatial_y/spatial_z exactly once."
        )
    return tuple(labels.index(label) for label in SLICER_SPATIAL_LABELS_XYZ)


def mlarray_geometry_to_slicer(
    metadata_spatial_labels: Sequence[str],
    *,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
    direction: Sequence[Sequence[float]] | None,
) -> SlicerGeometry:
    """Convert MLArray spatial metadata into Slicer's xyz/RAS convention."""
    permutation = _metadata_permutation_to_xyz(metadata_spatial_labels)
    spacing_values = _normalize_vector(spacing, default=(1.0, 1.0, 1.0))
    origin_values = _normalize_vector(origin, default=(0.0, 0.0, 0.0))
    direction_matrix = _normalize_direction(direction)
    return SlicerGeometry(
        spacing_xyz=tuple(spacing_values[index] for index in permutation),
        origin_xyz=tuple(origin_values[index] for index in permutation),
        direction_xyz=direction_matrix[np.ix_(permutation, permutation)],
    )


def slicer_geometry_to_mlarray(
    *,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    direction_xyz: Sequence[Sequence[float]],
    metadata_spatial_labels: Sequence[str] = MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Convert Slicer geometry into MLArray spatial metadata order."""
    permutation = _metadata_permutation_to_xyz(metadata_spatial_labels)
    reverse_permutation = tuple(permutation.index(index) for index in range(3))
    spacing_values = _normalize_vector(spacing_xyz, default=(1.0, 1.0, 1.0))
    origin_values = _normalize_vector(origin_xyz, default=(0.0, 0.0, 0.0))
    direction_matrix = _normalize_direction(direction_xyz)
    return (
        [float(spacing_values[index]) for index in reverse_permutation],
        [float(origin_values[index]) for index in reverse_permutation],
        direction_matrix[np.ix_(reverse_permutation, reverse_permutation)].tolist(),
    )


def ijk_to_ras_matrix_from_geometry(geometry: SlicerGeometry) -> np.ndarray:
    """Build a 4x4 IJK-to-RAS matrix from Slicer spacing/origin/direction."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = geometry.direction_xyz @ np.diag(geometry.spacing_xyz)
    matrix[:3, 3] = np.asarray(geometry.origin_xyz, dtype=float)
    return matrix
