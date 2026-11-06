"""Boxes of the ``ml`` profile (``profiles.ml.boxes``) <-> Slicer ROIs.

A box is ``min``/``max`` per named axis: array indices (``frame: index``, inclusive voxel centres) or space
coordinates (``frame: <space name>``). A Slicer ROI is a centre and a size in RAS millimetre along three orthonormal
axes. An index box becomes an ROI aligned with the voxel grid (also for flipped and oblique grids); it covers the
voxels from edge to edge, so its size is ``(max - min + 1) * spacing``. A sheared grid has no orthonormal axes:
its box becomes the RAS bounding box. Axes a box does not name span the whole image.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import space_to_ras


@dataclass
class Roi:
    center: np.ndarray          #: RAS
    size: np.ndarray            #: millimetre along ``axes``
    axes: np.ndarray            #: 3x3, columns = ROI axes in RAS (a rotation)

    def corners(self) -> np.ndarray:
        half = self.size / 2.0
        signs = np.array(list(product((-1.0, 1.0), repeat=3)))
        return self.center + (signs * half) @ self.axes.T


def box_spec(meta) -> Optional[dict]:
    boxes = (meta.profiles.get("ml") or {}).get("boxes") or {}
    return dict(boxes) if boxes.get("items") else None


def ijk_axis_names(meta, spatial_axes: Sequence[int]) -> List[Optional[str]]:
    """Array axis name of I, J, K (None for the K of a 2-D image)."""
    names = [meta.axes[i].name for i in spatial_axes][::-1]
    return names + [None] * (3 - len(names))


def roi_from_index_box(lo, hi, box_axes, ijk_names, ijk_to_ras, ijk_shape) -> Roi:
    m = np.asarray(ijk_to_ras, dtype=np.float64)
    low = np.zeros(3)
    high = np.array([n - 1 for n in ijk_shape], dtype=np.float64)
    for name, a, b in zip(box_axes, lo, hi):
        if name in ijk_names:
            c = ijk_names.index(name)
            low[c], high[c] = float(a), float(b)
    columns = m[:3, :3]
    spacing = np.linalg.norm(columns, axis=0)
    axes = columns / spacing
    if np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        if np.linalg.det(axes) < 0:
            axes = axes.copy()
            axes[:, 2] *= -1.0                       # same box, right-handed ROI axes
        center = m[:3, :3] @ ((low + high) / 2.0) + m[:3, 3]
        return Roi(center, (high - low + 1.0) * spacing, axes)
    edges = np.array([[low[c] - 0.5, high[c] + 0.5] for c in range(3)])
    corners = np.array([m[:3, :3] @ np.array(p) + m[:3, 3] for p in product(*edges)])
    return _aabb(corners)


def roi_from_space_box(lo, hi, box_axes, space) -> Roi:
    to_ras, _, _ = space_to_ras(space)
    names = [axis.name for axis in space.axes]
    low = np.zeros(len(names))
    high = np.zeros(len(names))
    for name, a, b in zip(box_axes, lo, hi):
        if name in names:
            low[names.index(name)], high[names.index(name)] = float(a), float(b)
    corners = np.array([to_ras @ np.array(p) for p in product(*zip(low, high))])
    return _aabb(corners)


def _aabb(corners: np.ndarray) -> Roi:
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return Roi((lo + hi) / 2.0, hi - lo, np.eye(3))


def index_box_from_roi(roi: Roi, box_axes, ijk_names, ijk_to_ras) -> Tuple[list, list]:
    """``min``/``max`` (array indices, in ``box_axes`` order) of the voxels an ROI covers."""
    inverse = np.linalg.inv(np.asarray(ijk_to_ras, dtype=np.float64))
    ijk = np.array([inverse[:3, :3] @ p + inverse[:3, 3] for p in roi.corners()])
    low, high = ijk.min(axis=0) + 0.5, ijk.max(axis=0) - 0.5
    lo, hi = [], []
    for name in box_axes:
        c = ijk_names.index(name)
        lo.append(_number(low[c]))
        hi.append(_number(max(high[c], low[c])))
    return lo, hi


def space_box_from_roi(roi: Roi, box_axes, space) -> Tuple[list, list]:
    """``min``/``max`` (space coordinates, in ``box_axes`` order) of the bounding box of an ROI."""
    to_ras, _, _ = space_to_ras(space)
    coords = np.array([np.linalg.pinv(to_ras) @ p for p in roi.corners()])
    names = [axis.name for axis in space.axes]
    low, high = coords.min(axis=0), coords.max(axis=0)
    return ([_number(low[names.index(n)], 1e-6) for n in box_axes],
            [_number(high[names.index(n)], 1e-6) for n in box_axes])


def _number(value: float, tolerance: float = 1e-3):
    """An int when ``value`` is one (within ``tolerance``), else a float rounded to 1e-6."""
    rounded = round(float(value))
    if abs(value - rounded) <= tolerance:
        return int(rounded)
    return round(float(value), 6)
