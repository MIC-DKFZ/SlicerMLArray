"""Pure-Python bounding-box ↔ Slicer ROI coordinate conversion helpers.

All functions are free of Slicer / VTK imports so they can be unit-tested
outside the Slicer runtime.

Coordinate conventions
----------------------
* **MLArray bbox** – ``bboxes[i]`` is a list of ``[min, max]`` pairs in
  *voxel index* space, one pair per array axis in *metadata axis order*
  (i.e. the order defined by ``axis_labels`` / ``metadata_spatial_labels``).
  Both min and max are inclusive voxel indices.

* **Slicer ROI** – a ``vtkMRMLMarkupsROINode`` described by three quantities:

  * ``center_ras`` – 3-D centre in RAS world coordinates.
  * ``size_mm`` – full extents ``[sx, sy, sz]`` in millimetres along the
    ROI's *local* X / Y / Z axes.
  * ``obj_to_world`` – 4 × 4 homogeneous matrix whose first three columns
    are the normalised local X, Y, Z axes in RAS world space, and whose
    fourth column is the centre in RAS.

The ROI's local axes are always chosen to align with the image's voxel
axes (i.e. ``affine`` column directions) so that the ROI snugly encloses
the voxel bounding box without rotation artefacts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .conversion import (
    _metadata_permutation_to_xyz,
    coord_system_transform_to_ras,
)


def bbox_vox_to_slicer_roi(
    bbox_per_axis: Sequence[Sequence[float]],
    affine: Sequence[Sequence[float]],
    metadata_spatial_labels: Sequence[str],
    coord_system: object = "RAS",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a voxel-space bounding box to Slicer ROI parameters.

    The returned ROI is axis-aligned with the image's voxel axes in RAS space.

    Args:
        bbox_per_axis: ``[[min0, max0], [min1, max1], [min2, max2]]`` in
            metadata axis order (inclusive voxel indices).
        affine: 4 × 4 homogeneous affine that maps voxel indices (in
            metadata axis order) to world coordinates in ``coord_system``.
        metadata_spatial_labels: Axis labels in the order they appear in
            the array (e.g. ``("spatial_z","spatial_y","spatial_x")``).
        coord_system: World coordinate system of ``affine``
            (``"RAS"`` or ``"LPS"``; aliases ``"RAS+"``/``"LPS+"`` accepted).

    Returns:
        ``(center_ras, size_mm, obj_to_world)`` where

        * ``center_ras`` – shape ``(3,)`` centre in RAS.
        * ``size_mm`` – shape ``(3,)`` full extents in ROI local X/Y/Z frame.
        * ``obj_to_world`` – shape ``(4,4)`` object-to-world matrix.
    """
    bbox = [list(pair) for pair in bbox_per_axis]
    ndim = len(bbox)
    affine_matrix = np.asarray(affine, dtype=float)
    basis_to_ras = coord_system_transform_to_ras(coord_system, default="RAS")

    # Permutation: permutation[d] = metadata axis index for spatial direction d (X=0,Y=1,Z=2)
    permutation = _metadata_permutation_to_xyz(metadata_spatial_labels)

    # Centre and size in metadata axis order
    vox_center_meta = np.array([(bbox[i][0] + bbox[i][1]) / 2.0 for i in range(ndim)])
    # Edge-to-edge inclusive: size = max - min + 1
    vox_size_meta = np.array([bbox[i][1] - bbox[i][0] + 1.0 for i in range(ndim)])

    # Centre in stored-coord world space then in RAS
    world_center = affine_matrix[:3, :3] @ vox_center_meta + affine_matrix[:3, 3]
    center_ras = basis_to_ras @ world_center

    # ROI local axes (XYZ) = image voxel axis directions in RAS
    # Column permutation[d] of affine → direction of spatial axis d (X/Y/Z) in stored coord
    obj_to_world = np.eye(4, dtype=float)
    size_mm = np.zeros(3, dtype=float)
    for d in range(3):
        col_idx = permutation[d]
        col = affine_matrix[:3, col_idx]
        spacing = float(np.linalg.norm(col))
        direction_stored = col / spacing if spacing > 0 else np.eye(3)[d]
        obj_to_world[:3, d] = basis_to_ras @ direction_stored
        size_mm[d] = vox_size_meta[col_idx] * spacing

    obj_to_world[:3, 3] = center_ras
    return center_ras, size_mm, obj_to_world


def slicer_roi_to_bbox_vox(
    center_ras: Sequence[float],
    size_mm: Sequence[float],
    obj_to_world: Sequence[Sequence[float]],
    affine: Sequence[Sequence[float]],
    metadata_spatial_labels: Sequence[str],
    coord_system: object = "RAS",
) -> list[list[float]]:
    """Convert Slicer ROI parameters to a voxel-space bounding box.

    The result is the axis-aligned bounding box (AABB) in voxel space that
    tightly contains all eight corners of the ROI.  This handles both
    image-aligned and arbitrarily oriented ROIs.

    Args:
        center_ras: ROI centre in RAS world coordinates.
        size_mm: Full extents ``[sx, sy, sz]`` in the ROI's local frame.
        obj_to_world: 4 × 4 object-to-world matrix (RAS).
        affine: 4 × 4 homogeneous affine (voxel metadata → stored coord_system).
        metadata_spatial_labels: Axis labels in the order they appear in
            the array.
        coord_system: World coordinate system of ``affine``.

    Returns:
        ``[[min0, max0], [min1, max1], [min2, max2]]`` in metadata axis order
        (inclusive voxel indices, floats).
    """
    half = np.asarray(size_mm, dtype=float) / 2.0
    obj_to_world_m = np.asarray(obj_to_world, dtype=float)
    basis_to_ras = coord_system_transform_to_ras(coord_system, default="RAS")
    basis_from_ras = basis_to_ras.T  # inverse of orthogonal 3×3

    # 8 corners of the ROI in its local frame
    corners_local = np.array([
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])  # (8, 3)

    # Transform corners to RAS world then to stored-coord world
    corners_ras = (obj_to_world_m[:3, :3] @ corners_local.T).T + np.asarray(center_ras)
    corners_stored = (basis_from_ras @ corners_ras.T).T  # (8, 3)

    # Map to voxel metadata space via affine inverse
    affine_matrix = np.asarray(affine, dtype=float)
    affine_inv = np.linalg.inv(affine_matrix)
    corners_hom = np.hstack([corners_stored, np.ones((8, 1))])
    corners_vox_meta = (affine_inv @ corners_hom.T)[:3, :].T  # (8, 3)

    ndim = len(metadata_spatial_labels)
    bbox_per_axis = [
        [float(corners_vox_meta[:, i].min()), float(corners_vox_meta[:, i].max())]
        for i in range(ndim)
    ]
    return bbox_per_axis
