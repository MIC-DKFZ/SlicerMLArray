"""Unit tests for bbox_conversion.py — pure math, no Slicer runtime needed."""

from __future__ import annotations

import numpy as np
import pytest

from slicermlarray.bbox_conversion import bbox_vox_to_slicer_roi, slicer_roi_to_bbox_vox
from slicermlarray.conversion import MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX

# Shorthand aliases
LABELS_XYZ = ("spatial_x", "spatial_y", "spatial_z")
LABELS_ZYX = MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX  # ("spatial_z","spatial_y","spatial_x")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _diag_affine(sx, sy, sz, ox=0.0, oy=0.0, oz=0.0):
    """Diagonal (no-rotation) 4×4 affine with given spacings and origin."""
    return np.array([
        [sx, 0,  0,  ox],
        [0,  sy, 0,  oy],
        [0,  0,  sz, oz],
        [0,  0,  0,  1],
    ], dtype=float)


# ── Forward: voxel bbox → Slicer ROI ──────────────────────────────────────────

class TestBboxVoxToSlicerRoi:

    def test_identity_affine_xyz_labels(self):
        """Unit spacing, XYZ labels: center and size in mm == voxel values."""
        affine = np.eye(4)
        bbox = [[2, 5], [10, 15], [0, 3]]  # x:[2,5] y:[10,15] z:[0,3]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox, affine, LABELS_XYZ
        )
        # center = midpoints: x=(2+5)/2=3.5, y=(10+15)/2=12.5, z=(0+3)/2=1.5
        assert np.allclose(center_ras, [3.5, 12.5, 1.5])
        # edge-to-edge size: x=5-2+1=4, y=15-10+1=6, z=3-0+1=4
        assert np.allclose(size_mm, [4.0, 6.0, 4.0])
        # Local axes = identity (no rotation)
        assert np.allclose(obj_to_world[:3, :3], np.eye(3))
        assert np.allclose(obj_to_world[:3, 3], [3.5, 12.5, 1.5])

    def test_diagonal_affine_xyz_labels(self):
        """Diagonal affine with non-unit spacing, XYZ labels."""
        sx, sy, sz = 2.0, 3.0, 1.5
        ox, oy, oz = -100.0, 50.0, 0.0
        affine = _diag_affine(sx, sy, sz, ox, oy, oz)
        bbox = [[10, 20], [5, 10], [0, 5]]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox, affine, LABELS_XYZ
        )
        # vox center: x=15, y=7.5, z=2.5 → world: x=15*2-100=−70, y=7.5*3+50=72.5, z=2.5*1.5=3.75
        assert np.allclose(center_ras, [-70.0, 72.5, 3.75])
        # size_mm: x=(20-10+1)*2=22, y=(10-5+1)*3=18, z=(5-0+1)*1.5=9
        assert np.allclose(size_mm, [22.0, 18.0, 9.0])

    def test_zyx_labels_permuted_correctly(self):
        """ZYX metadata order: bbox axes are [z,y,x] but ROI size_mm is [x,y,z]."""
        sx, sy, sz = 1.0, 2.0, 3.0
        affine = _diag_affine(sz, sy, sx)  # ZYX affine: col0=Z, col1=Y, col2=X
        # bbox in ZYX order: z:[0,9], y:[0,4], x:[0,2]
        bbox_zyx = [[0, 9], [0, 4], [0, 2]]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox_zyx, affine, LABELS_ZYX
        )
        # permutation_ZYX: X is at index 2, Y at 1, Z at 0
        # size_x_mm = (2-0+1)*sx = 3*1 = 3
        # size_y_mm = (4-0+1)*sy = 5*2 = 10
        # size_z_mm = (9-0+1)*sz = 10*3 = 30
        assert np.allclose(size_mm, [3.0, 10.0, 30.0])
        # center_ras for X: vox center_x = 1.0, world_x = 1.0*sx = 1.0  (Z col → RAS Z in world? no)
        # Wait: with ZYX affine, col0=direction of vox axis 0 = Z direction in world
        # center_meta = [4.5, 2.0, 1.0] (z, y, x centers)
        # world = affine[:3,:3] @ [4.5,2.0,1.0] = [4.5*3, 2.0*2, 1.0*1] = [13.5, 4.0, 1.0]
        # coord_system=RAS → center_ras = [13.5, 4.0, 1.0]
        assert np.allclose(center_ras, [13.5, 4.0, 1.0])

    def test_lps_to_ras_transform(self):
        """LPS coord_system: X and Y are negated in RAS output."""
        affine = _diag_affine(1.0, 1.0, 1.0, 10.0, 20.0, 30.0)
        bbox = [[0, 0], [0, 0], [0, 0]]  # single voxel at origin
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox, affine, LABELS_XYZ, coord_system="LPS"
        )
        # world center in LPS = [10, 20, 30]; RAS = [-10, -20, 30]
        assert np.allclose(center_ras, [-10.0, -20.0, 30.0])
        # LPS+ aliases
        center2, _, _ = bbox_vox_to_slicer_roi(
            bbox, affine, LABELS_XYZ, coord_system="LPS+"
        )
        assert np.allclose(center2, center_ras)

    def test_obj_to_world_is_valid_rotation(self):
        """obj_to_world[:3,:3] must be orthonormal (rotation matrix)."""
        affine = _diag_affine(2.0, 3.0, 1.5)
        bbox = [[5, 10], [3, 8], [0, 4]]
        _, _, obj_to_world = bbox_vox_to_slicer_roi(bbox, affine, LABELS_XYZ)
        R = obj_to_world[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


# ── Inverse: Slicer ROI → voxel bbox ─────────────────────────────────────────

class TestSlicerRoiToBboxVox:

    def test_identity_roundtrip_xyz(self):
        """Forward then inverse should recover the original bbox."""
        affine = _diag_affine(2.0, 3.0, 1.5, -100.0, 50.0, 0.0)
        bbox_orig = [[10, 20], [5, 10], [0, 5]]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox_orig, affine, LABELS_XYZ
        )
        bbox_back = slicer_roi_to_bbox_vox(
            center_ras, size_mm, obj_to_world, affine, LABELS_XYZ
        )
        for i in range(3):
            assert pytest.approx(bbox_back[i][0], abs=0.5) == bbox_orig[i][0]
            assert pytest.approx(bbox_back[i][1], abs=0.5) == bbox_orig[i][1]

    def test_identity_roundtrip_zyx(self):
        """ZYX labels: forward + inverse round-trip."""
        affine = _diag_affine(3.0, 2.0, 1.0)  # ZYX affine
        bbox_orig = [[0, 9], [0, 4], [0, 2]]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox_orig, affine, LABELS_ZYX
        )
        bbox_back = slicer_roi_to_bbox_vox(
            center_ras, size_mm, obj_to_world, affine, LABELS_ZYX
        )
        for i in range(3):
            assert pytest.approx(bbox_back[i][0], abs=0.5) == bbox_orig[i][0]
            assert pytest.approx(bbox_back[i][1], abs=0.5) == bbox_orig[i][1]

    def test_roundtrip_lps(self):
        """LPS coord_system round-trip."""
        affine = _diag_affine(1.0, 1.5, 2.0, 30.0, 20.0, 10.0)
        bbox_orig = [[3, 10], [2, 8], [1, 6]]
        center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
            bbox_orig, affine, LABELS_XYZ, coord_system="LPS"
        )
        bbox_back = slicer_roi_to_bbox_vox(
            center_ras, size_mm, obj_to_world, affine, LABELS_XYZ, coord_system="LPS"
        )
        for i in range(3):
            assert pytest.approx(bbox_back[i][0], abs=0.5) == bbox_orig[i][0]
            assert pytest.approx(bbox_back[i][1], abs=0.5) == bbox_orig[i][1]


# ── Single-voxel and edge cases ───────────────────────────────────────────────

class TestEdgeCases:

    def test_single_voxel_bbox(self):
        """Single voxel: size_mm == spacing, center == world position of voxel."""
        sx, sy, sz = 2.0, 2.0, 2.0
        ox, oy, oz = 0.0, 0.0, 0.0
        affine = _diag_affine(sx, sy, sz, ox, oy, oz)
        bbox = [[5, 5], [3, 3], [7, 7]]  # single voxel
        center_ras, size_mm, _ = bbox_vox_to_slicer_roi(bbox, affine, LABELS_XYZ)
        assert np.allclose(center_ras, [10.0, 6.0, 14.0])
        assert np.allclose(size_mm, [2.0, 2.0, 2.0])

    def test_zero_origin_unit_spacing_xyz(self):
        """Sanity: unit affine, first voxel bbox."""
        affine = np.eye(4)
        bbox = [[0, 0], [0, 0], [0, 0]]
        center_ras, size_mm, _ = bbox_vox_to_slicer_roi(bbox, affine, LABELS_XYZ)
        assert np.allclose(center_ras, [0.0, 0.0, 0.0])
        assert np.allclose(size_mm, [1.0, 1.0, 1.0])
