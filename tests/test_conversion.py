from __future__ import annotations

import numpy as np
import pytest

from slicermlarray.conversion import (
    MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    build_export_axis_labels,
    coord_system_transform_to_ras,
    export_scalar_like_array_to_mla,
    ijk_to_ras_matrix_from_geometry,
    mlarray_affine_to_slicer,
    mlarray_geometry_to_slicer,
    mla_array_to_slicer,
    resolve_supported_mla_layout,
    resolve_world_coord_system,
    slicer_geometry_to_mlarray,
)
from slicermlarray.exceptions import UnsupportedLayoutError


def test_resolve_unlabeled_scalar_layout():
    layout = resolve_supported_mla_layout((8, 9, 10), None)
    assert layout.kind == "scalar"
    assert layout.spatial_axes_zyx == (0, 1, 2)
    assert layout.slicer_shape == (8, 9, 10)


def test_resolve_labelmap_layout_from_is_seg():
    layout = resolve_supported_mla_layout((8, 9, 10), None, is_seg=True)
    assert layout.kind == "labelmap"


def test_resolve_explicit_vector_layout_with_axis_permutation():
    layout = resolve_supported_mla_layout(
        (5, 2, 7, 3),
        ("spatial_y", "channel", "spatial_x", "spatial_z"),
    )
    assert layout.kind == "vector"
    assert layout.spatial_axes_zyx == (3, 0, 2)
    assert layout.component_axis == 1
    assert layout.slicer_shape == (3, 5, 7, 2)


def test_resolve_single_component_channel_as_scalar():
    # Shape (1, z, y, x) with a size-1 channel axis should become scalar, not fail.
    layout = resolve_supported_mla_layout(
        (1, 344, 512, 512),
        ("channel", "spatial_z", "spatial_y", "spatial_x"),
    )
    assert layout.kind == "scalar"
    assert layout.component_axis is None
    assert layout.spatial_axes_zyx == (1, 2, 3)
    assert layout.slicer_shape == (344, 512, 512)


def test_mla_array_to_slicer_squeezes_single_component_channel():
    layout = resolve_supported_mla_layout(
        (1, 344, 512, 512),
        ("channel", "spatial_z", "spatial_y", "spatial_x"),
    )
    array = np.zeros((1, 344, 512, 512), dtype=np.float32)
    result = mla_array_to_slicer(array, layout)
    assert result.shape == (344, 512, 512)


def test_reject_unlabeled_4d_layout():
    with pytest.raises(UnsupportedLayoutError):
        resolve_supported_mla_layout((5, 6, 7, 3), None)


def test_reject_generic_spatial_labels():
    with pytest.raises(UnsupportedLayoutError):
        resolve_supported_mla_layout((5, 6, 7), ("spatial", "spatial", "spatial"))


def test_reject_extra_non_image_axes():
    with pytest.raises(UnsupportedLayoutError):
        resolve_supported_mla_layout(
            (2, 5, 6, 7),
            ("temporal", "spatial_z", "spatial_y", "spatial_x"),
        )


def test_mla_array_to_slicer_reorders_axes():
    layout = resolve_supported_mla_layout(
        (2, 3, 4, 5),
        ("channel", "spatial_x", "spatial_y", "spatial_z"),
    )
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    reordered = mla_array_to_slicer(array, layout)
    assert reordered.shape == (5, 4, 3, 2)
    assert np.array_equal(reordered, np.transpose(array, axes=(3, 2, 1, 0)))


def test_geometry_roundtrip_between_mlarray_and_slicer():
    spacing_xyz = (0.7, 0.8, 1.5)
    origin_xyz = (12.0, -4.5, 99.0)
    direction_xyz = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    spacing_mla, origin_mla, direction_mla = slicer_geometry_to_mlarray(
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        direction_xyz=direction_xyz,
        metadata_spatial_labels=MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    )

    geometry = mlarray_geometry_to_slicer(
        MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        spacing=spacing_mla,
        origin=origin_mla,
        direction=direction_mla,
    )

    assert geometry.spacing_xyz == pytest.approx(spacing_xyz)
    assert geometry.origin_xyz == pytest.approx(origin_xyz)
    assert np.allclose(geometry.direction_xyz, direction_xyz)


def test_geometry_roundtrip_between_mlarray_lps_and_slicer_ras():
    spacing_mla = (1.5, 2.0, 3.0)
    origin_mla_lps = (30.0, 20.0, 10.0)
    direction_mla_lps = np.eye(3)

    geometry = mlarray_geometry_to_slicer(
        MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        spacing=spacing_mla,
        origin=origin_mla_lps,
        direction=direction_mla_lps,
        coord_system="LPS",
    )

    assert geometry.spacing_xyz == pytest.approx((3.0, 2.0, 1.5))
    assert geometry.origin_xyz == pytest.approx((-10.0, -20.0, 30.0))
    assert np.allclose(geometry.direction_xyz, np.diag((-1.0, -1.0, 1.0)))

    spacing_back, origin_back, direction_back = slicer_geometry_to_mlarray(
        spacing_xyz=geometry.spacing_xyz,
        origin_xyz=geometry.origin_xyz,
        direction_xyz=geometry.direction_xyz,
        metadata_spatial_labels=MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        coord_system="LPS",
    )
    assert spacing_back == pytest.approx(list(spacing_mla))
    assert origin_back == pytest.approx(list(origin_mla_lps))
    assert np.allclose(direction_back, direction_mla_lps)


def test_ijk_to_ras_matrix_from_geometry():
    geometry = mlarray_geometry_to_slicer(
        MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        spacing=(1.5, 2.0, 3.0),
        origin=(30.0, 20.0, 10.0),
        direction=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    matrix = ijk_to_ras_matrix_from_geometry(geometry)
    assert np.allclose(matrix[:3, :3], np.diag((3.0, 2.0, 1.5)))
    assert np.allclose(matrix[:3, 3], (10.0, 20.0, 30.0))


def test_mlarray_affine_to_slicer_for_zyx_ras():
    affine = np.array(
        [
            [1.5, 0.0, 0.0, 30.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix = mlarray_affine_to_slicer(
        MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        affine=affine,
        coord_system="RAS",
    )
    assert np.allclose(matrix[:3, :3], np.diag((3.0, 2.0, 1.5)))
    assert np.allclose(matrix[:3, 3], (10.0, 20.0, 30.0))


def test_mlarray_affine_to_slicer_for_zyx_lps():
    affine = np.array(
        [
            [1.5, 0.0, 0.0, 30.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix = mlarray_affine_to_slicer(
        MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        affine=affine,
        coord_system="LPS",
    )
    assert np.allclose(matrix[:3, :3], np.diag((-3.0, -2.0, 1.5)))
    assert np.allclose(matrix[:3, 3], (-10.0, -20.0, 30.0))


def test_export_scalar_like_array_validation():
    array = np.zeros((4, 5, 6), dtype=np.int16)
    assert export_scalar_like_array_to_mla(array) is array
    with pytest.raises(UnsupportedLayoutError):
        export_scalar_like_array_to_mla(np.zeros((4, 5), dtype=np.int16))


def test_build_export_axis_labels():
    assert build_export_axis_labels("scalar") == ["spatial_z", "spatial_y", "spatial_x"]
    assert build_export_axis_labels("labelmap") == ["spatial_z", "spatial_y", "spatial_x"]


def test_resolve_world_coord_system():
    assert resolve_world_coord_system("RAS") == "RAS"
    assert resolve_world_coord_system("LPS") == "LPS"
    assert resolve_world_coord_system(None, default="RAS") == "RAS"
    with pytest.raises(UnsupportedLayoutError):
        resolve_world_coord_system("unknown")


def test_resolve_world_coord_system_ras_plus_alias():
    assert resolve_world_coord_system("RAS+") == "RAS"
    assert resolve_world_coord_system("LPS+") == "LPS"


def test_coord_system_transform_to_ras():
    assert np.allclose(coord_system_transform_to_ras("RAS"), np.eye(3))
    assert np.allclose(coord_system_transform_to_ras("LPS"), np.diag((-1.0, -1.0, 1.0)))


def test_coord_system_transform_to_ras_plus_aliases():
    assert np.allclose(coord_system_transform_to_ras("RAS+"), np.eye(3))
    assert np.allclose(coord_system_transform_to_ras("LPS+"), np.diag((-1.0, -1.0, 1.0)))
