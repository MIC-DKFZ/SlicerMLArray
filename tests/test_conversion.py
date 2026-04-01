from __future__ import annotations

import numpy as np
import pytest

from slicermlarray.conversion import (
    MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    build_export_axis_labels,
    export_scalar_like_array_to_mla,
    ijk_to_ras_matrix_from_geometry,
    mlarray_geometry_to_slicer,
    mla_array_to_slicer,
    resolve_supported_mla_layout,
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


def test_export_scalar_like_array_validation():
    array = np.zeros((4, 5, 6), dtype=np.int16)
    assert export_scalar_like_array_to_mla(array) is array
    with pytest.raises(UnsupportedLayoutError):
        export_scalar_like_array_to_mla(np.zeros((4, 5), dtype=np.int16))


def test_build_export_axis_labels():
    assert build_export_axis_labels("scalar") == ["spatial_z", "spatial_y", "spatial_x"]
    assert build_export_axis_labels("labelmap") == ["spatial_z", "spatial_y", "spatial_x"]
