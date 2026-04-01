"""Slicer runtime bridge for MLArray file IO."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .conversion import (
    MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    ResolvedMlaLayout,
    VolumeKind,
    build_export_axis_labels,
    export_scalar_like_array_to_mla,
    ijk_to_ras_matrix_from_geometry,
    mlarray_geometry_to_slicer,
    mla_array_to_slicer,
    resolve_supported_mla_layout,
    slicer_geometry_to_mlarray,
)
from .dependencies import ensure_mlarray_available
from .exceptions import DependencyError, UnsupportedLayoutError

FILE_TYPE = "MLArrayFile"
FILE_DESCRIPTION = "MLArray volume"
FILE_EXTENSION_FILTER = "MLArray volume (*.mla)"

ATTR_SOURCE_META_JSON = "SlicerMLArray.SourceMetaJSON"
ATTR_EXTRA_META_JSON = "SlicerMLArray.ExtraMetaJSON"
ATTR_SOURCE_PATH = "SlicerMLArray.SourcePath"
ATTR_KIND = "SlicerMLArray.Kind"
ATTR_FILE_CATEGORY = "SlicerMLArray.FileCategory"
FILE_CATEGORY_VALUE = "mla"


def _confirm_install(message: str) -> bool:
    import slicer

    return slicer.util.confirmYesNoDisplay(message, windowTitle="Install MLArray")


def _install_mlarray(requirement: str) -> None:
    import slicer

    slicer.util.pip_install(requirement)


def _runtime_mlarray(auto_install: bool):
    return ensure_mlarray_available(
        auto_install=auto_install,
        installer=_install_mlarray if auto_install else None,
        prompt=_confirm_install if auto_install else None,
    )


def _vtk_matrix4x4_from_numpy(array: np.ndarray):
    import vtk

    matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            matrix.SetElement(row, column, float(array[row, column]))
    return matrix


def _numpy_from_vtk_matrix3x3(matrix) -> np.ndarray:
    array = np.eye(3, dtype=float)
    for row in range(3):
        for column in range(3):
            array[row, column] = matrix.GetElement(row, column)
    return array


def _get_base_name(file_path: str, explicit_name: str | None) -> str:
    import slicer

    if explicit_name:
        return explicit_name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return slicer.mrmlScene.GenerateUniqueName(base_name)


def _get_json_attribute(node, key: str) -> dict[str, Any]:
    raw_value = node.GetAttribute(key)
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _set_json_attribute(node, key: str, value: dict[str, Any]) -> None:
    node.SetAttribute(key, json.dumps(value, sort_keys=True))


def _store_mlarray_node_attributes(node, *, file_path: str, kind: VolumeKind, source_meta: dict[str, Any], extra_meta: dict[str, Any]) -> None:
    node.SetAttribute(ATTR_SOURCE_PATH, file_path)
    node.SetAttribute(ATTR_KIND, kind)
    node.SetAttribute(ATTR_FILE_CATEGORY, FILE_CATEGORY_VALUE)
    _set_json_attribute(node, ATTR_SOURCE_META_JSON, source_meta)
    _set_json_attribute(node, ATTR_EXTRA_META_JSON, extra_meta)


def _center_slice_views_on_node(node) -> None:
    import slicer

    bounds = [0.0] * 6
    node.GetRASBounds(bounds)
    center_ras = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )

    layout_manager = slicer.app.layoutManager()
    if layout_manager is None:
        return

    for slice_view_name in layout_manager.sliceViewNames():
        slice_widget = layout_manager.sliceWidget(slice_view_name)
        if slice_widget is None:
            continue
        slice_widget.mrmlSliceNode().JumpSliceByCentering(*center_ras)


def _show_loaded_volume_in_views(node, kind: VolumeKind) -> None:
    import slicer

    application_logic = slicer.app.applicationLogic()
    selection_node = application_logic.GetSelectionNode()

    if kind == "labelmap":
        selection_node.SetReferenceActiveLabelVolumeID(node.GetID())
        slicer.util.setSliceViewerLayers(label=node, fit=True, rotateToVolumePlane=True)
    else:
        selection_node.SetReferenceActiveVolumeID(node.GetID())
        slicer.util.setSliceViewerLayers(background=node, fit=True, rotateToVolumePlane=True)

    application_logic.PropagateVolumeSelection(0)
    application_logic.FitSliceToAll()
    _center_slice_views_on_node(node)


def inspect_mla_file(file_path: str) -> ResolvedMlaLayout:
    """Inspect an `.mla` file quickly for reader confidence decisions."""
    mlarray = _runtime_mlarray(auto_install=False)
    image = mlarray.MLArray.open(file_path, mmap_mode="r")
    try:
        return resolve_supported_mla_layout(
            image.shape,
            image.meta.spatial.axis_labels,
            is_seg=bool(image.meta.is_seg),
        )
    finally:
        image.close()


def load_mla_volume(file_path: str, *, explicit_name: str | None = None):
    """Load an `.mla` file into the Slicer scene as a volume node."""
    import slicer

    mlarray = _runtime_mlarray(auto_install=True)
    image = mlarray.MLArray.open(file_path, mmap_mode="r")
    try:
        layout = resolve_supported_mla_layout(
            image.shape,
            image.meta.spatial.axis_labels,
            is_seg=bool(image.meta.is_seg),
        )
        geometry = mlarray_geometry_to_slicer(
            layout.metadata_spatial_labels,
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction,
        )
        ijk_to_ras = _vtk_matrix4x4_from_numpy(ijk_to_ras_matrix_from_geometry(geometry))
        voxel_array = np.asarray(image.to_numpy())
        slicer_array = np.asarray(mla_array_to_slicer(voxel_array, layout))

        if layout.kind == "labelmap":
            node_class_name = "vtkMRMLLabelMapVolumeNode"
        elif layout.kind == "vector":
            node_class_name = "vtkMRMLVectorVolumeNode"
        else:
            node_class_name = "vtkMRMLScalarVolumeNode"

        loaded_node = slicer.util.addVolumeFromArray(
            slicer_array,
            ijkToRAS=ijk_to_ras,
            name=_get_base_name(file_path, explicit_name),
            nodeClassName=node_class_name,
        )
        display_node = loaded_node.GetDisplayNode()
        if display_node is not None and hasattr(display_node, "AutoWindowLevelOn"):
            display_node.AutoWindowLevelOn()
        _store_mlarray_node_attributes(
            loaded_node,
            file_path=file_path,
            kind=layout.kind,
            source_meta=dict(image.meta.source.to_plain() or {}),
            extra_meta=dict(image.meta.extra.to_plain() or {}),
        )
        _show_loaded_volume_in_views(loaded_node, layout.kind)
        return loaded_node
    finally:
        image.close()


def can_write_slicer_node(node) -> bool:
    """Return True when the writer supports exporting this node."""
    class_name = node.GetClassName()
    return class_name in {"vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode"}


def write_mla_volume(node, file_path: str) -> None:
    """Export a supported Slicer volume node to `.mla`."""
    import vtk
    import slicer

    if not can_write_slicer_node(node):
        raise UnsupportedLayoutError(
            f"Unsupported node type for MLArray export: {node.GetClassName()}."
        )

    is_labelmap = node.IsA("vtkMRMLLabelMapVolumeNode")
    volume_kind: VolumeKind = "labelmap" if is_labelmap else "scalar"
    voxel_array = np.array(slicer.util.arrayFromVolume(node), copy=True)
    voxel_array = export_scalar_like_array_to_mla(voxel_array)

    direction_matrix = vtk.vtkMatrix3x3()
    node.GetIJKToRASDirectionMatrix(direction_matrix)
    spacing, origin, direction = slicer_geometry_to_mlarray(
        spacing_xyz=node.GetSpacing(),
        origin_xyz=node.GetOrigin(),
        direction_xyz=_numpy_from_vtk_matrix3x3(direction_matrix),
        metadata_spatial_labels=MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    )

    source_meta = _get_json_attribute(node, ATTR_SOURCE_META_JSON)
    extra_meta = _get_json_attribute(node, ATTR_EXTRA_META_JSON)
    slicer_extra = dict(extra_meta.get("slicer_mlarray", {}))
    slicer_extra.update(
        {
            "coordinate_system": "RAS",
            "saved_node_class": node.GetClassName(),
        }
    )
    extra_meta["slicer_mlarray"] = slicer_extra

    mlarray = _runtime_mlarray(auto_install=True)
    meta = mlarray.Meta(
        source=source_meta,
        extra=extra_meta,
        is_seg=is_labelmap,
    )
    image = mlarray.MLArray(
        voxel_array,
        spacing=spacing,
        origin=origin,
        direction=direction,
        axis_labels=build_export_axis_labels(volume_kind),
        meta=meta,
    )
    image.save(file_path)


def reader_confidence_for_path(file_path: str) -> float:
    """Determine reader confidence for an `.mla` path."""
    if not file_path.lower().endswith(".mla"):
        return 0.0

    try:
        inspect_mla_file(file_path)
    except DependencyError:
        return 0.65
    except Exception:
        return 0.15
    return 0.85


def writer_confidence_for_node(node) -> float:
    """Determine writer confidence for a Slicer node."""
    if not can_write_slicer_node(node):
        return 0.0
    if node.GetAttribute(ATTR_FILE_CATEGORY) == FILE_CATEGORY_VALUE:
        return 0.85
    return 0.3
