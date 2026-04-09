"""Slicer runtime bridge for MLArray file IO."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .bbox_io import (
    has_bboxes,
    load_mla_rois,
    _affine_from_image,
    _metadata_spatial_labels_from_image,
    _coord_system_from_image,
)
from .conversion import (
    MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    ResolvedMlaLayout,
    VolumeKind,
    build_export_axis_labels,
    export_scalar_like_array_to_mla,
    ijk_to_ras_matrix_from_geometry,
    mlarray_affine_to_slicer,
    mlarray_geometry_to_slicer,
    mla_array_to_slicer,
    resolve_supported_mla_layout,
    resolve_world_coord_system,
    slicer_geometry_to_mlarray,
)
from .dependencies import ensure_mlarray_available
from .exceptions import DependencyError, UnsupportedLayoutError

FILE_TYPE = "MLArrayFile"
FILE_DESCRIPTION = "MLArray volume"
FILE_EXTENSION_FILTER = "MLArray volume (*.mla)"
FILE_WRITER_EXTENSION_FILTER = "MLArray volume (.mla)"

ATTR_SOURCE_META_JSON = "SlicerMLArray.SourceMetaJSON"
ATTR_EXTRA_META_JSON = "SlicerMLArray.ExtraMetaJSON"
ATTR_SOURCE_PATH = "SlicerMLArray.SourcePath"
ATTR_KIND = "SlicerMLArray.Kind"
ATTR_FILE_CATEGORY = "SlicerMLArray.FileCategory"
FILE_CATEGORY_VALUE = "mla"
LEGACY_COORD_SYSTEM_KEY = "coordinate_system"


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


def _numpy_from_vtk_direction_matrix4x4(matrix) -> np.ndarray:
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


def normalize_mla_path(file_path: str) -> str:
    """Ensure that a save path ends with `.mla`."""
    normalized = str(file_path)
    if normalized.lower().endswith(".mla"):
        return normalized
    return normalized + ".mla"


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


def _coord_system_from_mlarray(image) -> str:
    coord_system = image.meta.spatial.coord_system
    if coord_system is not None:
        return resolve_world_coord_system(coord_system)

    # Backward compatibility with older files written by this plugin before
    # mlarray gained the dedicated spatial coord_system field.
    legacy_coord_system = image.meta.extra.to_plain().get("slicer_mlarray", {}).get(
        LEGACY_COORD_SYSTEM_KEY
    )
    if legacy_coord_system is not None:
        return resolve_world_coord_system(legacy_coord_system)

    # Older plugin versions implicitly wrote RAS-space geometry.
    return "RAS"


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


def inspect_mla_file(file_path: str) -> ResolvedMlaLayout | None:
    """Inspect an `.mla` file quickly for reader confidence decisions.

    Returns ``None`` for bbox-only files (no array data) that only carry
    bounding-box metadata — these are still accepted by the reader.
    """
    mlarray = _runtime_mlarray(auto_install=False)
    image = mlarray.MLArray.open(file_path, mmap_mode="r")
    try:
        if image.shape is None:
            # Bbox-only file: valid iff it carries at least one bbox.
            if has_bboxes(image):
                return None
            raise UnsupportedLayoutError(
                "MLArray file has no array data and no bounding-box metadata."
            )
        return resolve_supported_mla_layout(
            image.shape,
            image.meta.spatial.axis_labels,
            is_seg=bool(image.meta.is_seg),
        )
    finally:
        image.close()


def load_mla_volume(file_path: str, *, explicit_name: str | None = None):
    """Load an `.mla` file into the Slicer scene.

    * If the file contains **array data**, it is loaded as a volume node
      (``vtkMRMLScalarVolumeNode``, ``vtkMRMLLabelMapVolumeNode``, or
      ``vtkMRMLVectorVolumeNode``).
    * If the file contains **bounding-box metadata**, each bbox is loaded
      as a ``vtkMRMLMarkupsROINode`` with a floating class/score label.
    * Both can be present simultaneously — in that case both are loaded.

    Returns the volume node (or ``None`` for bbox-only files).
    """
    import slicer

    mlarray = _runtime_mlarray(auto_install=True)
    image = mlarray.MLArray.open(file_path, mmap_mode="r")
    try:
        _has_bboxes = has_bboxes(image)
        _has_array = image.shape is not None

        loaded_node = None

        if _has_array:
            layout = resolve_supported_mla_layout(
                image.shape,
                image.meta.spatial.axis_labels,
                is_seg=bool(image.meta.is_seg),
            )
            coord_system = _coord_system_from_mlarray(image)
            if image.meta.spatial.affine is not None:
                ijk_to_ras_array = mlarray_affine_to_slicer(
                    layout.metadata_spatial_labels,
                    affine=image.affine,
                    coord_system=coord_system,
                )
            else:
                geometry = mlarray_geometry_to_slicer(
                    layout.metadata_spatial_labels,
                    spacing=image.spacing,
                    origin=image.origin,
                    direction=image.direction,
                    coord_system=coord_system,
                )
                ijk_to_ras_array = ijk_to_ras_matrix_from_geometry(geometry)
            ijk_to_ras = _vtk_matrix4x4_from_numpy(ijk_to_ras_array)
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

        if _has_bboxes:
            # Pass geometry from the image so bbox_io doesn't re-open the file.
            affine = _affine_from_image(image)
            spatial_labels = _metadata_spatial_labels_from_image(image)
            coord_system_bbox = _coord_system_from_image(image)
            import os
            prefix = explicit_name or os.path.splitext(os.path.basename(file_path))[0]
            load_mla_rois(
                file_path,
                explicit_prefix=prefix,
                affine_override=affine,
                metadata_spatial_labels_override=spatial_labels,
                coord_system_override=coord_system_bbox,
            )

        if not _has_array and not _has_bboxes:
            raise UnsupportedLayoutError(
                "MLArray file has no array data and no bounding-box metadata."
            )

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

    if node is None:
        raise UnsupportedLayoutError("No MRML node was provided for MLArray export.")

    if not can_write_slicer_node(node):
        raise UnsupportedLayoutError(
            f"Unsupported node type for MLArray export: {node.GetClassName()}."
        )

    is_labelmap = bool(node.IsA("vtkMRMLLabelMapVolumeNode"))
    volume_kind: VolumeKind = "labelmap" if is_labelmap else "scalar"
    voxel_array = np.array(slicer.util.arrayFromVolume(node), copy=True)
    voxel_array = export_scalar_like_array_to_mla(voxel_array)

    direction_matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASDirectionMatrix(direction_matrix)
    spacing, origin, direction = slicer_geometry_to_mlarray(
        spacing_xyz=node.GetSpacing(),
        origin_xyz=node.GetOrigin(),
        direction_xyz=_numpy_from_vtk_direction_matrix4x4(direction_matrix),
        metadata_spatial_labels=MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
        coord_system="RAS",
    )

    source_meta = _get_json_attribute(node, ATTR_SOURCE_META_JSON)
    extra_meta = _get_json_attribute(node, ATTR_EXTRA_META_JSON)
    slicer_extra = dict(extra_meta.get("slicer_mlarray", {}))
    slicer_extra.update(
        {
            LEGACY_COORD_SYSTEM_KEY: "RAS+",
            "saved_node_class": node.GetClassName(),
        }
    )
    extra_meta["slicer_mlarray"] = slicer_extra

    mlarray = _runtime_mlarray(auto_install=True)
    meta = mlarray.Meta(
        source=source_meta,
        extra=extra_meta,
        spatial={"coord_system": "RAS+"},
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
    image.save(normalize_mla_path(file_path))


def reader_confidence_for_path(file_path: str) -> float:
    """Determine reader confidence for an `.mla` path.

    Accepts both array-backed files and bbox-only files (``inspect_mla_file``
    returns ``None`` for the latter but does not raise).
    """
    if not file_path.lower().endswith(".mla"):
        return 0.0

    try:
        inspect_mla_file(file_path)  # returns None for bbox-only — that's fine
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
