"""Slicer runtime bridge for MLArray bounding-box IO.

Loads ``meta.bbox`` from an ``.mla`` file as ``vtkMRMLMarkupsROINode`` nodes,
and exports selected ROI nodes back to ``MetaBbox`` metadata in a new or
existing ``.mla`` file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bbox_conversion import bbox_vox_to_slicer_roi, slicer_roi_to_bbox_vox
from .conversion import (
    MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    _metadata_permutation_to_xyz,
    resolve_world_coord_system,
)
from .dependencies import ensure_mlarray_available
from .exceptions import UnsupportedLayoutError

# Node attribute used to flag an ROI node as originating from an MLA bbox.
ATTR_MLA_BBOX = "SlicerMLArray.BboxSource"
ATTR_MLA_BBOX_LABEL = "SlicerMLArray.BboxLabel"
ATTR_MLA_BBOX_SCORE = "SlicerMLArray.BboxScore"
ATTR_MLA_BBOX_FILE = "SlicerMLArray.BboxFile"

# Colour cycle for ROI display (RGBA 0-1 floats)
_ROI_COLORS = [
    (1.0, 0.4, 0.0, 1.0),   # orange
    (0.2, 0.6, 1.0, 1.0),   # sky blue
    (0.1, 0.8, 0.2, 1.0),   # green
    (0.9, 0.2, 0.8, 1.0),   # magenta
    (1.0, 0.9, 0.1, 1.0),   # yellow
    (0.5, 0.2, 1.0, 1.0),   # violet
    (0.0, 0.8, 0.8, 1.0),   # cyan
    (1.0, 0.2, 0.2, 1.0),   # red
]


def _runtime_mlarray(auto_install: bool = False):
    return ensure_mlarray_available(auto_install=auto_install)


def has_bboxes(image) -> bool:
    """Return ``True`` when *image* carries at least one bounding box."""
    try:
        bbox = image.meta.bbox
        return bbox.bboxes is not None and len(bbox.bboxes) > 0
    except Exception:
        return False


def _affine_from_image(image) -> np.ndarray:
    """Return a 4×4 numpy affine for *image*, falling back to identity.

    ``MLArray.affine`` returns ``None`` for array-less (bbox-only) files even
    when ``meta.spatial.affine`` is populated, so we check that field directly
    before trying spacing/origin/direction.
    """
    if image.affine is not None:
        return np.asarray(image.affine, dtype=float)
    # bbox-only files: affine lives in meta.spatial.affine, not image.affine
    meta_affine = image.meta.spatial.affine
    if meta_affine is not None:
        return np.asarray(meta_affine, dtype=float)
    # Build from spacing/origin/direction
    spacing = image.spacing or [1.0, 1.0, 1.0]
    origin = image.origin or [0.0, 0.0, 0.0]
    direction = image.direction
    mat = np.eye(4, dtype=float)
    if direction is not None:
        mat[:3, :3] = np.asarray(direction, dtype=float) * np.asarray(spacing, dtype=float)
    else:
        mat[:3, :3] = np.diag(spacing)
    mat[:3, 3] = origin
    return mat


def _metadata_spatial_labels_from_image(image) -> tuple[str, ...]:
    """Resolve metadata spatial labels for *image* (the order stored in meta)."""
    axis_labels = image.meta.spatial.axis_labels
    if axis_labels is None:
        return MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX
    return tuple(str(label) for label in axis_labels)


def _coord_system_from_image(image) -> str:
    """Resolve the stored coord_system for *image*, defaulting to RAS."""
    cs = image.meta.spatial.coord_system
    return resolve_world_coord_system(cs, default="RAS")


def load_mla_rois(
    file_path: str,
    *,
    explicit_prefix: str | None = None,
    affine_override: np.ndarray | None = None,
    metadata_spatial_labels_override: tuple[str, ...] | None = None,
    coord_system_override: str | None = None,
) -> list:
    """Load bounding boxes from an ``.mla`` file as Slicer ROI nodes.

    Each bounding box in ``meta.bbox`` becomes one
    ``vtkMRMLMarkupsROINode`` added to the Slicer scene.  The node is
    named ``<prefix>_<label>`` when a label is available, or
    ``<prefix>_<i>`` otherwise.

    Args:
        file_path: Path to a ``.mla`` file that contains bbox metadata.
        explicit_prefix: Name prefix for the created nodes.  Defaults to
            the file stem.
        affine_override: Override the affine read from the file.
        metadata_spatial_labels_override: Override the spatial labels.
        coord_system_override: Override the coordinate system.

    Returns:
        List of created ``vtkMRMLMarkupsROINode`` instances.
    """
    import slicer

    mlarray = _runtime_mlarray()
    image = mlarray.MLArray.open(file_path, mmap_mode="r")
    try:
        bbox_meta = image.meta.bbox
        if bbox_meta.bboxes is None or len(bbox_meta.bboxes) == 0:
            return []

        affine = affine_override if affine_override is not None else _affine_from_image(image)
        spatial_labels = metadata_spatial_labels_override or _metadata_spatial_labels_from_image(image)
        coord_system = coord_system_override or _coord_system_from_image(image)

        bboxes = bbox_meta.bboxes
        labels = bbox_meta.labels or [None] * len(bboxes)
        scores = bbox_meta.scores or [None] * len(bboxes)

        import os
        prefix = explicit_prefix or os.path.splitext(os.path.basename(file_path))[0]

        created_nodes: list = []
        for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            center_ras, size_mm, obj_to_world = bbox_vox_to_slicer_roi(
                bbox, affine, spatial_labels, coord_system
            )

            # Build human-readable node name
            label_str = str(label) if label is not None else str(i)
            node_name = slicer.mrmlScene.GenerateUniqueName(f"{prefix}_{label_str}")

            roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", node_name)

            # Center and size
            roi_node.SetCenter(center_ras.tolist())
            roi_node.SetSize(size_mm.tolist())
            # Orientation: GetObjectToNodeMatrix returns the live 3×3 vtkMatrix3x3;
            # modify it in-place then call Modified() to propagate the change.
            # (There is no SetObjectToWorldMatrix setter in vtkMRMLMarkupsROINode.)
            object_to_node = roi_node.GetObjectToNodeMatrix()
            for r in range(3):
                for c in range(3):
                    object_to_node.SetElement(r, c, float(obj_to_world[r, c]))
            roi_node.Modified()

            # Display settings
            color_rgba = _ROI_COLORS[i % len(_ROI_COLORS)]
            display_node = roi_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetSelectedColor(*color_rgba[:3])
                display_node.SetColor(*color_rgba[:3])
                display_node.SetOpacity(0.2)
                display_node.SetFillVisibility(True)
                display_node.SetHandlesInteractive(False)
                # Show floating label with class and score in 3D viewer
                display_node.SetPropertiesLabelVisibility(True)

            # Store source attributes for round-trip export
            roi_node.SetAttribute(ATTR_MLA_BBOX, "1")
            roi_node.SetAttribute(ATTR_MLA_BBOX_FILE, str(file_path))
            if label is not None:
                roi_node.SetAttribute(ATTR_MLA_BBOX_LABEL, str(label))
            if score is not None:
                roi_node.SetAttribute(ATTR_MLA_BBOX_SCORE, str(score))

            created_nodes.append(roi_node)

        return created_nodes
    finally:
        image.close()


def write_mla_bboxes(
    roi_nodes: list,
    file_path: str,
    *,
    affine: np.ndarray | None = None,
    metadata_spatial_labels: tuple[str, ...] = MLARRAY_DEFAULT_SPATIAL_LABELS_ZYX,
    coord_system: str = "RAS",
) -> None:
    """Export Slicer ROI nodes as bounding-box metadata in a new ``.mla`` file.

    Args:
        roi_nodes: ``vtkMRMLMarkupsROINode`` instances to export.
        file_path: Destination ``.mla`` path.
        affine: 4 × 4 affine matrix (voxel → world in ``coord_system``).
            When *None*, an identity affine is used.
        metadata_spatial_labels: Axis label order to use for the stored bbox
            voxel coordinates.
        coord_system: World coordinate system of *affine*.
    """
    if not roi_nodes:
        raise UnsupportedLayoutError("No ROI nodes provided for MLArray bbox export.")

    if affine is None:
        affine = np.eye(4, dtype=float)
    else:
        affine = np.asarray(affine, dtype=float)

    mlarray = _runtime_mlarray()

    bboxes: list[list[list[float]]] = []
    labels: list[str | int | float] = []
    scores_list: list[float | None] = []
    has_scores = False

    for roi_node in roi_nodes:
        # Read center and size from ROI node
        center = [0.0, 0.0, 0.0]
        roi_node.GetCenter(center)
        size = [0.0, 0.0, 0.0]
        roi_node.GetSize(size)

        import vtk
        obj_to_world_vtk = vtk.vtkMatrix4x4()
        roi_node.GetObjectToWorldMatrix(obj_to_world_vtk)
        obj_to_world_np = np.eye(4, dtype=float)
        for r in range(4):
            for c in range(4):
                obj_to_world_np[r, c] = obj_to_world_vtk.GetElement(r, c)

        bbox_per_axis = slicer_roi_to_bbox_vox(
            center_ras=center,
            size_mm=size,
            obj_to_world=obj_to_world_np,
            affine=affine,
            metadata_spatial_labels=metadata_spatial_labels,
            coord_system=coord_system,
        )
        bboxes.append(bbox_per_axis)

        # Recover label: prefer stored attribute, fall back to node name
        label_attr = roi_node.GetAttribute(ATTR_MLA_BBOX_LABEL)
        label = label_attr if label_attr else roi_node.GetName()
        labels.append(label)

        score_attr = roi_node.GetAttribute(ATTR_MLA_BBOX_SCORE)
        if score_attr is not None:
            try:
                scores_list.append(float(score_attr))
                has_scores = True
            except ValueError:
                scores_list.append(None)
        else:
            scores_list.append(None)

    scores = [float(s) for s in scores_list if s is not None] if has_scores else None

    meta = mlarray.Meta(
        spatial={
            "affine": affine.tolist(),
            "axis_labels": list(metadata_spatial_labels),
            "coord_system": coord_system,
        },
        bbox=mlarray.MetaBbox(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        ),
    )

    mlarray.MLArray(meta=meta).save(file_path)
