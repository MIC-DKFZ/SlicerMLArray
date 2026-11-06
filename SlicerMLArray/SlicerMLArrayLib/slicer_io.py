"""MLArray files <-> Slicer nodes (needs Slicer)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .boxes import Roi, box_spec, ijk_axis_names, index_box_from_roi, roi_from_index_box, roi_from_space_box, \
    space_box_from_roi
from .dependencies import ensure_mlarray
from .errors import UnsupportedFile
from .export import edited_meta, new_meta, values_dict
from .geometry import grid, patient_space, same
from .layout import index_name, index_value, layout
from .values import Label, background, box_name, label_table, physical, smallest_label_dtype, stored, table_dict

import slicer
import vtk

TOOL = f"SlicerMLArray {__version__}"
ATTR_PATH = "MLArray.Path"              #: file the node was loaded from
ATTR_STAMP = "MLArray.FileStamp"        #: size and mtime of that file at load
ATTR_META = "MLArray.Meta"              #: the file's metadata without source/user (JSON), if the file is gone
ATTR_RECORD = "MLArray.Record"          #: how the node was made from the file (JSON)
ATTR_BOX = "MLArray.Box"                #: an ROI's box item without min/max (JSON)
TAG_VALUE = "MLArray.Value"             #: segment tag: label value (or channel index)
TAG_CODE = "MLArray.Code"               #: segment tag: ontology code (JSON)
TAG_EXTRA = "MLArray.Extra"             #: segment tag: other label table fields (JSON)
SETTING_LABELS = "SlicerMLArray/LabelsAs"
VOLUME_CLASSES = ("vtkMRMLScalarVolumeNode", "vtkMRMLLabelMapVolumeNode", "vtkMRMLVectorVolumeNode")


# --------------------------------------------------------------------------------------------- dependencies


def mlarray(install: bool = True):
    if not install:
        return ensure_mlarray()
    return ensure_mlarray(ask=lambda message: slicer.util.confirmYesNoDisplay(message, windowTitle="MLArray"),
                          install=slicer.util.pip_install)


def labels_as() -> str:
    import qt

    value = str(qt.QSettings().value(SETTING_LABELS, "segmentation"))
    return value if value in ("segmentation", "labelmap") else "segmentation"


def set_labels_as(value: str) -> None:
    import qt

    qt.QSettings().setValue(SETTING_LABELS, value)


# --------------------------------------------------------------------------------------------- loading


def load(path: str, name: Optional[str] = None) -> Tuple[List, List[str]]:
    """Load an ``.mla`` file; returns the created nodes and notes for the user."""
    ml = mlarray()
    image = ml.MLArray.open(path, mmap_mode="r")
    try:
        meta = image.meta
        shape = tuple(int(n) for n in image.shape)
        plan = layout(meta, shape, labels_as())
        base = name or _stem(path)
        g = grid(meta)
        notes = plan.notes + g.notes
        common = {ATTR_PATH: os.path.abspath(path), ATTR_STAMP: _stamp(path),
                  ATTR_META: json.dumps(meta.to_dict(), default=str)}
        nodes: List = []
        if not plan.boxes_only:
            array = np.asarray(image.to_numpy())
            nodes += _load_array(array, meta, plan, g, base, common)
        if box_spec(meta):
            nodes += _load_boxes(meta, g, shape, base, common)
        _show(nodes)
        return nodes, notes
    finally:
        image.close()


def _stem(path: str) -> str:
    name = os.path.basename(path)
    return name[:-4] if name.lower().endswith(".mla") else name


def _stamp(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _record(g, plan, shape, **extra) -> Dict:
    return {"ijk_to_ras": g.ijk_to_ras.tolist(), "spatial_axes": list(g.spatial_axes), "space": g.space,
            "is_2d": g.is_2d, "extra": plan.extra, "extra_axis": plan.extra_axis, "shape": list(shape), **extra}


def _tag(node, common: Dict, record: Dict) -> None:
    for key, value in common.items():
        node.SetAttribute(key, value)
    node.SetAttribute(ATTR_RECORD, json.dumps(record))


def _kji(array: np.ndarray, g, extra_axis: Optional[int]) -> np.ndarray:
    """Array with the extra axis first (if any), then K, J, I (a size-1 K inserted for 2-D images)."""
    if extra_axis is not None:
        array = np.moveaxis(array, extra_axis, 0)
    if g.is_2d:
        array = array[..., None, :, :]
    return array


def _vtk_matrix(matrix: np.ndarray):
    out = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            out.SetElement(r, c, float(matrix[r, c]))
    return out


def _numpy_matrix(vtk_matrix) -> np.ndarray:
    return np.array([[vtk_matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


def ijk_to_ras(node) -> np.ndarray:
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return _numpy_matrix(matrix)


def _volume(array: np.ndarray, g, name: str, class_name: str = "vtkMRMLScalarVolumeNode"):
    return slicer.util.addVolumeFromArray(np.ascontiguousarray(array), ijkToRAS=_vtk_matrix(g.ijk_to_ras),
                                          name=slicer.mrmlScene.GenerateUniqueName(name), nodeClassName=class_name)


def _load_array(array, meta, plan, g, base: str, common: Dict) -> List:
    data = _kji(array, g, plan.extra_axis)
    values = meta.values
    nodes = []
    if plan.extra == "label_channels":
        node = _segmentation_from_channels(data, meta, g, base)
        _tag(node, common, _record(g, plan, array.shape, element="segmentation"))
        return [node]
    if plan.extra == "vector":
        physical_values, scaling = physical(np.moveaxis(data, 0, -1), values)
        node = _volume(physical_values, g, base, "vtkMRMLVectorVolumeNode")
        _tag(node, common, _record(g, plan, array.shape, element="scalar", scaling=scaling))
        return [node]
    frames = [(None, data)] if plan.extra is None else [(i, data[i]) for i in range(data.shape[0])]
    axis = meta.axes[plan.extra_axis] if plan.extra_axis is not None else None
    for index, frame in frames:
        name = base if index is None else f"{base} {index_name(axis, index)}"
        node, extra_record = _element(frame, meta, plan.element, g, name)
        _tag(node, common, _record(g, plan, array.shape, element=plan.element, index=index, **extra_record))
        nodes.append(node)
    if plan.extra == "sequence":
        nodes = [_sequence(nodes, axis, base)]
    return nodes


def _element(frame: np.ndarray, meta, element: str, g, name: str):
    """One 3-D image as a scalar volume, labelmap volume or segmentation; and what to record for export."""
    values = meta.values
    if element == "scalar":
        data, scaling = physical(frame, values)
        node = _volume(data, g, name)
        return node, {"scaling": scaling}
    labels = frame if np.issubdtype(frame.dtype, np.integer) else np.rint(frame).astype(np.int32)
    if labels.dtype in (np.uint64, np.int64):
        labels = labels.astype(np.int32)
    present = [int(v) for v in np.unique(labels) if v != 0]
    table = label_table(values, present)
    labelmap = _volume(labels, g, name, "vtkMRMLLabelMapVolumeNode")
    colours = _colour_table(table, name)
    labelmap.GetDisplayNode().SetAndObserveColorNodeID(colours.GetID())
    record = {"dtype": str(frame.dtype)}
    if element == "labelmap":
        return labelmap, record
    segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                      slicer.mrmlScene.GenerateUniqueName(name))
    segmentation.SetReferenceImageGeometryParameterFromVolumeNode(labelmap)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentation, "")
    by_name: Dict[int, Label] = {label.value: label for label in table}
    seg = segmentation.GetSegmentation()
    for k in range(seg.GetNumberOfSegments()):
        segment = seg.GetNthSegment(k)
        label = by_name.get(int(segment.GetLabelValue()))
        if label is None:
            continue
        _describe_segment(segment, label)
    slicer.mrmlScene.RemoveNode(labelmap)
    slicer.mrmlScene.RemoveNode(colours)
    segmentation.CreateDefaultDisplayNodes()
    return segmentation, record


def _describe_segment(segment, label: Label) -> None:
    segment.SetName(label.name)
    segment.SetColor(*[c / 255.0 for c in label.color])
    segment.SetTag(TAG_VALUE, str(label.value))
    if label.code:
        segment.SetTag(TAG_CODE, json.dumps(label.code))
        entry = _terminology(label.code)
        if entry:
            segment.SetTag(segment.GetTerminologyEntryTagName(), entry)
    if label.extra:
        segment.SetTag(TAG_EXTRA, json.dumps(label.extra))


def _terminology(code: Dict) -> Optional[str]:
    """Slicer terminology string with the code as type (no category: MLArray stores none)."""
    try:
        scheme, value, meaning = code["scheme"], code["value"], code.get("meaning", code["value"])
    except (KeyError, TypeError):
        return None
    clean = lambda s: str(s).replace("~", " ").replace("^", " ")
    return f"Segmentation category and type - DICOM master list~^^~{clean(scheme)}^{clean(value)}^{clean(meaning)}" \
           "~^^~Anatomic codes - DICOM master list~^^~^^"


def _colour_table(table: List[Label], name: str):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", slicer.mrmlScene.GenerateUniqueName(f"{name} labels"))
    node.SetTypeToUser()
    size = (max(label.value for label in table) + 1) if table else 1
    node.SetNumberOfColors(size)
    node.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
    for v in range(1, size):
        node.SetColor(v, "", 0.0, 0.0, 0.0, 0.0)
        node.SetColorDefined(v, False)
    for label in table:
        node.SetColor(label.value, label.name, *[c / 255.0 for c in label.color], 1.0)
    node.SetAttribute("Category", "MLArray")
    return node


def _segmentation_from_channels(data: np.ndarray, meta, g, base: str):
    """``label_layout: channel``: channel i (non-zero voxels) becomes segment i."""
    reference = _volume(np.zeros(data.shape[1:], np.uint8), g, base + " reference", "vtkMRMLLabelMapVolumeNode")
    segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", slicer.mrmlScene.GenerateUniqueName(base))
    segmentation.SetReferenceImageGeometryParameterFromVolumeNode(reference)
    table = {label.value: label for label in label_table(meta.values, list(range(1, data.shape[0])))}
    for channel in range(data.shape[0]):
        label = table.get(channel) or Label(channel, f"label {channel}", (128, 128, 128))
        segment_id = segmentation.GetSegmentation().AddEmptySegment("", label.name)
        slicer.util.updateSegmentBinaryLabelmapFromArray((data[channel] != 0).astype(np.uint8), segmentation,
                                                        segment_id, reference)
        _describe_segment(segmentation.GetSegmentation().GetSegment(segment_id), label)
    slicer.mrmlScene.RemoveNode(reference)
    segmentation.CreateDefaultDisplayNodes()
    return segmentation


def _sequence(nodes: List, axis, base: str):
    """Frames of a time axis as a sequence with a browser; the proxy node shows the current frame."""
    sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", slicer.mrmlScene.GenerateUniqueName(base))
    sequence.SetIndexName(axis.name)
    sequence.SetIndexUnit(axis.unit or "")
    sequence.SetIndexType(sequence.NumericIndex)
    for index, node in enumerate(nodes):
        sequence.SetDataNodeAtValue(node, f"{index_value(axis, index):g}")
    record = json.loads(nodes[0].GetAttribute(ATTR_RECORD))
    for key in (ATTR_PATH, ATTR_STAMP, ATTR_META):
        sequence.SetAttribute(key, nodes[0].GetAttribute(key))
    record["index"] = None
    sequence.SetAttribute(ATTR_RECORD, json.dumps(record))
    for node in nodes:
        slicer.mrmlScene.RemoveNode(node)
    browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", slicer.mrmlScene.GenerateUniqueName(base + " browser"))
    set_primary = getattr(browser, "SetAndObservePrimarySequenceNodeID", None) or browser.SetAndObserveMasterSequenceNodeID
    set_primary(sequence.GetID())
    proxy = browser.GetProxyNode(sequence)
    return proxy if proxy is not None else sequence


def _load_boxes(meta, g, shape, base: str, common: Dict) -> List:
    spec = box_spec(meta)
    frame, box_axes = spec["frame"], list(spec["axes"])
    ijk_names = ijk_axis_names(meta, g.spatial_axes)
    ijk_shape = [max(int(shape[a]), 1) for a in g.spatial_axes][::-1] + ([1] if g.is_2d else [])
    space = next((s for s in meta.spaces if s.name == frame), None)
    if frame != "index" and space is None:
        raise UnsupportedFile(f"boxes are in space {frame!r}, which the file does not define")
    classes = {label.value: label for label in label_table(meta.values)}
    hierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
    folder = hierarchy.CreateFolderItem(hierarchy.GetSceneItemID(), f"{base} boxes")
    record = {"frame": frame, "axes": box_axes, "ijk_names": ijk_names, "ijk_to_ras": g.ijk_to_ras.tolist()}
    nodes = []
    for item in spec["items"]:
        if frame == "index":
            roi = roi_from_index_box(item["min"], item["max"], box_axes, ijk_names, g.ijk_to_ras, ijk_shape)
        else:
            roi = roi_from_space_box(item["min"], item["max"], box_axes, space)
        label = item.get("label")
        cls = classes.get(int(label)) if isinstance(label, (int, float)) and float(label).is_integer() else None
        name = box_name(cls.name if cls else (str(label) if label is not None else None), label, item.get("score"))
        colour = cls.color if cls else (255, 160, 40)
        node = _roi_node(roi, slicer.mrmlScene.GenerateUniqueName(name), colour)
        _tag(node, common, record)
        node.SetAttribute(ATTR_BOX, json.dumps({k: v for k, v in dict(item).items() if k not in ("min", "max")},
                                               default=str))
        hierarchy.SetItemParent(hierarchy.GetItemByDataNode(node), folder)
        nodes.append(node)
    return nodes


def _roi_node(roi: Roi, name: str, colour):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", name)
    node.SetROIType(node.ROITypeBox)
    matrix = np.eye(4)
    matrix[:3, :3] = roi.axes
    matrix[:3, 3] = roi.center
    node.GetObjectToNodeMatrix().DeepCopy(_vtk_matrix(matrix))
    node.SetSize(*[float(v) for v in roi.size])
    node.SetCenter(*[float(v) for v in roi.center])
    display = node.GetDisplayNode()
    rgb = [c / 255.0 for c in colour]
    display.SetSelectedColor(*rgb)
    display.SetColor(*rgb)
    display.SetFillOpacity(0.05)
    display.SetHandlesInteractive(False)
    display.SetPropertiesLabelVisibility(True)
    display.SetPointLabelsVisibility(False)
    return node


def roi_of(node) -> Roi:
    m = _numpy_matrix(node.GetObjectToWorldMatrix())
    axes = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    center = [0.0, 0.0, 0.0]
    node.GetCenterWorld(center)
    size = [0.0, 0.0, 0.0]
    node.GetSizeWorld(size)
    return Roi(np.array(center), np.array(size), axes)


def _show(nodes: List) -> None:
    volumes = [n for n in nodes if n.IsA("vtkMRMLScalarVolumeNode") and not n.IsA("vtkMRMLLabelMapVolumeNode")]
    labelmaps = [n for n in nodes if n.IsA("vtkMRMLLabelMapVolumeNode")]
    if volumes or labelmaps:
        slicer.util.setSliceViewerLayers(background=volumes[0] if volumes else None,
                                         label=labelmaps[0] if labelmaps else None, fit=True)


# --------------------------------------------------------------------------------------------- saving


def can_write(node) -> bool:
    return node is not None and (node.GetClassName() in VOLUME_CLASSES or node.IsA("vtkMRMLSegmentationNode"))


def _original(node, ml) -> Tuple[Optional[dict], Optional[dict]]:
    """(file metadata incl. source and user, record) of a node loaded from ``.mla``; (None, None) otherwise."""
    record = node.GetAttribute(ATTR_RECORD)
    if not record:
        return None, None
    record = json.loads(record)
    path, stamp = node.GetAttribute(ATTR_PATH), node.GetAttribute(ATTR_STAMP)
    if path and os.path.exists(path) and _stamp(path) == stamp:
        image = ml.MLArray.open(path, mmap_mode="r")
        try:
            return image.meta.to_dict(include_lazy=True), record
        finally:
            image.close()
    return json.loads(node.GetAttribute(ATTR_META)), record


def write(node, path: str) -> List[str]:
    """Save a volume or segmentation node as ``.mla``; returns notes for the user."""
    ml = mlarray()
    original, record = _original(node, ml)
    notes: List[str] = []
    original_values = (original or {}).get("values")
    if node.IsA("vtkMRMLSegmentationNode"):
        array, matrix, labels, extra_axis = _segmentation_array(node, record, original_values)
        kind = "label"
        values = values_dict(original_values, kind, labels=labels)
        component = extra_axis is not None
    else:
        array = np.array(slicer.util.arrayFromVolume(node), copy=True)
        matrix = ijk_to_ras(node)
        component = node.IsA("vtkMRMLVectorVolumeNode")
        if node.IsA("vtkMRMLLabelMapVolumeNode"):
            table = _labels_from_colours(node, array, original_values, record)
            values = values_dict(original_values, "label", labels=table)
            array = _label_dtype(array, record)
        else:
            array, scaling = stored(array, (record or {}).get("scaling"))
            kind = (original_values or {}).get("kind", "measurement")
            kind = kind if kind in ("measurement", "probability", "vector", "other") else "measurement"
            values = values_dict(original_values, kind, scaling=scaling)
        extra_axis = (record or {}).get("extra_axis") if component else None
    if record is not None and record.get("is_2d"):
        array = array[0]                                    # drop the size-1 K of a 2-D image
    if component:
        array = np.moveaxis(array, -1, extra_axis if (record and extra_axis is not None) else -1)
    meta = None
    if record is not None and original is not None:
        try:
            meta = edited_meta(ml.Meta, ml.Space, original, record, array.shape, matrix, values, TOOL)
        except Exception as error:          # the original metadata no longer fits (e.g. another shape)
            notes.append(f"original metadata not kept: {error}")
    if meta is None:
        meta = new_meta(ml.Meta, array.shape, matrix, values, TOOL, component_axis=component)
    ml.MLArray(np.ascontiguousarray(array), meta=meta).save(path)
    return notes


def _label_dtype(array: np.ndarray, record: Optional[dict]) -> np.ndarray:
    target = np.dtype((record or {}).get("dtype") or smallest_label_dtype(int(array.max()) if array.size else 0))
    if np.issubdtype(target, np.integer) and array.size:
        info = np.iinfo(target)
        if array.min() >= info.min and array.max() <= info.max:
            return array.astype(target)
    return array


def _labels_from_colours(node, array, original_values, record) -> dict:
    """Label table of a labelmap volume: the loaded table, names/colours of the colour node for other values."""
    from mlarray import Values

    original = Values.from_dict(original_values) if original_values and original_values.get("kind") == "label" else None
    table = {label.value: label for label in (label_table(original) if original else [])}
    colours = node.GetDisplayNode().GetColorNode() if node.GetDisplayNode() else None
    for value in [int(v) for v in np.unique(array) if v != 0]:
        if value in table or colours is None:
            continue
        rgba = [0.0] * 4
        colours.GetColor(value, rgba)
        name = colours.GetColorName(value) or f"label {value}"
        table[value] = Label(value, name, tuple(int(round(c * 255)) for c in rgba[:3]))
    return table_dict(list(table.values()), background(original) if original else None)


def _segmentation_array(node, record, original_values):
    """Labelmap array (or channel stack) of a segmentation on its reference grid, IJKToRAS, label table."""
    reference, temporary = _reference_volume(node, record)
    try:
        matrix = ijk_to_ras(reference)
        segmentation = node.GetSegmentation()
        ids = [segmentation.GetNthSegmentID(k) for k in range(segmentation.GetNumberOfSegments())]
        channels = record is not None and record.get("extra") == "label_channels"
        used = set()
        wanted = {}
        for segment_id in ids:
            tag = _get_tag(segmentation.GetSegment(segment_id), TAG_VALUE)
            if tag is not None and tag.isdigit() and int(tag) not in used and (channels or int(tag) > 0):
                wanted[segment_id] = int(tag)
                used.add(int(tag))
        next_value = 1
        for segment_id in ids:
            if segment_id not in wanted:
                while next_value in used:
                    next_value += 1
                wanted[segment_id] = next_value
                used.add(next_value)
        shape = tuple(int(n) for n in slicer.util.arrayFromVolume(reference).shape)
        labels: List[Label] = []
        if channels:
            out = np.zeros((max(used) + 1,) + shape, np.uint8)
        else:
            out = np.zeros(shape, smallest_label_dtype(max(used) if used else 0))
        for segment_id in ids:
            segment = segmentation.GetSegment(segment_id)
            value = wanted[segment_id]
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(node, segment_id, reference) != 0
            if channels:
                out[value][mask] = 1
            else:
                if np.any(out[mask] != 0):
                    raise UnsupportedFile(f"segment {segment.GetName()!r} overlaps another segment; an MLArray label "
                                          "map holds one label per voxel. Remove the overlap (Segment Editor, "
                                          "'Logical operators') and save again.")
                out[mask] = value
            code = _get_tag(segment, TAG_CODE)
            extra = _get_tag(segment, TAG_EXTRA)
            labels.append(Label(value, segment.GetName(), tuple(int(round(c * 255)) for c in segment.GetColor()),
                                json.loads(code) if code else None, json.loads(extra) if extra else None))
        from mlarray import Values

        original = Values.from_dict(original_values) if original_values and original_values.get("kind") == "label" else None
        table = table_dict(labels, background(original) if original else None)
        if channels:
            return np.moveaxis(out, 0, -1), matrix, table, record["extra_axis"]
        if record and record.get("dtype"):
            out = _label_dtype(out, record)
        return out, matrix, table, None
    finally:
        if temporary:
            slicer.mrmlScene.RemoveNode(reference)


def _get_tag(segment, key: str) -> Optional[str]:
    value = vtk.mutable("") if hasattr(vtk, "mutable") else None
    if value is not None and segment.GetTag(key, value):
        return str(value)
    return None


def _reference_volume(node, record):
    """The grid to rasterise a segmentation on: the loaded file's grid, the reference image, or the segments."""
    if record is not None:
        shape = record["shape"]
        k_j_i = [shape[a] for a in record["spatial_axes"]]
        if record.get("is_2d"):
            k_j_i = [1] + k_j_i
        g = type("G", (), {"ijk_to_ras": np.array(record["ijk_to_ras"])})()
        return _volume(np.zeros(k_j_i, np.uint8), g, "reference", "vtkMRMLLabelMapVolumeNode"), True
    reference = node.GetNodeReference(node.GetReferenceImageGeometryReferenceRole())
    if reference is not None and reference.IsA("vtkMRMLVolumeNode"):
        return reference, False
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "reference")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(node, labelmap)
    return labelmap, True


# --------------------------------------------------------------------------------------------- boxes


def write_boxes(roi_nodes: List, path: str, reference=None) -> List[str]:
    """Save ROIs as boxes in a box-only ``.mla`` file.

    - ``reference`` (a volume or segmentation): boxes in that image's array indices, with its geometry;
    - else ROIs loaded from one ``.mla`` file: in that file's frame (index or space), with its metadata;
    - else: in the DICOM patient frame (LPS millimetre coordinates).
    """
    ml = mlarray()
    if not roi_nodes:
        raise UnsupportedFile("no ROIs to save")
    notes: List[str] = []
    items_extra = [json.loads(n.GetAttribute(ATTR_BOX)) if n.GetAttribute(ATTR_BOX) else {} for n in roi_nodes]
    rois = [roi_of(n) for n in roi_nodes]
    if reference is not None:
        original, record = _original(reference, ml)
        matrix = ijk_to_ras(reference) if reference.IsA("vtkMRMLVolumeNode") else None
        if matrix is None:
            volume, temporary = _reference_volume(reference, record)
            matrix = ijk_to_ras(volume)
            if temporary:
                slicer.mrmlScene.RemoveNode(volume)
        if original is not None and record is not None and same(matrix, record["ijk_to_ras"]):
            axes = [a for a in original["axes"] if a.get("type") == "space"]
            names = [a["name"] for a in axes]
            spaces = original.get("spaces") or []
            values = original.get("values") if (original.get("values") or {}).get("kind") == "label" else None
        else:
            names = ["z", "y", "x"]
            axes = [{"name": n, "type": "space"} for n in names]
            spaces = [patient_space(matrix, names)]
            values = None
        ijk_names = names[::-1] + [None] * (3 - len(names))
        spec_frame, box_axes = "index", names
        convert = lambda roi: index_box_from_roi(roi, box_axes, ijk_names, matrix)
        meta_parts = {"axes": axes, "spaces": spaces, "values": values}
    else:
        paths = {n.GetAttribute(ATTR_PATH) for n in roi_nodes}
        record = roi_nodes[0].GetAttribute(ATTR_RECORD)
        if len(paths) == 1 and None not in paths and record:
            record = json.loads(record)
            original = json.loads(roi_nodes[0].GetAttribute(ATTR_META))
            spec_frame, box_axes = record["frame"], record["axes"]
            if spec_frame == "index":
                convert = lambda roi: index_box_from_roi(roi, box_axes, record["ijk_names"], record["ijk_to_ras"])
            else:
                space = ml.Space.from_dict(next(s for s in original["spaces"] if s["name"] == spec_frame))
                convert = lambda roi: space_box_from_roi(roi, box_axes, space)
            meta_parts = {"axes": original["axes"], "spaces": original.get("spaces"), "values": original.get("values")}
            user = original.get("user")
            if user:
                meta_parts["user"] = user
        else:
            names = ["z", "y", "x"]
            spaces = [patient_space(np.diag([-1.0, -1.0, 1.0, 1.0]), names)]
            space = ml.Space.from_dict(spaces[0])
            spec_frame, box_axes = "patient", ["x", "y", "z"]
            convert = lambda roi: space_box_from_roi(roi, box_axes, space)
            meta_parts = {"axes": [{"name": n, "type": "space"} for n in names], "spaces": spaces, "values": None}
    items = []
    for roi, extra in zip(rois, items_extra):
        lo, hi = convert(roi)
        items.append({"min": lo, "max": hi, **extra})
    meta_parts = {k: v for k, v in meta_parts.items() if v}
    profiles = {"ml": {"boxes": {"frame": spec_frame, "axes": list(box_axes), "items": items}}}
    image = ml.MLArray(profiles=profiles, **meta_parts)
    image.meta.add_history("edited" if items_extra and any(items_extra) else "created", tool=TOOL)
    image.save(path)
    return notes
