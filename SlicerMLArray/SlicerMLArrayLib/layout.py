"""Which Slicer node(s) an MLArray file becomes.

Two independent choices:

- the **element**: what one 2-D/3-D image is (``scalar`` volume, ``segmentation``, ``labelmap`` volume);
- the **extra axis** (at most one non-space axis): ``vector`` (one vector volume), ``channels`` (one node per
  index), ``sequence`` (a Slicer sequence, for a time axis) or ``label_channels`` (one segment per channel,
  ``values.label_layout: channel``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .errors import UnsupportedFile
from .geometry import space_axis_indices

COLOUR_LABELS = ({"red", "green", "blue"}, {"red", "green", "blue", "alpha"},
                 {"r", "g", "b"}, {"r", "g", "b", "a"})


@dataclass
class Layout:
    element: str                       #: scalar | segmentation | labelmap
    extra: Optional[str] = None        #: None | vector | channels | sequence | label_channels
    extra_axis: Optional[int] = None   #: array index of the extra axis
    boxes_only: bool = False           #: an empty array that only carries boxes
    notes: List[str] = field(default_factory=list)


def has_boxes(meta) -> bool:
    boxes = (meta.profiles.get("ml") or {}).get("boxes") or {}
    return bool(boxes.get("items"))


def layout(meta, shape, labels_as: str = "segmentation") -> Layout:
    """Decide the nodes for an array of ``shape`` with metadata ``meta``.

    ``labels_as``: ``segmentation`` (default) or ``labelmap`` for ``values.kind == "label"``.
    """
    if shape is not None and any(int(n) == 0 for n in shape):
        if has_boxes(meta):
            return Layout("scalar", boxes_only=True)
        raise UnsupportedFile("the array is empty and the file has no boxes")
    space_axes = space_axis_indices(meta)
    if len(space_axes) not in (2, 3):
        raise UnsupportedFile(f"the array has {len(space_axes)} space axes; Slicer shows 2-D and 3-D images")
    others = [i for i in range(len(meta.axes)) if i not in space_axes]
    if len(others) > 1:
        names = ", ".join(f"{meta.axes[i].name} ({meta.axes[i].type})" for i in others)
        raise UnsupportedFile(f"the array has {len(others)} non-space axes ({names}); Slicer shows at most one")
    values = meta.values
    notes: List[str] = []
    if values.kind == "label":
        element = "labelmap" if labels_as == "labelmap" else "segmentation"
    elif values.kind == "instance":
        element = "labelmap"
        notes.append("instance ids are shown as a labelmap volume")
    else:
        element = "scalar"
    if not others:
        if values.label_layout == "channel":
            raise UnsupportedFile("values.label_layout is 'channel' but the array has no channel axis")
        return Layout(element, notes=notes)
    axis_index = others[0]
    axis = meta.axes[axis_index]
    if values.kind == "label" and values.label_layout == "channel":
        return Layout("segmentation", "label_channels", axis_index, notes=notes)
    if axis.type == "time":
        return Layout(element, "sequence", axis_index, notes=notes)
    if element == "scalar" and (axis.type == "component" or _is_colour(axis)):
        return Layout("scalar", "vector", axis_index, notes=notes)
    return Layout(element, "channels", axis_index, notes=notes)


def _is_colour(axis) -> bool:
    labels = {str(v).lower() for v in (axis.labels or ())}
    return axis.type == "channel" and labels in COLOUR_LABELS


def index_name(axis, index: int) -> str:
    """Display name of one index of a non-space axis: its label, its coordinate, or ``<axis> <index>``."""
    if axis.labels:
        return str(axis.labels[index])
    if axis.coordinates:
        return f"{axis.name} {axis.coordinates[index]:g}{(' ' + axis.unit) if axis.unit else ''}"
    if axis.step is not None:
        value = (axis.start or 0.0) + index * axis.step
        return f"{axis.name} {value:g}{(' ' + axis.unit) if axis.unit else ''}"
    return f"{axis.name} {index}"


def index_value(axis, index: int) -> float:
    """Coordinate of one index (time axis of a sequence)."""
    if axis.coordinates:
        return float(axis.coordinates[index])
    if axis.step is not None:
        return float((axis.start or 0.0) + index * axis.step)
    return float(index)
