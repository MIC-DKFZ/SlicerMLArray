"""What the numbers mean: value scaling and label tables (spec 4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Colours for classes without one (RGB 0-255).
PALETTE = [(230, 60, 60), (60, 160, 230), (90, 200, 90), (230, 160, 40), (170, 90, 220), (40, 200, 200),
           (230, 100, 180), (160, 160, 60), (120, 120, 230), (200, 120, 90)]


@dataclass
class Label:
    value: int
    name: str
    color: Tuple[int, int, int]               #: RGB 0-255
    code: Optional[dict] = None               #: {scheme, value, meaning}
    extra: Optional[dict] = None              #: other fields of the table entry, kept for export


def physical(array: np.ndarray, values) -> Tuple[np.ndarray, Optional[dict]]:
    """Physical values ``scale * stored + offset`` and the scaling to restore on export (None if unscaled).

    Integers stay integers when the scaling maps them to integers (e.g. CT with slope 1, intercept -1024):
    the smallest integer type holding the result is used. Otherwise float32 (float64 for float64 input).
    """
    scale = 1.0 if values.scale is None else float(values.scale)
    offset = 0.0 if values.offset is None else float(values.offset)
    if scale == 1.0 and offset == 0.0:
        return array, None
    record = {"dtype": str(array.dtype), "scale": scale, "offset": offset}
    if np.issubdtype(array.dtype, np.integer) and scale.is_integer() and offset.is_integer():
        out = array.astype(np.int64) * int(scale) + int(offset)
        return out.astype(_smallest_int(out)), record
    dtype = np.float64 if array.dtype == np.float64 else np.float32
    return (array.astype(dtype) * dtype(scale) + dtype(offset)).astype(dtype), record


def stored(array: np.ndarray, record: Optional[dict]) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Back to the stored values of ``record`` if they are exact and fit its dtype; else the array unchanged.

    Returns the array to write and the (scale, offset) to write with it (None: no scaling).
    """
    if not record:
        return array, None
    dtype = np.dtype(record["dtype"])
    raw = (array.astype(np.float64) - record["offset"]) / record["scale"]
    if np.issubdtype(dtype, np.integer):
        rounded = np.rint(raw)
        info = np.iinfo(dtype)
        if raw.size and (np.abs(raw - rounded).max() > 1e-3 or rounded.min() < info.min or rounded.max() > info.max):
            return array, None
        raw = rounded
    return raw.astype(dtype), (record["scale"], record["offset"])


def _smallest_int(array: np.ndarray):
    lo, hi = (int(array.min()), int(array.max())) if array.size else (0, 0)
    for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return dtype
    return np.int64


def smallest_label_dtype(max_value: int):
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    return np.uint64


def label_table(values, present: Optional[List[int]] = None) -> List[Label]:
    """Labels of ``values.labels`` (background 0 excluded) plus default entries for values present but unnamed."""
    table: Dict[int, Label] = {}
    for key, entry in (values.labels or {}).items():
        value = int(key)
        entry = dict(entry)
        name = str(entry.pop("name", f"label {value}"))
        color = entry.pop("color", None)
        code = entry.pop("code", None)
        table[value] = Label(value, name, _rgb(color, value), dict(code) if code else None, entry or None)
    for value in present or []:
        if value not in table:
            table[value] = Label(int(value), f"label {value}", _rgb(None, int(value)))
    return [table[v] for v in sorted(table) if v != 0]


def background(values) -> Optional[dict]:
    """The table entry of value 0, kept as is on export."""
    labels = values.labels or {}
    return dict(labels["0"]) if "0" in labels else None


def table_dict(labels: List[Label], background_entry: Optional[dict] = None) -> Dict[str, dict]:
    """``values.labels`` for export."""
    out: Dict[str, dict] = {}
    if background_entry is not None:
        out["0"] = background_entry
    for label in sorted(labels, key=lambda l: l.value):
        entry = dict(label.extra or {})
        entry["name"] = label.name
        entry["color"] = [int(c) for c in label.color]
        if label.code:
            entry["code"] = dict(label.code)
        out[str(int(label.value))] = entry
    return out


def _rgb(color, value: int) -> Tuple[int, int, int]:
    if color is not None and len(color) >= 3:
        return tuple(int(c) for c in color[:3])
    return PALETTE[(value - 1) % len(PALETTE)] if value > 0 else (0, 0, 0)


def box_name(class_name: Optional[str], label, score) -> str:
    """``tumor 0.84``, ``3 0.84``, ``tumor`` or ``box``."""
    name = class_name or (str(label) if label is not None else "box")
    return f"{name} {float(score):.2f}" if score is not None else name
