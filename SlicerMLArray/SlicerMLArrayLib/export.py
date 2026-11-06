"""Metadata for saving a Slicer node as ``.mla``.

A node that was loaded from an ``.mla`` file keeps that file's metadata (axes, spaces, values, profiles, source,
user, history); only what Slicer can have changed is replaced: geometry (only if IJKToRAS changed, so unchanged
files keep their exact matrix), values (label table, scaling) and the axis of a split channel or sequence frame.
A new node is written in the DICOM patient frame (LPS, millimetre) with array axes ``z, y, x``.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import patient_space, same, updated_space


def values_dict(original: Optional[dict], kind: str, scaling: Optional[Tuple[float, float]] = None,
                labels: Optional[dict] = None) -> dict:
    """``values`` for export: the original entry (unit, fill, ...) with kind, scaling and label table replaced."""
    out = dict(original or {})
    if out.get("kind") != kind:
        out = {key: value for key, value in out.items() if key in ("unit", "unit_code", "fill")}
    out["kind"] = kind
    out.pop("scale", None)
    out.pop("offset", None)
    if scaling is not None:
        out["scale"], out["offset"] = float(scaling[0]), float(scaling[1])
    if labels is not None:
        out["labels"] = labels
    elif kind not in ("label", "instance", "probability"):
        out.pop("labels", None)
        out.pop("label_layout", None)
    return out


def new_meta(Meta, shape: Sequence[int], ijk_to_ras: np.ndarray, values: dict, tool: str,
             component_axis: bool = False):
    """Metadata of a node Slicer created. ``shape`` is the array to write (``k, j, i`` [+ components])."""
    n_space = len(shape) - (1 if component_axis else 0)
    names = ["z", "y", "x"][-n_space:]
    axes = [{"name": n, "type": "space"} for n in names]
    if component_axis:
        axes.append({"name": "c", "type": "component"})
    meta = Meta(axes=axes, spaces=[patient_space(ijk_to_ras, names)], values=values, shape=tuple(shape))
    meta.add_history("created", tool=tool)
    return meta


def edited_meta(Meta, Space, original: dict, record: dict, shape: Sequence[int], ijk_to_ras: np.ndarray,
                values: dict, tool: str):
    """Metadata of a node loaded from ``.mla`` (``original`` = that file's ``meta.to_dict(include_lazy=True)``).

    ``record`` is what the loader stored on the node (``spatial_axes``, ``ijk_to_ras``, ``space``, ``extra``,
    ``extra_axis``, ``index``).
    """
    data = copy.deepcopy(original)
    for key in ("writer", "storage"):
        data.pop(key, None)
    notes: List[str] = []
    axes = data["axes"]
    if record.get("extra") in ("channels", "sequence") and record.get("index") is not None:
        removed = axes.pop(record["extra_axis"])
        _drop_box_axis(data, removed["name"])
        notes.append(f"kept {removed['name']} index {record['index']}")
    space_names = [a["name"] for a in axes if a.get("type") == "space"]
    if not same(ijk_to_ras, record["ijk_to_ras"]):
        spaces = data.get("spaces") or []
        chosen = next((s for s in spaces if s["name"] == record.get("space")), None)
        new = updated_space(Space.from_dict(chosen), ijk_to_ras, space_names) if chosen else None
        if new is None:
            new = patient_space(ijk_to_ras, space_names)
        dropped = [s["name"] for s in spaces if s["name"] != new["name"]]
        data["spaces"] = [new]
        notes.append("geometry changed" + (f"; dropped spaces {dropped}" if dropped else ""))
    data["values"] = values
    meta = Meta.from_dict(data, shape=tuple(shape))
    fields = {"tool": tool}
    if notes:
        fields["settings"] = {"notes": notes}
    meta.add_history("edited", **fields)
    return meta


def _drop_box_axis(data: dict, name: str) -> None:
    boxes = ((data.get("profiles") or {}).get("ml") or {}).get("boxes")
    if not boxes or boxes.get("frame") != "index" or name not in boxes.get("axes", []):
        return
    k = boxes["axes"].index(name)
    boxes["axes"].pop(k)
    for item in boxes.get("items", []):
        item["min"].pop(k)
        item["max"].pop(k)
