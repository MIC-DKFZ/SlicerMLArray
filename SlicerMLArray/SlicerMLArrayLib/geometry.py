"""MLArray v1 spaces <-> Slicer's IJKToRAS.

Slicer keeps every volume as an array ``[k, j, i]`` plus a 4x4 IJKToRAS matrix. An MLArray array is stored in C
order with named axes, so its space axes are used as they are: the last space axis is I, the one before J, the
first K. Nothing is reordered or resampled; the space's ``from_array`` matrix only gets its columns picked in I, J, K
order and its rows turned into RAS.

The direction of a space axis comes from ``increases_toward`` (spec 4.3), never from letter codes. A space without
anatomical directions (a microscope stage, napari's ``world``, ...) is shown as if it were the DICOM patient frame
(1st axis toward left, 2nd posterior, 3rd superior), and the load says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .errors import UnsupportedFile

#: Unit vector in RAS of each anatomical direction word.
RAS_OF = {
    "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0),
    "posterior": (0.0, -1.0, 0.0), "anterior": (0.0, 1.0, 0.0),
    "inferior": (0.0, 0.0, -1.0), "superior": (0.0, 0.0, 1.0),
}
#: Space axes of the DICOM patient frame (LPS, millimetre): written for every new file.
PATIENT_AXES = [
    {"name": "x", "unit": "millimeter", "unit_code": "mm", "increases_toward": "left"},
    {"name": "y", "unit": "millimeter", "unit_code": "mm", "increases_toward": "posterior"},
    {"name": "z", "unit": "millimeter", "unit_code": "mm", "increases_toward": "superior"},
]
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
#: Factor to millimetre, by unit name or UCUM code.
MILLIMETRES = {
    "millimeter": 1.0, "millimetre": 1.0, "mm": 1.0,
    "micrometer": 1e-3, "micrometre": 1e-3, "um": 1e-3,
    "nanometer": 1e-6, "nanometre": 1e-6, "nm": 1e-6,
    "centimeter": 10.0, "centimetre": 10.0, "cm": 10.0,
    "meter": 1e3, "metre": 1e3, "m": 1e3,
}
TOLERANCE = 1e-6


@dataclass
class Grid:
    """Where an array's space axes are in Slicer."""

    ijk_to_ras: np.ndarray             #: 4x4
    spatial_axes: Tuple[int, ...]      #: array axis index of K, J, I (or J, I for a 2-D image)
    space: Optional[str] = None        #: name of the space used, None for index space
    anatomical: bool = False           #: the space states anatomical directions
    notes: List[str] = field(default_factory=list)

    @property
    def is_2d(self) -> bool:
        return len(self.spatial_axes) == 2


def space_axis_indices(meta) -> Tuple[int, ...]:
    return tuple(i for i, axis in enumerate(meta.axes) if axis.type == "space")


def is_anatomical(space) -> bool:
    words = [axis.increases_toward for axis in space.axes]
    if len(words) != 3 or any(w not in RAS_OF for w in words):
        return False
    return abs(np.linalg.det(np.array([RAS_OF[w] for w in words]))) > 0.5      # three different body axes


def choose_space(meta):
    """The first space with anatomical directions, else the first space, else None."""
    spaces = list(meta.spaces)
    for space in spaces:
        if is_anatomical(space):
            return space
    return spaces[0] if spaces else None


def millimetres(axis) -> Tuple[float, Optional[str]]:
    """Factor from the axis unit to millimetre, and a note if the unit is not a length Slicer knows."""
    for key in (axis.unit_code, axis.unit):
        if key is not None and key in MILLIMETRES:
            return MILLIMETRES[key], None
    if axis.unit is None and axis.unit_code is None:
        return 1.0, None
    return 1.0, f"unit {axis.unit or axis.unit_code!r} of space axis {axis.name!r} shown as millimetre"


def space_to_ras(space) -> Tuple[np.ndarray, bool, List[str]]:
    """3 x n matrix taking space coordinates to RAS millimetre, whether it is anatomical, notes."""
    notes: List[str] = []
    if len(space.axes) > 3:
        raise UnsupportedFile(f"space {space.name!r} has {len(space.axes)} axes; Slicer shows at most 3")
    anatomical = is_anatomical(space)
    columns = []
    for i, axis in enumerate(space.axes):
        factor, note = millimetres(axis)
        if note:
            notes.append(note)
        direction = RAS_OF[axis.increases_toward] if anatomical else LPS_TO_RAS[:, i]
        columns.append(np.asarray(direction, dtype=np.float64) * factor)
    if not anatomical:
        notes.append(f"space {space.name!r} has no anatomical directions: its axes are shown as left, posterior, "
                     "superior")
    return np.stack(columns, axis=1), anatomical, notes


def grid(meta) -> Grid:
    """IJKToRAS of an array described by ``meta`` (MLArray ``Meta``)."""
    spatial = space_axis_indices(meta)
    if len(spatial) not in (2, 3):
        raise UnsupportedFile(f"the array has {len(spatial)} space axes; Slicer shows 2-D and 3-D images")
    names = [meta.axes[i].name for i in spatial]
    ijk_names = names[::-1]                                 # I = last space axis
    space = choose_space(meta)
    notes: List[str] = []
    columns: List[Optional[np.ndarray]] = [None, None, None]
    shift = np.zeros(3)
    anatomical = False
    if space is None:
        notes.append("no space: shown in index coordinates (1 mm voxels, axes left, posterior, superior)")
        for c in range(len(ijk_names)):
            columns[c] = LPS_TO_RAS[:, c].copy()
    else:
        to_ras, anatomical, space_notes = space_to_ras(space)
        notes += space_notes
        matrix = space.matrix()
        linear = to_ras @ matrix[:-1, :-1]
        shift = to_ras @ matrix[:-1, -1]
        mapped = list(space.from_array.array_axes)
        for c, name in enumerate(ijk_names):
            if name in mapped:
                columns[c] = linear[:, mapped.index(name)]
    columns = _complete(columns, notes)
    out = np.eye(4)
    out[:3, :3] = np.stack(columns, axis=1)
    out[:3, 3] = shift
    if abs(np.linalg.det(out[:3, :3])) < 1e-12:
        raise UnsupportedFile("the space matrix is singular")
    return Grid(out, tuple(spatial), space.name if space is not None else None, anatomical, notes)


def _complete(columns: List[Optional[np.ndarray]], notes: List[str]) -> List[np.ndarray]:
    """Fill a missing I, J or K column (2-D images, axes the space does not map) with the right-handed normal."""
    missing = [c for c, v in enumerate(columns) if v is None]
    if not missing:
        return columns
    if len(missing) == 1:
        c = missing[0]
        a, b = columns[(c + 1) % 3], columns[(c + 2) % 3]
        normal = np.cross(a, b)
        if np.linalg.norm(normal) > 1e-12:
            columns[c] = normal / np.linalg.norm(normal)
            return columns
    notes.append("the space does not map every image axis: unmapped axes get 1 mm along a free direction")
    used = [v for v in columns if v is not None]
    for c in missing:
        for candidate in LPS_TO_RAS.T:
            trial = used + [candidate]
            if np.linalg.matrix_rank(np.stack(trial, axis=1), tol=1e-9) == len(trial):
                columns[c] = candidate.copy()
                used.append(candidate)
                break
    return columns


def patient_space(ijk_to_ras: np.ndarray, array_axes: Sequence[str]) -> dict:
    """A new ``patient`` space (DICOM LPS, millimetre) for IJKToRAS; ``array_axes`` in array order (K, J, I)."""
    lps = LPS_TO_RAS @ np.asarray(ijk_to_ras, dtype=np.float64)[:3]   # rows L, P, S; columns I, J, K, shift
    n = len(array_axes)
    order = list(range(n))[::-1]                                       # array order K, J, I -> IJK columns
    matrix = np.zeros((4, n + 1))
    matrix[:3, :n] = lps[:, order]
    matrix[:3, n] = lps[:, 3]
    matrix[3, n] = 1.0
    return {"name": "patient", "axes": [dict(a) for a in PATIENT_AXES],
            "from_array": {"array_axes": list(array_axes), "matrix": _clean(matrix)}}


def updated_space(space, ijk_to_ras: np.ndarray, array_axes: Sequence[str]) -> Optional[dict]:
    """``space`` (as dict) with its ``from_array`` recomputed for a new IJKToRAS; None if that is not possible.

    Only for spaces with three axes (whose RAS mapping can be inverted) mapping exactly the image's space axes.
    """
    if len(space.axes) != 3 or sorted(space.from_array.array_axes) != sorted(array_axes):
        return None
    to_ras, _, _ = space_to_ras(space)
    from_ras = np.linalg.inv(to_ras)
    ijk_names = list(array_axes)[::-1]
    m = np.asarray(ijk_to_ras, dtype=np.float64)
    mapped = list(space.from_array.array_axes)
    matrix = np.zeros((4, len(mapped) + 1))
    for c, name in enumerate(mapped):
        matrix[:3, c] = from_ras @ m[:3, ijk_names.index(name)]
    matrix[:3, -1] = from_ras @ m[:3, 3]
    matrix[3, -1] = 1.0
    out = space.to_dict()
    out["from_array"] = {"array_axes": mapped, "matrix": _clean(matrix)}
    return out


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=TOLERANCE, rtol=0)


def _clean(matrix: np.ndarray) -> list:
    """Plain floats, with -0.0 written as 0.0."""
    return [[float(v) + 0.0 for v in row] for row in matrix]
