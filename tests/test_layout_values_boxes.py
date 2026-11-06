"""Layouts, values and boxes (no Slicer)."""

import numpy as np
import pytest

from mlarray import Meta, Space, Values
from SlicerMLArrayLib.boxes import Roi, index_box_from_roi, roi_from_index_box, roi_from_space_box, space_box_from_roi
from SlicerMLArrayLib.errors import UnsupportedFile
from SlicerMLArrayLib.export import edited_meta, new_meta, values_dict
from SlicerMLArrayLib.geometry import patient_space
from SlicerMLArrayLib.layout import index_name, layout
from SlicerMLArrayLib.values import box_name, label_table, physical, stored, table_dict


def _meta(axes, shape, values=None, profiles=None):
    return Meta(axes=axes, shape=shape, values=values, profiles=profiles)


@pytest.mark.parametrize("axes, values, expected", [
    (["z", "y", "x"], None, ("scalar", None)),
    (["y", "x"], None, ("scalar", None)),
    (["z", "y", "x"], {"kind": "label"}, ("segmentation", None)),
    (["z", "y", "x"], {"kind": "instance"}, ("labelmap", None)),
    (["z", "y", "x", {"name": "c", "type": "component"}], None, ("scalar", "vector")),
    (["y", "x", {"name": "c", "type": "channel", "labels": ["red", "green", "blue"]}], None, ("scalar", "vector")),
    ([{"name": "c", "type": "channel", "labels": ["T1", "T2"]}, "z", "y", "x"], None, ("scalar", "channels")),
    ([{"name": "t", "type": "time"}, "z", "y", "x"], None, ("scalar", "sequence")),
    ([{"name": "c", "type": "channel"}, "z", "y", "x"], {"kind": "label", "label_layout": "channel"},
     ("segmentation", "label_channels")),
    ([{"name": "e", "type": "other"}, "z", "y", "x"], {"kind": "label"}, ("segmentation", "channels")),
])
def test_layouts(axes, values, expected):
    shape = tuple(len(a["labels"]) if isinstance(a, dict) and "labels" in a else 2 for a in axes)
    plan = layout(_meta(axes, shape, values), shape)
    assert (plan.element, plan.extra) == expected


def test_labels_as_labelmap_and_rejections():
    assert layout(_meta(["z", "y", "x"], (2, 2, 2), {"kind": "label"}), (2, 2, 2), "labelmap").element == "labelmap"
    with pytest.raises(UnsupportedFile):
        layout(_meta([{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}, "z", "y", "x"], (2,) * 5),
               (2,) * 5)
    with pytest.raises(UnsupportedFile):
        layout(_meta(["x"], (2,)), (2,))


def test_box_only_file():
    profiles = {"ml": {"boxes": {"frame": "index", "axes": ["z", "y", "x"], "items": [{"min": [0, 0, 0], "max": [1, 1, 1]}]}}}
    assert layout(Meta(axes=["z", "y", "x"], profiles=profiles), (0, 0, 0)).boxes_only
    with pytest.raises(UnsupportedFile):
        layout(Meta(axes=["z", "y", "x"]), (0, 0, 0))


def test_index_names():
    axis = Meta(axes=[{"name": "t", "type": "time", "start": 0.0, "step": 2.5, "unit": "second"}, "x"], shape=(2, 2)).axes[0]
    assert index_name(axis, 2) == "t 5 second"
    axis = Meta(axes=[{"name": "c", "type": "channel", "labels": ["T1", "T2"]}, "x"], shape=(2, 2)).axes[0]
    assert index_name(axis, 1) == "T2"


def test_scaling_round_trip():
    stored_values = np.array([0, 100, 4095], np.uint16)
    out, record = physical(stored_values, Values(scale=1.0, offset=-1024.0))
    assert out.dtype == np.int16 and list(out) == [-1024, -924, 3071]
    back, scaling = stored(out, record)
    assert back.dtype == np.uint16 and list(back) == [0, 100, 4095] and scaling == (1.0, -1024.0)
    out, record = physical(np.array([1, 2], np.int16), Values(scale=0.5, offset=0.0))
    assert out.dtype == np.float32
    edited = out.copy()
    edited[0] = 0.3                                              # not representable: written as is, unscaled
    back, scaling = stored(edited, record)
    assert back.dtype == np.float32 and scaling is None
    assert physical(np.array([1, 2], np.int16), Values())[1] is None


def test_label_tables():
    values = Values(kind="label", labels={"0": {"name": "background"}, "1": {"name": "liver", "color": [1, 2, 3],
                    "code": {"scheme": "SCT", "value": "10200004", "meaning": "Liver"}, "note": "x"}})
    table = label_table(values, present=[1, 4])
    assert [(l.value, l.name) for l in table] == [(1, "liver"), (4, "label 4")]
    assert table[0].color == (1, 2, 3) and table[0].code["value"] == "10200004" and table[0].extra == {"note": "x"}
    out = table_dict(table, {"name": "background"})
    assert out["0"] == {"name": "background"} and out["1"]["note"] == "x" and out["1"]["code"]["scheme"] == "SCT"
    assert box_name("tumor", 1, 0.8432) == "tumor 0.84" and box_name(None, 3, None) == "3"


ROTATION = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1.0]])


@pytest.mark.parametrize("linear", [np.diag([0.7, 0.8, 2.5]), np.diag([-0.7, 0.8, -2.5]),
                                    ROTATION @ np.diag([0.7, 0.8, 2.5]),
                                    np.array([[0, 0, 2.0], [0.5, 0, 0], [0, -0.6, 0]])])
def test_index_box_round_trip(linear):
    ijk = np.eye(4)
    ijk[:3, :3] = linear
    ijk[:3, 3] = [3, -4, 5]
    names = ["x", "y", "z"]                                   # I, J, K
    roi = roi_from_index_box([2, 3, 4], [5, 6, 9], ["z", "y", "x"], names, ijk, (20, 20, 20))
    np.testing.assert_allclose(roi.axes.T @ roi.axes, np.eye(3), atol=1e-9)
    assert np.linalg.det(roi.axes) > 0
    # ROI corners are the outer voxel edges: (4-0.5 .. 5+0.5) along I etc.
    corners = (np.linalg.inv(ijk) @ np.c_[roi.corners(), np.ones(8)].T).T[:, :3]
    np.testing.assert_allclose(corners.min(axis=0), [3.5, 2.5, 1.5], atol=1e-9)
    np.testing.assert_allclose(corners.max(axis=0), [9.5, 6.5, 5.5], atol=1e-9)
    assert index_box_from_roi(roi, ["z", "y", "x"], names, ijk) == ([2, 3, 4], [5, 6, 9])


def test_box_missing_axes_span_the_image_and_sheared_grids_give_bounding_boxes():
    ijk = np.diag([1.0, 1.0, 1.0, 1.0])
    roi = roi_from_index_box([2, 3], [4, 5], ["y", "x"], ["x", "y", "z"], ijk, (10, 10, 7))
    np.testing.assert_allclose(roi.size, [3, 3, 7])
    sheared = np.eye(4)
    sheared[0, 1] = 0.5
    roi = roi_from_index_box([0, 0, 0], [1, 1, 1], ["z", "y", "x"], ["x", "y", "z"], sheared, (5, 5, 5))
    np.testing.assert_allclose(roi.axes, np.eye(3))


def test_space_box_round_trip():
    space = Space.from_dict(patient_space(np.diag([-1.0, -1.0, 1.0, 1.0]), ["z", "y", "x"]))
    roi = roi_from_space_box([-10, 5, 1], [20, 15, 3.5], ["x", "y", "z"], space)
    np.testing.assert_allclose(roi.center, [-5, -10, 2.25])            # LPS -> RAS
    assert space_box_from_roi(roi, ["x", "y", "z"], space) == ([-10, 5, 1], [20, 15, 3.5])


def test_new_and_edited_metadata():
    ijk = np.diag([-0.5, -0.5, 2.0, 1.0])
    meta = new_meta(Meta, (3, 4, 5), ijk, values_dict(None, "label", labels={"1": {"name": "a"}}), "test")
    assert [a.name for a in meta.axes] == ["z", "y", "x"] and meta.spaces[0].name == "patient"
    np.testing.assert_allclose(meta.spaces[0].matrix()[:3, :3], [[0, 0, 0.5], [0, 0.5, 0], [2.0, 0, 0]])
    original = Meta(axes=[{"name": "c", "type": "channel", "labels": ["T1", "T2"]}, "z", "y", "x"],
                    spaces=[patient_space(ijk, ["z", "y", "x"])], user={"who": "me"}, shape=(2, 3, 4, 5),
                    profiles={"ml": {"boxes": {"frame": "index", "axes": ["c", "z", "y", "x"],
                                               "items": [{"min": [0, 0, 0, 0], "max": [1, 1, 1, 1]}]}}})
    record = {"extra": "channels", "extra_axis": 0, "index": 1, "ijk_to_ras": ijk.tolist(), "space": "patient"}
    moved = ijk.copy()
    moved[0, 3] = 7.0
    out = edited_meta(Meta, Space, original.to_dict(include_lazy=True), record, (3, 4, 5), moved,
                      values_dict(None, "measurement"), "test")
    assert [a.name for a in out.axes] == ["z", "y", "x"] and out.user == {"who": "me"}
    assert out.profiles["ml"]["boxes"]["axes"] == ["z", "y", "x"]
    assert out.history[-1]["action"] == "edited"
    from SlicerMLArrayLib.geometry import grid

    np.testing.assert_allclose(grid(out).ijk_to_ras, moved, atol=1e-12)
