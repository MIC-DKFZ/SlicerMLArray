"""Checks SlicerMLArray inside 3D Slicer against Slicer's own readers, the example data and save round trips.

    Slicer --no-splash --no-main-window --disable-settings \
        --additional-module-paths SlicerMLArray --python-script scripts/verify_in_slicer.py DATA OUT [EXAMPLES]

DATA: output of ``make_verify_data.py``; OUT: directory for saved files and ``results.json``; EXAMPLES (optional):
directory of example datasets (``<dataset>/<case>.mla``, ``<case>_seg.mla``, ``<case>_boxes.mla``).
With a main window (no ``--no-main-window``), screenshots of every slice view and the 3-D view are written to OUT.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import slicer
import vtk

from mlarray import MLArray
from SlicerMLArrayLib import slicer_io
from SlicerMLArrayLib.boxes import index_box_from_roi
from SlicerMLArrayLib.geometry import grid

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
EXAMPLES = Path(sys.argv[3]) if len(sys.argv) > 3 else None
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append({"check": name, "ok": bool(condition), "detail": str(detail)})
    print(("PASS " if condition else "FAIL ") + name + (f"  ({detail})" if not condition else ""))


def run(name, fn):
    try:
        fn()
    except Exception as error:
        traceback.print_exc()
        check(name, False, f"exception: {error}")


def load(path):
    nodes, notes = slicer_io.load(str(path))
    return nodes, notes


def first(nodes, cls):
    return next(n for n in nodes if n.IsA(cls))


def ijk(node):
    return slicer_io.ijk_to_ras(node)


# ------------------------------------------------------------------------------------------------ phantoms

def same_world(reference, node):
    """Every voxel of ``node`` lies on a voxel of ``reference`` with the same value.

    Slicer's own reader may store a grid with an axis reversed (it does so for left-handed NIfTI grids), so the
    matrices are compared through the index mapping, which must be a signed permutation with integer shifts.
    """
    t = np.linalg.inv(ijk(reference)) @ ijk(node)
    if not (np.allclose(t, np.rint(t), atol=1e-4) and np.allclose(np.abs(np.rint(t[:3, :3])).sum(axis=0), 1)):
        return False, f"not the same grid: {np.round(t, 4)}"
    t = np.rint(t).astype(int)
    a, b = slicer.util.arrayFromVolume(node), slicer.util.arrayFromVolume(reference)
    k, j, i = np.indices(a.shape)
    ijk_index = np.stack([i, j, k, np.ones_like(i)], axis=-1) @ t.T
    inside = np.all((ijk_index[..., :3] >= 0) & (ijk_index[..., :3] < np.array(b.shape[::-1])), axis=-1)
    if not inside.all():
        return False, "grid not inside the reference"
    values = b[ijk_index[..., 2], ijk_index[..., 1], ijk_index[..., 0]]
    return bool(np.array_equal(values, a)), "" if np.array_equal(values, a) else "values differ"


def phantoms():
    for source in sorted((DATA / "phantom").iterdir()):
        if source.suffix == ".mla":
            continue
        stem = source.name.replace(".nii.gz", "").replace(".nrrd", "") + ("_nrrd" if source.suffix == ".nrrd" else "")
        native = slicer.util.loadVolume(str(source))
        mine = first(load(DATA / "phantom" / f"{stem}.mla")[0], "vtkMRMLScalarVolumeNode")
        same, detail = same_world(native, mine)
        check(f"phantom {source.name}: every voxel where Slicer's own reader puts it", same, detail)
        slicer.mrmlScene.Clear(0)


def save_new_nodes():
    """Volumes and segmentations made in Slicer, saved as .mla: geometry equals MedVol's conversion of the file."""
    for stem, source in (("lps", "lps.nii.gz"), ("oblique", "oblique.nii.gz"), ("sagittal", "sagittal.nii.gz")):
        native = slicer.util.loadVolume(str(DATA / "phantom" / source))
        out = OUT / f"new_{stem}.mla"
        slicer_io.write(native, str(out))
        saved = MLArray.open(out, mmap_mode="r")
        check(f"save new volume {source}: file grid is the node's grid",
              np.allclose(grid(saved.meta).ijk_to_ras, ijk(native), atol=1e-4))
        reloaded = first(load(out)[0], "vtkMRMLScalarVolumeNode")
        medvol_node = first(load(DATA / "phantom" / f"{stem}.mla")[0], "vtkMRMLScalarVolumeNode")
        same, detail = same_world(medvol_node, reloaded)
        check(f"save new volume {source}: same world content as MedVol's conversion", same, detail)
        # a segmentation painted on it (the letters)
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(native)
        segment = segmentation.GetSegmentation().AddEmptySegment("", "letters", [1.0, 0.8, 0.2])
        mask = (slicer.util.arrayFromVolume(native) >= 1000).astype(np.uint8)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, segment, native)
        out = OUT / f"new_{stem}_seg.mla"
        slicer_io.write(segmentation, str(out))
        saved = MLArray.open(out, mmap_mode="r")
        check(f"save new segmentation on {source}: grid", np.allclose(grid(saved.meta).ijk_to_ras, ijk(native), atol=1e-4))
        check(f"save new segmentation on {source}: voxels", np.array_equal(np.asarray(saved.to_numpy()) != 0, mask != 0))
        check(f"save new segmentation on {source}: label table",
              saved.meta.values.kind == "label" and saved.meta.values.labels["1"]["name"] == "letters")
        slicer.mrmlScene.Clear(0)


# ------------------------------------------------------------------------------------------------ layouts

def layouts():
    L = DATA / "layouts"
    nodes, _ = load(L / "coronal_2d.mla")
    node = nodes[0]
    expected = np.array([[-0.5, 0, 0, -1], [0, 0, -1, -2], [0, -0.5, 0, 3], [0, 0, 0, 1]])   # K = I x J
    check("2-D coronal image: IJKToRAS", np.allclose(ijk(node), expected), ijk(node))
    check("2-D coronal image: shape", slicer.util.arrayFromVolume(node).shape == (1, 30, 40))
    out = OUT / "coronal_2d.mla"
    slicer_io.write(node, str(out))
    saved, original = MLArray.open(out, mmap_mode="r"), MLArray.open(L / "coronal_2d.mla", mmap_mode="r")
    check("2-D coronal image: saved as 2-D, same matrix", saved.shape == (30, 40) and np.allclose(
        saved.meta.spaces[0].matrix(), original.meta.spaces[0].matrix()))

    nodes, _ = load(L / "rgb.mla")
    node = nodes[0]
    original = MLArray.open(L / "rgb.mla", mmap_mode="r")
    check("RGB channel axis: vector volume", node.IsA("vtkMRMLVectorVolumeNode")
          and np.array_equal(slicer.util.arrayFromVolume(node), np.asarray(original.to_numpy())))
    slicer_io.write(node, str(OUT / "rgb.mla"))
    saved = MLArray.open(OUT / "rgb.mla", mmap_mode="r")
    check("RGB: round trip keeps axes and data", [a.name for a in saved.meta.axes] == ["z", "y", "x", "c"]
          and saved.meta.axes[3].labels == ("red", "green", "blue")
          and np.array_equal(np.asarray(saved.to_numpy()), np.asarray(original.to_numpy())))

    nodes, _ = load(L / "channels.mla")
    original = np.asarray(MLArray.open(L / "channels.mla", mmap_mode="r").to_numpy())
    check("channels: one volume each, named by label", [n.GetName() for n in nodes] == ["channels T1", "channels T2"]
          and np.array_equal(slicer.util.arrayFromVolume(nodes[1]), original[1]))
    slicer_io.write(nodes[1], str(OUT / "channel_T2.mla"))
    saved = MLArray.open(OUT / "channel_T2.mla", mmap_mode="r")
    check("channels: saving one keeps the others' metadata without the axis",
          [a.name for a in saved.meta.axes] == ["z", "y", "x"] and np.array_equal(np.asarray(saved.to_numpy()), original[1]))

    nodes, _ = load(L / "time.mla")
    proxy = nodes[0]
    browser = slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(proxy)
    sequence = browser.GetSequenceNode(proxy) if browser else None
    values = [sequence.GetNthIndexValue(i) for i in range(sequence.GetNumberOfDataNodes())] if sequence else []
    check("time axis: sequence with the axis' times", values == ["0", "2", "4"], values)

    nodes, _ = load(L / "ct_scaled.mla")
    node = nodes[0]
    original = MLArray.open(L / "ct_scaled.mla", mmap_mode="r")
    data = slicer.util.arrayFromVolume(node)
    check("scaled CT: physical values (HU) as int16", data.dtype == np.int16
          and np.array_equal(data, np.asarray(original.to_numpy()).astype(np.int32) - 1024))
    slicer_io.write(node, str(OUT / "ct_scaled.mla"))
    saved = MLArray.open(OUT / "ct_scaled.mla", mmap_mode="r")
    check("scaled CT: saved with the original dtype and scaling", saved.dtype == np.uint16
          and saved.meta.values.offset == -1024.0
          and np.array_equal(np.asarray(saved.to_numpy()), np.asarray(original.to_numpy())))

    nodes, _ = load(L / "label_channels.mla")
    segmentation = nodes[0]
    names = [segmentation.GetSegmentation().GetNthSegment(i).GetName()
             for i in range(segmentation.GetSegmentation().GetNumberOfSegments())]
    check("label channels: one segment per class channel", names == ["kidney", "cyst"], names)
    slicer_io.write(segmentation, str(OUT / "label_channels.mla"))
    saved = MLArray.open(OUT / "label_channels.mla", mmap_mode="r")
    original = MLArray.open(L / "label_channels.mla", mmap_mode="r")
    check("label channels: round trip (overlap kept, background channel)",
          np.array_equal(np.asarray(saved.to_numpy()), np.asarray(original.to_numpy())))

    nodes, _ = load(L / "seg_codes.mla")
    segmentation = first(nodes, "vtkMRMLSegmentationNode")
    roi = first(nodes, "vtkMRMLMarkupsROINode")
    seg = segmentation.GetSegmentation()
    segments = {seg.GetNthSegment(i).GetName(): seg.GetNthSegment(i) for i in range(seg.GetNumberOfSegments())}
    liver = segments.get("liver")
    entry = vtk.mutable("")
    check("labels: names", sorted(segments) == ["liver", "spleen"], sorted(segments))
    check("labels: colours", liver is not None and np.allclose(liver.GetColor(), np.array([221, 130, 101]) / 255, atol=1e-3))
    check("labels: code as terminology", liver is not None and liver.GetTag(liver.GetTerminologyEntryTagName(), entry)
          and "10200004" in str(entry), str(entry))
    centre = [0.0] * 3
    roi.GetCenterWorld(centre)
    # patient box x -8..-5, y 21..23, z -25..-18 (LPS) -> RAS centre (6.5, -22, -21.5)
    check("box in patient frame: ROI centre", np.allclose(centre, [6.5, -22, -21.5]), centre)
    check("box: name with score", roi.GetName() == "liver 0.50", roi.GetName())
    slicer_io.write(segmentation, str(OUT / "seg_codes.mla"))
    saved = MLArray.open(OUT / "seg_codes.mla", mmap_mode="r")
    original = MLArray.open(L / "seg_codes.mla", mmap_mode="r")
    check("labels: round trip keeps values, table, codes, boxes",
          np.array_equal(np.asarray(saved.to_numpy()), np.asarray(original.to_numpy()))
          and dict(saved.meta.values.labels) == dict(original.meta.values.labels)
          and saved.meta.profiles == original.meta.profiles, saved.meta.values.labels)
    slicer_io.write_boxes([roi], str(OUT / "seg_codes_boxes.mla"))
    boxes = MLArray.open(OUT / "seg_codes_boxes.mla", mmap_mode="r").meta.profiles["ml"]["boxes"]
    check("boxes: export in the file's frame", boxes["frame"] == "patient"
          and dict(boxes["items"][0]) == dict(original.meta.profiles["ml"]["boxes"]["items"][0]), boxes)
    slicer_io.write_boxes([roi], str(OUT / "seg_codes_boxes_index.mla"), segmentation)
    boxes = MLArray.open(OUT / "seg_codes_boxes_index.mla", mmap_mode="r").meta.profiles["ml"]["boxes"]
    check("boxes: export in a reference image's indices", boxes["frame"] == "index", boxes)
    new_roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
    new_roi.SetCenter(10.0, -20.0, 30.0)
    new_roi.SetSize(4.0, 6.0, 8.0)
    slicer_io.write_boxes([new_roi], str(OUT / "new_boxes.mla"))
    boxes = MLArray.open(OUT / "new_boxes.mla", mmap_mode="r").meta.profiles["ml"]["boxes"]
    item = boxes["items"][0]
    check("boxes: new ROI in patient LPS coordinates", boxes["frame"] == "patient"
          and list(item["min"]) == [-12, 17, 26] and list(item["max"]) == [-8, 23, 34], dict(item))
    slicer.mrmlScene.Clear(0)

    slicer_io.set_labels_as("labelmap")
    try:
        nodes, _ = load(L / "seg_codes.mla")
        labelmap = first(nodes, "vtkMRMLLabelMapVolumeNode")
        colours = labelmap.GetDisplayNode().GetColorNode()
        check("labelmap option: labelmap with named colours", colours.GetColorName(3) == "spleen")
        slicer_io.write(labelmap, str(OUT / "seg_labelmap.mla"))
        saved = MLArray.open(OUT / "seg_labelmap.mla", mmap_mode="r")
        original = MLArray.open(L / "seg_codes.mla", mmap_mode="r")
        check("labelmap option: round trip", np.array_equal(np.asarray(saved.to_numpy()), np.asarray(original.to_numpy()))
              and dict(saved.meta.values.labels) == dict(original.meta.values.labels))
    finally:
        slicer_io.set_labels_as("segmentation")
    slicer.mrmlScene.Clear(0)


# ------------------------------------------------------------------------------------------------ examples

def examples():
    for dataset in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
        for image_path in sorted(dataset.glob("*.mla")):
            if image_path.stem.endswith(("_seg", "_boxes")):
                continue
            case = image_path.stem
            start = time.time()
            images, _ = load(image_path)
            segs, _ = load(dataset / f"{case}_seg.mla")
            rois, _ = load(dataset / f"{case}_boxes.mla")
            segmentation = first(segs, "vtkMRMLSegmentationNode")
            meta = MLArray.open(dataset / f"{case}_seg.mla", mmap_mode="r").meta
            seg_array = np.asarray(MLArray.open(dataset / f"{case}_seg.mla", mmap_mode="r").to_numpy())
            reference = json.loads(segmentation.GetAttribute(slicer_io.ATTR_RECORD))
            same_grid = all(np.allclose(ijk(v), reference["ijk_to_ras"], atol=1e-4) for v in images)
            items = MLArray.open(dataset / f"{case}_boxes.mla", mmap_mode="r").meta.profiles["ml"]["boxes"]["items"]
            ok_boxes = len(rois) == len(items)
            names = [a.name for a in meta.axes][::-1]
            for roi, item in zip(rois, items):
                lo, hi = index_box_from_roi(slicer_io.roi_of(roi), [a.name for a in meta.axes], names,
                                            reference["ijk_to_ras"])
                region = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
                inside = int((seg_array[region] == item["label"]).sum())
                ok_boxes &= lo == list(item["min"]) and hi == list(item["max"]) and inside >= item.get("voxels", 1)
            check(f"example {dataset.name}/{case}: image and seg on one grid", same_grid)
            check(f"example {dataset.name}/{case}: {len(items)} boxes enclose their components", ok_boxes)
            print(f"   {time.time() - start:.1f} s")
            slicer.mrmlScene.Clear(0)


# ------------------------------------------------------------------------------------------------ screenshots

def screenshots():
    import ScreenCapture

    capture = ScreenCapture.ScreenCaptureLogic()
    manager = slicer.app.layoutManager()
    manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    window = slicer.util.mainWindow()
    window.resize(1600, 1200)
    window.showMaximized()
    for panel in ("PanelDockWidget", "PythonConsoleDockWidget"):
        widget = slicer.util.findChild(window, panel) if hasattr(slicer.util, "findChild") else None
        if widget is not None:
            widget.hide()
    slicer.app.processEvents()

    def shoot(prefix):
        slicer.app.processEvents()
        for view in ("Red", "Green", "Yellow"):
            widget = manager.sliceWidget(view)
            widget.sliceLogic().FitSliceToAll()
            capture.captureImageFromView(widget.sliceView(), str(OUT / f"{prefix}_{view}.png"))
        three = manager.threeDWidget(0).threeDView()
        three.resetFocalPoint()
        three.resetCamera()
        capture.captureImageFromView(three, str(OUT / f"{prefix}_3d.png"))

    for stem in ("lps", "oblique", "sagittal"):
        load(DATA / "phantom" / f"{stem}.mla")
        shoot(f"phantom_{stem}")
        slicer.mrmlScene.Clear(0)
    if EXAMPLES:
        datasets = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())
        dataset = next((d for d in datasets if "AMOS" in d.name), datasets[0])
        case = sorted(p.stem for p in dataset.glob("*_seg.mla"))[0][:-4]
        nodes = []
        for suffix in ("", "_seg", "_boxes"):
            nodes += load(dataset / f"{case}{suffix}.mla")[0]
        for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            node.CreateClosedSurfaceRepresentation()
            node.GetDisplayNode().SetOpacity2DFill(0.35)
        rois = [n for n in nodes if n.IsA("vtkMRMLMarkupsROINode")]
        largest = max(rois, key=lambda r: np.prod(r.GetSize()))
        centre = [0.0] * 3
        largest.GetCenterWorld(centre)
        for view in ("Red", "Green", "Yellow"):
            manager.sliceWidget(view).sliceLogic().FitSliceToAll()
            manager.sliceWidget(view).mrmlSliceNode().JumpSliceByCentering(*centre)
        three = manager.threeDWidget(0).threeDView()
        three.resetFocalPoint()
        three.resetCamera()
        slicer.app.processEvents()
        for view in ("Red", "Green", "Yellow"):
            capture.captureImageFromView(manager.sliceWidget(view).sliceView(), str(OUT / f"example_{case}_{view}.png"))
        capture.captureImageFromView(three, str(OUT / f"example_{case}_3d.png"))


try:
    if slicer.util.mainWindow() is None:
        run("phantoms", phantoms)
        run("save new nodes", save_new_nodes)
        run("layouts", layouts)
        if EXAMPLES:
            run("examples", examples)
    else:
        run("screenshots", screenshots)
finally:
    failed = [r for r in RESULTS if not r["ok"]]
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))
    print(f"{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    slicer.app.exit(1 if failed else 0)
