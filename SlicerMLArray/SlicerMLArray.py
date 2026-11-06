"""MLArray (.mla) files in 3D Slicer: open and save them with Slicer's file dialogs, export boxes."""

import os
import tempfile
import traceback

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)

FILE_TYPE = "MLArrayFile"
DESCRIPTION = "MLArray"


class SlicerMLArray(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "MLArray"
        parent.categories = ["Informatics"]
        parent.dependencies = []
        parent.contributors = ["Karol Gotkowski (DKFZ)"]
        parent.helpText = (
            "Open and save <a href='https://github.com/MIC-DKFZ/mlarray'>MLArray</a> (.mla) files with "
            "File > Add Data and File > Save. Images become volumes, label maps segmentations (with names, colours "
            "and codes), boxes Markups ROIs. This module sets how label maps open and exports ROIs as boxes.")
        parent.acknowledgementText = (
            "Developed by the Applied Computer Vision Lab of Helmholtz Imaging and the Division of Medical Image "
            "Computing at DKFZ.")


class SlicerMLArrayFileReader:
    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return DESCRIPTION

    def fileType(self):
        return FILE_TYPE

    def extensions(self):
        return ["MLArray (*.mla)"]

    def canLoadFileConfidence(self, filePath):
        return 0.6 if str(filePath).lower().endswith(".mla") else 0.0

    def load(self, properties):
        from SlicerMLArrayLib import slicer_io

        try:
            nodes, notes = slicer_io.load(properties["fileName"], properties.get("name"))
        except Exception as error:
            traceback.print_exc()
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, f"Could not load the MLArray file: {error}")
            return False
        for note in notes:
            self.parent.userMessages().AddMessage(vtk.vtkCommand.WarningEvent, note)
        self.parent.loadedNodes = [node.GetID() for node in nodes]
        return True


class SlicerMLArrayFileWriter:
    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return DESCRIPTION

    def fileType(self):
        return FILE_TYPE

    def extensions(self, obj):
        return ["MLArray (.mla)"] if self.canWriteObjectConfidence(obj) > 0 else []

    def canWriteObjectConfidence(self, obj):
        from SlicerMLArrayLib.slicer_io import ATTR_RECORD, can_write

        if not can_write(obj):
            return 0.0
        return 0.7 if obj.GetAttribute(ATTR_RECORD) else 0.3       # default format only for nodes from .mla

    def write(self, properties):
        from SlicerMLArrayLib import slicer_io

        node = slicer.mrmlScene.GetNodeByID(properties["nodeID"])
        path = properties["fileName"]
        if not path.lower().endswith(".mla"):
            path += ".mla"
        try:
            notes = slicer_io.write(node, path)
        except Exception as error:
            traceback.print_exc()
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, f"Could not save as MLArray: {error}")
            return False
        for note in notes:
            self.parent.userMessages().AddMessage(vtk.vtkCommand.WarningEvent, note)
        self.parent.writtenNodes = [node.GetID()]
        return True


class SlicerMLArrayWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        from SlicerMLArrayLib import slicer_io

        options = ctk.ctkCollapsibleButton()
        options.text = "Opening files"
        self.layout.addWidget(options)
        form = qt.QFormLayout(options)
        self.labelsAs = qt.QComboBox()
        self.labelsAs.addItem("Segmentation (names, colours, codes)", "segmentation")
        self.labelsAs.addItem("Labelmap volume", "labelmap")
        self.labelsAs.setCurrentIndex(0 if slicer_io.labels_as() == "segmentation" else 1)
        self.labelsAs.connect("currentIndexChanged(int)",
                              lambda i: slicer_io.set_labels_as(self.labelsAs.itemData(i)))
        form.addRow("Label maps open as:", self.labelsAs)

        boxes = ctk.ctkCollapsibleButton()
        boxes.text = "Export boxes"
        self.layout.addWidget(boxes)
        form = qt.QFormLayout(boxes)
        self.rois = slicer.qMRMLCheckableNodeComboBox()
        self.rois.nodeTypes = ["vtkMRMLMarkupsROINode"]
        self.rois.setMRMLScene(slicer.mrmlScene)
        self.rois.setToolTip("ROIs to save as boxes")
        form.addRow("ROIs:", self.rois)
        self.reference = slicer.qMRMLNodeComboBox()
        self.reference.nodeTypes = ["vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]
        self.reference.noneEnabled = True
        self.reference.addEnabled = False
        self.reference.removeEnabled = False
        self.reference.setMRMLScene(slicer.mrmlScene)
        self.reference.setToolTip("Store the boxes in this image's voxel indices. None: the ROIs' own file, or "
                                  "patient coordinates (LPS, mm) for new ROIs.")
        form.addRow("Image:", self.reference)
        self.path = ctk.ctkPathLineEdit()
        self.path.filters = ctk.ctkPathLineEdit.Files
        self.path.nameFilters = ["MLArray (*.mla)"]
        self.path.setToolTip("Box-only .mla file to write")
        form.addRow("File:", self.path)
        button = qt.QPushButton("Export")
        button.connect("clicked()", self.onExport)
        form.addRow(button)
        self.layout.addStretch(1)

    def onExport(self):
        from SlicerMLArrayLib import slicer_io

        nodes = self.rois.checkedNodes()
        path = self.path.currentPath
        if not nodes or not path:
            slicer.util.errorDisplay("Select ROIs and a file.")
            return
        if not path.lower().endswith(".mla"):
            path += ".mla"
        try:
            slicer_io.write_boxes(nodes, path, self.reference.currentNode())
        except Exception as error:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not export the boxes: {error}")
            return
        slicer.util.showStatusMessage(f"Saved {len(nodes)} boxes to {path}", 3000)


class SlicerMLArrayLogic(ScriptedLoadableModuleLogic):
    """Python access: ``load``, ``save``, ``saveBoxes``."""

    def load(self, path, name=None):
        from SlicerMLArrayLib import slicer_io

        return slicer_io.load(path, name)[0]

    def save(self, node, path):
        from SlicerMLArrayLib import slicer_io

        return slicer_io.write(node, path)

    def saveBoxes(self, roiNodes, path, reference=None):
        from SlicerMLArrayLib import slicer_io

        return slicer_io.write_boxes(roiNodes, path, reference)


class SlicerMLArrayTest(ScriptedLoadableModuleTest):
    """Round trips through real files: geometry, labels and boxes (needs mlarray)."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_geometry_labels_boxes()

    def test_geometry_labels_boxes(self):
        from mlarray import MLArray

        self.delayDisplay("MLArray round trip")
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "seg.mla")
        array = np.zeros((4, 5, 6), np.uint8)              # z, y, x
        array[1:3, 2:4, 3:5] = 2
        matrix = [[0, 0, -0.8, 10.0], [0, -0.7, 0, 20.0], [2.5, 0, 0, -30.0], [0, 0, 0, 1]]   # LPS, x flipped
        MLArray(array, axes=["z", "y", "x"],
                spaces=[{"name": "patient", "axes": [
                    {"name": "x", "unit": "millimeter", "increases_toward": "left"},
                    {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
                    {"name": "z", "unit": "millimeter", "increases_toward": "superior"}],
                    "from_array": {"array_axes": ["z", "y", "x"], "matrix": matrix}}],
                values={"kind": "label", "labels": {"2": {"name": "liver", "color": [200, 100, 50]}}},
                profiles={"ml": {"boxes": {"frame": "index", "axes": ["z", "y", "x"],
                                           "items": [{"min": [1, 2, 3], "max": [2, 3, 4], "label": 2, "score": 0.9}]}}},
                ).save(path)
        nodes = SlicerMLArrayLogic().load(path)
        segmentation = [n for n in nodes if n.IsA("vtkMRMLSegmentationNode")][0]
        roi = [n for n in nodes if n.IsA("vtkMRMLMarkupsROINode")][0]
        segment = segmentation.GetSegmentation().GetNthSegment(0)
        self.assertEqual(segment.GetName(), "liver")
        self.assertEqual(roi.GetName(), "liver 0.90")
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        expected = lps_to_ras @ np.array(matrix)[:, [2, 1, 0, 3]]           # columns I=x, J=y, K=z
        reference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentation, reference, 0)            # extent of the reference geometry
        from SlicerMLArrayLib.slicer_io import ijk_to_ras
        np.testing.assert_allclose(ijk_to_ras(reference), expected, atol=1e-6)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(reference) != 0, array != 0)
        out = os.path.join(directory, "out.mla")
        SlicerMLArrayLogic().save(segmentation, out)
        saved = MLArray.open(out, mmap_mode="r")
        np.testing.assert_array_equal(np.asarray(saved.to_numpy()), array)
        np.testing.assert_allclose(saved.meta.spaces[0].matrix(), matrix, atol=1e-9)
        self.assertEqual(saved.meta.values.labels["2"]["name"], "liver")
        boxes = os.path.join(directory, "boxes.mla")
        SlicerMLArrayLogic().saveBoxes([roi], boxes)
        box = dict(MLArray.open(boxes, mmap_mode="r").meta.profiles["ml"]["boxes"]["items"][0])
        self.assertEqual((list(box["min"]), list(box["max"]), box["label"]), ([1, 2, 3], [2, 3, 4], 2))
        self.delayDisplay("MLArray round trip passed")
