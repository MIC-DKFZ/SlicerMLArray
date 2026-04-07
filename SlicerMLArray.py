"""Hidden scripted Slicer module that registers `.mla` file IO."""

from __future__ import annotations

import traceback

import vtk

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from slicermlarray.slicer_io import (
    FILE_DESCRIPTION,
    FILE_EXTENSION_FILTER,
    FILE_WRITER_EXTENSION_FILTER,
    FILE_TYPE,
    load_mla_volume,
    normalize_mla_path,
    reader_confidence_for_path,
    write_mla_volume,
    writer_confidence_for_node,
)


class SlicerMLArray(ScriptedLoadableModule):
    """Module shell used only for Slicer file reader/writer discovery."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "SlicerMLArray"
        parent.categories = ["Informatics"]
        parent.dependencies = []
        parent.contributors = ["OpenAI Codex"]
        parent.helpText = (
            "Load and save MLArray (.mla) volumes through Slicer's normal file dialogs."
        )
        parent.acknowledgementText = (
            "Implemented as a scripted Python file reader/writer for 3D Slicer."
        )
        parent.hidden = True


class SlicerMLArrayFileReader:
    """Slicer scripted file reader for `.mla` volumes."""

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return FILE_DESCRIPTION

    def fileType(self):
        return FILE_TYPE

    def extensions(self):
        return [FILE_EXTENSION_FILTER]

    def canLoadFileConfidence(self, filePath):
        if not self.parent.supportedNameFilters(filePath):
            return 0.0
        return reader_confidence_for_path(filePath)

    def load(self, properties):
        try:
            loaded_node = load_mla_volume(
                properties["fileName"],
                explicit_name=properties.get("name"),
            )
        except Exception as exc:
            traceback.print_exc()
            self.parent.userMessages().AddMessage(
                vtk.vtkCommand.ErrorEvent,
                f"Failed to load MLArray file: {exc}",
            )
            return False

        self.parent.loadedNodes = [loaded_node.GetID()]
        return True


class SlicerMLArrayFileWriter:
    """Slicer scripted file writer for exporting volumes to `.mla`."""

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return FILE_DESCRIPTION

    def fileType(self):
        return FILE_TYPE

    def extensions(self, obj):
        if writer_confidence_for_node(obj) <= 0.0:
            return []
        return [FILE_WRITER_EXTENSION_FILTER]

    def canWriteObjectConfidence(self, obj):
        return writer_confidence_for_node(obj)

    def write(self, properties):
        try:
            node = slicer.mrmlScene.GetNodeByID(properties["nodeID"])
            write_mla_volume(node, normalize_mla_path(properties["fileName"]))
        except Exception as exc:
            traceback.print_exc()
            self.parent.userMessages().AddMessage(
                vtk.vtkCommand.ErrorEvent,
                f"Failed to save MLArray file: {exc}",
            )
            return False

        self.parent.writtenNodes = [node.GetID()]
        return True
