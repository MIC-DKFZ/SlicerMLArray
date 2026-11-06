"""Errors shown to the user when a file cannot be loaded or a node cannot be saved."""


class UnsupportedFile(ValueError):
    """The file (or node) has a layout Slicer cannot show (or MLArray cannot store) without guessing."""


class MissingDependency(RuntimeError):
    """``mlarray`` (1.x) is not installed in Slicer's Python."""
