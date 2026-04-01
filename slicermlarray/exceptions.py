"""Project-specific exception types."""


class SlicerMLArrayError(RuntimeError):
    """Base error for the package."""


class UnsupportedLayoutError(SlicerMLArrayError, ValueError):
    """Raised when an MLArray layout cannot be mapped safely to Slicer."""


class DependencyError(SlicerMLArrayError):
    """Raised when required runtime dependencies are unavailable."""
