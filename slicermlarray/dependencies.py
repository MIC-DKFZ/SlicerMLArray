"""Runtime dependency management for Slicer."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from .exceptions import DependencyError

MLARRAY_INSTALL_REQUIREMENT = "mlarray"
MLARRAY_INSTALL_PROMPT = (
    "MLArray support requires the Python package 'mlarray'. "
    "Install it now into Slicer's Python environment?"
)


def import_mlarray_module(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import and validate the `mlarray` runtime module."""
    try:
        module = importer("mlarray")
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "The Python package 'mlarray' is not installed in this Python environment."
        ) from exc

    required_names = ("MLArray", "Meta", "MetaSpatial")
    missing_names = [name for name in required_names if not hasattr(module, name)]
    if missing_names:
        raise DependencyError(
            "The installed 'mlarray' package is incompatible with this extension. "
            f"Missing attributes: {', '.join(missing_names)}."
        )
    return module


def ensure_mlarray_available(
    *,
    auto_install: bool,
    installer: Callable[[str], None] | None = None,
    prompt: Callable[[str], bool] | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
    invalidate_caches: Callable[[], None] = importlib.invalidate_caches,
) -> ModuleType:
    """Ensure that `mlarray` can be imported, optionally installing it first."""
    try:
        return import_mlarray_module(importer=importer)
    except DependencyError as initial_error:
        if not auto_install:
            raise
        if installer is None:
            raise DependencyError(
                "The 'mlarray' package is missing and no installer callback was provided."
            ) from initial_error
        if prompt is not None and not prompt(MLARRAY_INSTALL_PROMPT):
            raise DependencyError(
                "MLArray support requires the Python package 'mlarray'. "
                "Installation was declined."
            ) from initial_error

        try:
            installer(MLARRAY_INSTALL_REQUIREMENT)
        except Exception as exc:  # pragma: no cover - error path is unit-tested via mocks
            raise DependencyError(
                "Failed to install 'mlarray' into Slicer's Python environment."
            ) from exc

        invalidate_caches()
        try:
            return import_mlarray_module(importer=importer)
        except DependencyError as exc:
            raise DependencyError(
                "The dependency installation finished, but importing 'mlarray' still failed. "
                "Restart Slicer and try again."
            ) from exc
