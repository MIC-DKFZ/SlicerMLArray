"""``mlarray`` 1.x inside Slicer's Python, installed on first use after asking."""

from __future__ import annotations

import importlib
import sys

from .errors import MissingDependency

REQUIREMENT = "mlarray>=1.0.0,<2"


def _major(version: str) -> int:
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return 0


def mlarray_module():
    """The ``mlarray`` module, or ``MissingDependency`` if it is missing or not 1.x."""
    try:
        module = importlib.import_module("mlarray")
    except ImportError as exc:
        raise MissingDependency("the Python package 'mlarray' is not installed") from exc
    version = getattr(module, "__version__", "0")
    if _major(version) != 1:
        raise MissingDependency(f"mlarray {version} is installed; MLArray files need mlarray 1.x")
    return module


def ensure_mlarray(ask=None, install=None):
    """``mlarray``, installing ``REQUIREMENT`` first if ``ask(message)`` agrees (both callbacks from Slicer)."""
    try:
        return mlarray_module()
    except MissingDependency as error:
        if ask is None or install is None:
            raise
        if not ask(f"Opening and saving MLArray files needs the Python package '{REQUIREMENT}' ({error}). "
                   "Install it now?"):
            raise
        install(REQUIREMENT)
        importlib.invalidate_caches()
        for name in [n for n in list(sys.modules) if n == "mlarray" or n.startswith("mlarray.")]:
            del sys.modules[name]
        try:
            return mlarray_module()
        except MissingDependency as exc:
            raise MissingDependency(f"{exc} after installing; restart Slicer and try again") from exc
