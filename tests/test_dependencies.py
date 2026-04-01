from __future__ import annotations

import types

import pytest

from slicermlarray.dependencies import ensure_mlarray_available, import_mlarray_module
from slicermlarray.exceptions import DependencyError


def _fake_mlarray_module():
    return types.SimpleNamespace(MLArray=object(), Meta=object(), MetaSpatial=object())


def test_import_mlarray_module_success():
    module = import_mlarray_module(importer=lambda name: _fake_mlarray_module())
    assert hasattr(module, "MLArray")


def test_import_mlarray_module_missing_package():
    def importer(_: str):
        raise ModuleNotFoundError("no module named mlarray")

    with pytest.raises(DependencyError, match="not installed"):
        import_mlarray_module(importer=importer)


def test_import_mlarray_module_incompatible_package():
    module = types.SimpleNamespace(MLArray=object())
    with pytest.raises(DependencyError, match="incompatible"):
        import_mlarray_module(importer=lambda name: module)


def test_ensure_mlarray_available_without_install():
    def importer(_: str):
        raise ModuleNotFoundError("missing")

    with pytest.raises(DependencyError, match="not installed"):
        ensure_mlarray_available(auto_install=False, importer=importer)


def test_ensure_mlarray_available_prompt_declined():
    attempts = {"count": 0}

    def importer(_: str):
        attempts["count"] += 1
        raise ModuleNotFoundError("missing")

    with pytest.raises(DependencyError, match="declined"):
        ensure_mlarray_available(
            auto_install=True,
            importer=importer,
            installer=lambda requirement: None,
            prompt=lambda message: False,
        )

    assert attempts["count"] == 1


def test_ensure_mlarray_available_installs_and_retries():
    state = {"installed": False}

    def importer(_: str):
        if not state["installed"]:
            raise ModuleNotFoundError("missing")
        return _fake_mlarray_module()

    def installer(_: str):
        state["installed"] = True

    module = ensure_mlarray_available(
        auto_install=True,
        importer=importer,
        installer=installer,
        prompt=lambda message: True,
    )

    assert hasattr(module, "MLArray")


def test_ensure_mlarray_available_install_failure():
    def importer(_: str):
        raise ModuleNotFoundError("missing")

    with pytest.raises(DependencyError, match="Failed to install"):
        ensure_mlarray_available(
            auto_install=True,
            importer=importer,
            installer=lambda requirement: (_ for _ in ()).throw(RuntimeError("boom")),
            prompt=lambda message: True,
        )
