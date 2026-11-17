"""Fixtures for tests that run real model weights.

Everything under tests/model is marked ``model`` and is skipped automatically
when the ``models`` extra is not installed or the weights cannot be loaded
(for example offline with a cold cache). CI runs ``-m "not model"``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from pii_extract.backends import create_backend


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(pytest.mark.model)


def _backend(name: str, package: str):
    if os.environ.get("PII_SKIP_MODEL_TESTS") == "1":
        pytest.skip("PII_SKIP_MODEL_TESTS=1")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec(package) is None:
        pytest.skip(f"the `models` extra ({package}, torch) is not installed")
    backend = create_backend(name, device=os.environ.get("PII_DEVICE", "auto"))
    try:
        backend.load()
    except Exception as error:
        pytest.skip(f"could not load {backend.model_id}: {type(error).__name__}: {error}")
    return backend


@pytest.fixture(scope="session")
def gliner2_backend():
    backend = _backend("gliner2", "gliner2")
    yield backend
    backend.close()


@pytest.fixture(scope="session")
def gliner_backend():
    backend = _backend("gliner", "gliner")
    yield backend
    backend.close()


@pytest.fixture(params=["gliner2", "gliner"])
def backend(request):
    return request.getfixturevalue(f"{request.param}_backend")
