import importlib.metadata
import importlib.util
import os
import sys

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


@pytest.fixture(scope="session")
def upstream():
    distribution = importlib.metadata.distribution("roman")
    source = distribution.locate_file("roman/__init__.py")
    spec = importlib.util.spec_from_file_location("_upstream_roman", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
