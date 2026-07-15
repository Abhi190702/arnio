import os
import sys


sys.path.append(r"C:\Users\anish\anaconda3\Lib\site-packages")
_dll_handle = os.add_dll_directory(r"C:\Users\anish\anaconda3\Library\bin")

# A global editable Arnio install registers a scikit-build meta-path finder
# ahead of PathFinder. Remove it so this runner genuinely imports the wheel
# site selected by sitecustomize/PYTHONPATH.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_arnio_editable"
]

# Pin the wheel package before pytest imports repository test modules.
import arnio

expected_site = os.path.normcase(
    os.path.abspath(
        os.environ.get(
            "ARNIO_WHEEL_SITE",
            r"C:\Users\anish\Desktop\arnio"
            r"\.qa_artifacts\wheel-smoke-clean\Lib\site-packages",
        )
    )
)
if not os.path.normcase(os.path.abspath(arnio.__file__)).startswith(expected_site):
    raise RuntimeError(f"Tests are not using the isolated wheel: {arnio.__file__}")

# Repository-only test helpers remain available without allowing the checkout
# to replace the already imported wheel package.
sys.path.insert(0, r"C:\Users\anish\Desktop\arnio")

import pytest


raise SystemExit(pytest.main(sys.argv[1:]))
