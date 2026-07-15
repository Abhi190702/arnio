import os
import sys


import arnio

expected_site = os.environ["ARNIO_PY314_SITE"]
if expected_site not in arnio.__file__:
    raise RuntimeError(
        f"Tests are not using the Python 3.14 wheel: {arnio.__file__}"
    )

if sys.version_info[:2] != (3, 14):
    raise RuntimeError(f"Expected Python 3.14, got {sys.version}")

sys.path.insert(0, r"C:\Users\anish\Desktop\arnio")

import pytest


raise SystemExit(pytest.main(sys.argv[1:]))
