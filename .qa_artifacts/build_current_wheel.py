from __future__ import annotations

import os
import sys

from pip._internal.cli.main import main as pip_main


msvc_bin = (
    r"C:\Program Files\Microsoft Visual Studio\18\Community"
    r"\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
)
sdk_bin = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
anaconda_scripts = r"C:\Users\anish\anaconda3\Scripts"

os.environ["PATH"] = os.pathsep.join(
    [msvc_bin, sdk_bin, anaconda_scripts, os.environ["PATH"]]
)
os.environ["CMAKE_GENERATOR"] = "Ninja"
os.environ["CC"] = os.path.join(msvc_bin, "cl.exe")
os.environ["CXX"] = os.path.join(msvc_bin, "cl.exe")

sys.exit(
    pip_main(
        [
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "-w",
            r".qa_artifacts\msvc-wheel-current",
        ]
    )
)
