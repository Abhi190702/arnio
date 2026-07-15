import os
import sys


repo_root = os.path.normcase(
    os.path.abspath(r"C:\Users\anish\Desktop\arnio")
)
wheel_site = os.path.normcase(
    os.path.abspath(
        os.environ.get(
            "ARNIO_WHEEL_SITE",
            r"C:\Users\anish\Desktop\arnio"
            r"\.qa_artifacts\wheel-smoke-clean\Lib\site-packages",
        )
    )
)

sys.path[:] = [
    path
    for path in sys.path
    if os.path.normcase(os.path.abspath(path or os.curdir)) != repo_root
]

for index, path in enumerate(list(sys.path)):
    if os.path.normcase(os.path.abspath(path)) == wheel_site:
        sys.path.insert(0, sys.path.pop(index))
        break

sys.path.append(r"C:\Users\anish\anaconda3\Lib\site-packages")
_dll_handle = os.add_dll_directory(
    r"C:\Users\anish\anaconda3\Library\bin"
)

# `python -m` restores an empty-string cwd entry after site customization.
# Move away only when launched from the repository root; example scripts rely
# on their configured data-directory working directories.
if os.path.normcase(os.path.abspath(os.curdir)) == repo_root:
    os.chdir(os.path.dirname(__file__))
