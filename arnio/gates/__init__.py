"""arnio.gates — Quality gates for CI/CD.

Public API:
    check  — Assertion-style validation that raises on failure.
"""

from arnio.gates._gate import check

__all__ = [
    "check",
]
