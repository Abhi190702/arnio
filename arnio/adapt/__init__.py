"""arnio.adapt — Adapter layer for DataFrame abstraction.

Public API:
    resolve_adapter  — Auto-detect input type and return the correct adapter.
"""

from arnio.adapt._detect import resolve_adapter

__all__ = [
    "resolve_adapter",
]
