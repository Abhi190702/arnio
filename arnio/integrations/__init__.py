"""arnio.integrations — Third-party integrations.

The pandas accessor is registered on import of the arnio package.
"""

from arnio.integrations._pandas_accessor import ArnioPandasAccessor

__all__ = [
    "ArnioPandasAccessor",
]
