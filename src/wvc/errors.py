"""The root of every error wvc raises on purpose.

A leaf module (standard library only), so that every module defining an error can import it.
``wvc.cli.main`` turns a ``WvcError`` into a ``wvc: ...`` message and exit code 2; any other
exception is a bug and propagates with its traceback. ``WvcError`` derives from ``ValueError``
so that code catching ``ValueError`` keeps catching wvc's errors. The object store's errors
are defined here too, so that the modules below ``wvc.records`` can raise them without
importing it; ``wvc.records`` and ``wvc.store`` re-export them.
"""

from __future__ import annotations


class WvcError(ValueError):
    """An error wvc reports to the user: bad input, a damaged store, a refused operation."""


class StoreError(WvcError):
    """The repository is missing or malformed, or lacks what was asked for."""


class IntegrityError(StoreError):
    """Stored content does not match the id it is stored under."""


class DeltaDepthError(StoreError):
    """A delta would make its chain longer than the repository's ``max_delta_depth``."""
