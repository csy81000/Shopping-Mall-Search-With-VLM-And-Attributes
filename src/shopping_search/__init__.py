"""VLM-based conversational shopping retrieval."""

from typing import Any


__all__ = ["CatalogIndex"]


def __getattr__(name: str) -> Any:
    """Load FAISS-dependent exports only when they are requested."""

    if name == "CatalogIndex":
        from .catalog import CatalogIndex

        return CatalogIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
