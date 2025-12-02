"""Small helpers shared across the forecasting package."""
from __future__ import annotations

from typing import Iterable


def canonical_channels(channels: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize user-specified channel identifiers."""
    if not channels:
        return tuple()
    return tuple(sorted({str(c).strip().lower() for c in channels if str(c).strip()}))
