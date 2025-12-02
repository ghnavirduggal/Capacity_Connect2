"""Shared dataclasses and interfaces for forecasting workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Protocol


@dataclass(slots=True)
class ForecastJob:
    """Execution context describing which plan/scope we are forecasting."""

    plan_id: int
    horizon_weeks: int = 8
    channels: Iterable[str] | None = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForecastResult:
    """Container for eventual forecast outputs and metadata."""

    job: ForecastJob
    dataframe: Any | None = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ForecastingStrategy(Protocol):
    """Minimal protocol every forecasting strategy should obey."""

    name: str

    def fit(self, job: ForecastJob, data) -> Any:  # pragma: no cover - interface only
        """Train/prepare the strategy for the supplied job scope."""

    def predict(self, job: ForecastJob, horizon: int) -> ForecastResult:  # pragma: no cover
        """Return a `ForecastResult` for the requested horizon."""
