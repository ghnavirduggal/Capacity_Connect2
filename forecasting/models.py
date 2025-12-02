"""Placeholder model registry for forecasting strategies."""
from __future__ import annotations

from typing import Dict

from .base import ForecastJob, ForecastResult, ForecastingStrategy


class NaiveCarryForward(ForecastingStrategy):
    """Example baseline: repeat last known values forward."""

    name = "naive_carry_forward"

    def __init__(self):
        self._last_observation = None

    def fit(self, job: ForecastJob, data):
        if getattr(data, "empty", True):
            self._last_observation = None
        else:
            self._last_observation = data.iloc[:, -1]
        return self

    def predict(self, job: ForecastJob, horizon: int) -> ForecastResult:
        from .base import ForecastResult  # local import to avoid cycles
        frame = None
        if self._last_observation is not None:
            frame = self._last_observation.to_frame().T
        return ForecastResult(job=job, dataframe=frame, diagnostics={"model": self.name})


_REGISTRY: Dict[str, ForecastingStrategy] = {
    NaiveCarryForward.name: NaiveCarryForward(),
}


def get_strategy(name: str) -> ForecastingStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown forecasting strategy '{name}'")
    return _REGISTRY[name]
