"""Pipeline orchestration for forecasting jobs."""
from __future__ import annotations

from typing import Callable

from .base import ForecastJob, ForecastResult
from . import data
from .models import get_strategy


def build_forecast_pipeline(strategy_name: str) -> Callable[[ForecastJob], ForecastResult]:
    """Return a callable that executes the requested strategy."""
    strategy = get_strategy(strategy_name)

    def _runner(job: ForecastJob) -> ForecastResult:
        frame = data.load_training_frame(job.plan_id)
        strategy.fit(job, frame)
        return strategy.predict(job, job.horizon_weeks)

    return _runner


def run_forecast_job(job: ForecastJob, strategy_name: str = "naive_carry_forward") -> ForecastResult:
    pipeline = build_forecast_pipeline(strategy_name)
    return pipeline(job)
