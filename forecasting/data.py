"""Data loading helpers for the forecasting package."""
from __future__ import annotations

from typing import Any

import pandas as pd

from cap_store import load_df


def _safe_df(key: str) -> pd.DataFrame:
    """Return a DataFrame for the given key or an empty frame if missing."""
    df = load_df(key)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def load_training_frame(plan_id: int) -> pd.DataFrame:
    """Fetch historical interval/day data for a plan."""
    return _safe_df(f"plan_{plan_id}_fw")


def load_feature_frame(plan_id: int) -> pd.DataFrame:
    """Return any engineered features that should accompany training data."""
    return _safe_df(f"plan_{plan_id}_features")
