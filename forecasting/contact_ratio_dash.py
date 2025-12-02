from __future__ import annotations
import io
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR, SARIMAX
from xgboost import XGBRegressor

import config_manager


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prep_input(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = {str(c).strip().lower(): c for c in d.columns}
    ds_col = cols.get("ds") or cols.get("date")
    if not ds_col:
        raise ValueError("Input must contain a 'ds' or 'date' column.")
    d["ds"] = pd.to_datetime(d[ds_col], errors="coerce")
    d = d.dropna(subset=["ds"])
    if "final_smoothed_value" not in cols and "y" in cols:
        d["Final_Smoothed_Value"] = _safe_num(d[cols["y"]]) * 100.0
    else:
        d["Final_Smoothed_Value"] = _safe_num(d.get(cols.get("final_smoothed_value", "Final_Smoothed_Value"), 0.0))
    d["IQ_value"] = _safe_num(d.get(cols.get("iq_value", "IQ_value"), 1.0))
    # holidays optional
    hcol = cols.get("holiday_month_start") or cols.get("holiday")
    if hcol:
        d["holiday_month_start"] = pd.to_datetime(d[hcol], errors="coerce")
    else:
        d["holiday_month_start"] = pd.NaT
    d["Year"] = d["ds"].dt.year
    d["Month"] = d["ds"].dt.strftime("%b")
    # scale IQ
    scaler = MinMaxScaler()
    d["IQ_value_scaled"] = scaler.fit_transform(d[["IQ_value"]]).round(4)
    return d.reset_index(drop=True)


def prophet_forecast(train: pd.DataFrame, periods: int, config: dict, holidays_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    m = Prophet(
        changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=config.get("seasonality_prior_scale", 0.1),
        holidays_prior_scale=config.get("holidays_prior_scale", 0.1),
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays_df if config.get("use_holidays", True) else None,
    )
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in train.columns:
        m.add_regressor("IQ_value_scaled")
    m.add_seasonality(name="monthly", period=30.5, fourier_order=config.get("monthly_fourier_order", 5))
    m.fit(train[["ds", "y"] + ([ "IQ_value_scaled"] if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in train.columns else [])])
    future = m.make_future_dataframe(periods=periods, freq="MS")
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in train.columns:
        last_val = train["IQ_value_scaled"].iloc[-1]
        future["IQ_value_scaled"] = list(train["IQ_value_scaled"]) + [last_val] * periods
    fc = m.predict(future)
    return fc[["ds", "yhat"]].iloc[len(train):]


def _add_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["ds"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def rf_forecast(train: pd.DataFrame, periods: int, config: dict) -> pd.DataFrame:
    t = _add_monthly_features(train)
    X = np.arange(len(t)).reshape(-1, 1)
    X = np.hstack([X, t[["month_sin", "month_cos"]].values])
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        X = np.hstack([X, t[["IQ_value_scaled"]].values])
    y = t["y"].values
    future_dates = pd.date_range(start=t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fut = pd.DataFrame({"ds": future_dates})
    fut = _add_monthly_features(fut)
    Xf = np.arange(len(t), len(t) + periods).reshape(-1, 1)
    Xf = np.hstack([Xf, fut[["month_sin", "month_cos"]].values])
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        last_val = t["IQ_value_scaled"].iloc[-1]
        Xf = np.hstack([Xf, np.full((periods, 1), last_val)])
    model = RandomForestRegressor(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=int(config.get("max_depth", 5)),
        random_state=int(config.get("random_state", 42)),
    )
    model.fit(X, y)
    y_pred = model.predict(Xf)
    return pd.DataFrame({"ds": future_dates, "yhat": y_pred})


def xgb_forecast(train: pd.DataFrame, periods: int, config: dict) -> pd.DataFrame:
    t = _add_monthly_features(train)
    X = np.arange(len(t)).reshape(-1, 1)
    X = np.hstack([X, t[["month_sin", "month_cos"]].values])
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        X = np.hstack([X, t[["IQ_value_scaled"]].values])
    y = t["y"].values
    future_dates = pd.date_range(start=t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fut = pd.DataFrame({"ds": future_dates})
    fut = _add_monthly_features(fut)
    Xf = np.arange(len(t), len(t) + periods).reshape(-1, 1)
    Xf = np.hstack([Xf, fut[["month_sin", "month_cos"]].values])
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        last_val = t["IQ_value_scaled"].iloc[-1]
        Xf = np.hstack([Xf, np.full((periods, 1), last_val)])
    model = XGBRegressor(
        n_estimators=int(config.get("n_estimators", 200)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_depth=int(config.get("max_depth", 3)),
        random_state=int(config.get("random_state", 42)),
        subsample=float(config.get("subsample", 1.0)),
        colsample_bytree=float(config.get("colsample_bytree", 1.0)),
        objective="reg:squarederror",
    )
    model.fit(X, y)
    y_pred = model.predict(Xf)
    return pd.DataFrame({"ds": future_dates, "yhat": y_pred})


def var_forecast(train: pd.DataFrame, periods: int, config: dict) -> pd.DataFrame:
    t = _add_monthly_features(train)
    vars_use = ["y"]
    if config.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
        vars_use.append("IQ_value_scaled")
    vars_use.extend(["month_sin", "month_cos"])
    model = VAR(t[vars_use].dropna())
    lag = int(config.get("lags", 12))
    results = model.fit(lag)
    fc = results.forecast(t[vars_use].iloc[-lag:].values, steps=periods)
    future_dates = pd.date_range(t["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"ds": future_dates, "yhat": fc[:, 0]})


def sarimax_forecast(train: pd.DataFrame, periods: int, config: dict, exog_cols: list[str]) -> pd.DataFrame:
    exog_train = train[exog_cols] if exog_cols else None
    order = tuple(config.get("order", (1, 1, 1)))
    seasonal = tuple(config.get("seasonal_order", (1, 1, 1, 12)))
    model = SARIMAX(train["y"], exog=exog_train, order=order, seasonal_order=seasonal)
    results = model.fit(disp=False)
    exog_forecast = None
    if exog_cols:
        last_vals = {c: train[c].iloc[-1] for c in exog_cols}
        exog_forecast = pd.DataFrame({c: [last_vals[c]] * periods for c in exog_cols})
    fc = results.get_forecast(steps=periods, exog=exog_forecast)
    future_dates = pd.date_range(train["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"ds": future_dates, "yhat": fc.predicted_mean})


def run_phase2_forecast(df: pd.DataFrame, forecast_months: int, config: Optional[dict] = None) -> Dict[str, Any]:
    cfg = config or config_manager.load_config()
    rf_cfg = cfg.get("random_forest", {})
    xgb_cfg = cfg.get("xgboost", {})
    var_cfg = cfg.get("var", {})
    sarimax_cfg = cfg.get("sarimax", {})
    prop_cfg = cfg.get("prophet", {})
    gen_cfg = cfg.get("general", {})

    t = _prep_input(df)
    t = t.sort_values("ds")
    t["holiday_month_start_flag"] = t["holiday_month_start"].notna().astype(int)
    t["y"] = _safe_num(t["Final_Smoothed_Value"]) / 100.0

    results: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    holidays_df = None
    if prop_cfg.get("use_holidays", True):
        if "holiday_month_start" in t.columns and t["holiday_month_start"].notna().any():
            holidays_df = pd.DataFrame({"ds": t["holiday_month_start"].dropna().unique()})

    try:
        results["prophet"] = prophet_forecast(t, forecast_months, prop_cfg, holidays_df)
    except Exception as exc:
        errors["prophet"] = str(exc)
    try:
        results["random_forest"] = rf_forecast(t, forecast_months, rf_cfg)
    except Exception as exc:
        errors["random_forest"] = str(exc)
    try:
        results["xgboost"] = xgb_forecast(t, forecast_months, xgb_cfg)
    except Exception as exc:
        errors["xgboost"] = str(exc)
    try:
        results["var"] = var_forecast(t, forecast_months, var_cfg)
    except Exception as exc:
        errors["var"] = str(exc)
    try:
        exog_cols = []
        if sarimax_cfg.get("use_iq_value_scaled", True) and "IQ_value_scaled" in t.columns:
            exog_cols.append("IQ_value_scaled")
        if sarimax_cfg.get("use_holidays", True):
            exog_cols.append("holiday_month_start_flag")
        results["sarimax"] = sarimax_forecast(t, forecast_months, sarimax_cfg, exog_cols)
    except Exception as exc:
        errors["sarimax"] = str(exc)

    # attach smoothing values for downstream pivot
    t["Date"] = pd.to_datetime(t["ds"]).dt.to_period("M").dt.to_timestamp()
    results["final_smoothed_values"] = t[["Final_Smoothed_Value", "Date"]].rename(columns={"Final_Smoothed_Value": "Final_Smoothed_Value"})

    return {
        "forecast_results": results,
        "errors": errors,
        "config": cfg,
    }
