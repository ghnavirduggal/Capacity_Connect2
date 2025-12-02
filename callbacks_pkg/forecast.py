from __future__ import annotations
import base64
import io
import json
from typing import Optional, Tuple

import dash
from dash import Input, Output, State, no_update, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from app_instance import app
from forecasting.process_and_IQ_data import (
    forecast_group_pivot_and_long_style,
    process_forecast_results,
    plot_contact_ratio_seasonality,
    accuracy_phase1,
    fill_final_smoothed_row,
    create_download_csv_with_metadata,
)
from forecasting.contact_ratio_dash import run_phase2_forecast
import config_manager
import os


def _parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, str]:
    if not contents or "," not in contents:
        return pd.DataFrame(), "No file supplied."
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        lower = filename.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            # Try common engines for xlsx/xls
            try:
                df = pd.read_excel(io.BytesIO(decoded), engine="openpyxl")
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(decoded))
                except Exception:
                    # Fallback: attempt CSV decode
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        msg = f"Loaded {len(df):,} rows from {filename}."
        return df, msg
    except Exception as exc:
        return pd.DataFrame(), f"Failed to read {filename}: {exc}"


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for nm in candidates:
        col = low.get(str(nm).strip().lower())
        if col:
            return col
    return None


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    d = df.copy()
    date_col = _pick_col(d, ("date", "ds", "datetime", "timestamp"))
    val_col = _pick_col(d, ("volume", "items", "calls", "count", "value"))

    if date_col and val_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col])
        d["_month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
        d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
        grouped = (
            d.groupby("_month", as_index=False)[val_col]
            .sum()
            .rename(columns={val_col: "Total"})
        )
        grouped["Month"] = grouped["_month"].dt.strftime("%b %Y")
        grouped = grouped[["Month", "Total"]]
        return grouped

    stats = d.describe(include="all").reset_index().rename(columns={"index": "metric"})
    return stats


def _empty_fig(msg: str = ""):
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, showarrow=False)
    fig.update_layout(margin=dict(t=30, l=20, r=20, b=20))
    return fig


def _cols(df: pd.DataFrame):
    return [{"name": c, "id": c} for c in df.columns]


def _ratio_fig(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return _empty_fig("No ratio data")
    melted = df.melt(id_vars="Year", var_name="Month", value_name="Ratio")
    melted = melted.dropna(subset=["Ratio"])
    if melted.empty:
        return _empty_fig("No ratio data")
    fig = px.line(melted, x="Month", y="Ratio", color="Year", markers=True, title=title)
    fig.update_traces(mode="lines+markers")
    return fig


def _smoothing_core(df: pd.DataFrame, window: int, threshold: float, prophet_order: Optional[int] = None):
    """Run EWMA smoothing (or Prophet) + anomaly detection and seasonality pivots."""
    date_col = _pick_col(df, ("date", "ds", "timestamp"))
    val_col = _pick_col(df, ("final_smoothed_value", "volume", "value", "items", "calls", "count", "y"))
    if not date_col or not val_col:
        raise ValueError("Expected columns for date and volume.")

    d = df.copy()
    d["ds"] = pd.to_datetime(d[date_col], errors="coerce")
    d["y"] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=["ds", "y"]).sort_values("ds")

    if prophet_order:
        from prophet import Prophet

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=int(prophet_order or 5))
        m.fit(d[["ds", "y"]])
        preds = m.predict(d[["ds"]])
        d["smoothed"] = preds["yhat"]
    else:
        span = max(int(window or 6), 1)
        d["smoothed"] = d["y"].ewm(span=span, adjust=False).mean()

    resid = d["y"] - d["smoothed"]
    std = resid.std() or 1e-9
    d["zscore"] = (resid - resid.mean()) / std
    d["is_anomaly"] = d["zscore"].abs() > float(threshold or 3.0)

    d["Year"] = d["ds"].dt.year
    d["Month"] = d["ds"].dt.strftime("%b")
    pivot = d.pivot_table(index="Year", columns="Month", values="smoothed", aggfunc="mean").reset_index()
    pivot = pivot.fillna(0)

    ratio_fig, capped, ratio = None, pd.DataFrame(), pd.DataFrame()
    try:
        ratio_fig, capped, ratio = plot_contact_ratio_seasonality(pivot)
    except Exception:
        ratio_fig = _empty_fig("Seasonality not available")

    ratio_disp = ratio.copy()
    capped_disp = capped.copy()
    for col in ratio_disp.columns:
        if col != "Year":
            ratio_disp[col] = pd.to_numeric(ratio_disp[col], errors="coerce").round(2)
    for col in capped_disp.columns:
        if col != "Year":
            capped_disp[col] = pd.to_numeric(capped_disp[col], errors="coerce").round(2)

    anomalies = d[d["is_anomaly"]][["ds", "y", "smoothed", "zscore"]]
    smoothed_tbl = d[["ds", "y", "smoothed", "zscore", "is_anomaly"]]

    fig_series = px.line(
        d,
        x="ds",
        y=["y", "smoothed"],
        labels={"value": "Volume", "ds": "Date", "variable": "Series"},
        title="Smoothed vs Original",
    )
    if not anomalies.empty:
        fig_series.add_scatter(
            x=anomalies["ds"],
            y=anomalies["y"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=10, symbol="x"),
        )

    return {
        "fig_series": fig_series,
        "fig_ratio1": _ratio_fig(ratio_disp, "Normalized Ratio 1"),
        "fig_ratio2": _ratio_fig(capped_disp, "Normalized Ratio 2 (capped)"),
        "ratio": ratio_disp,
        "capped": capped_disp,
        "smoothed": smoothed_tbl,
        "anomalies": anomalies,
        "pivot": pivot,
    }


_TRANSFORMATION_COLS = [
    "Transformation 1",
    "Remarks_Tr 1",
    "Transformation 2",
    "Remarks_Tr 2",
    "Transformation 3",
    "Remarks_Tr 3",
    "IA 1",
    "Remarks_IA 1",
    "IA 2",
    "Remarks_IA 2",
    "IA 3",
    "Remarks_IA 3",
    "Marketing Campaign 1",
    "Remarks_Mkt 1",
    "Marketing Campaign 2",
    "Remarks_Mkt 2",
    "Marketing Campaign 3",
    "Remarks_Mkt 3",
]


def _apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    def calculate_sequential(row: pd.Series, field: str, base_col: str):
        try:
            value = row.get(field, "")
            if pd.notna(value) and str(value).strip() != "":
                adj_percent = float(str(value).replace("%", "").strip())
                return round(row[base_col] * (1 + adj_percent / 100), 0)
        except Exception:
            pass
        return round(row[base_col], 0)

    def apply_sequential_adjustments(df_in: pd.DataFrame) -> pd.DataFrame:
        df_copy = df_in.copy()
        adjustment_fields = [
            "Transformation 1",
            "Transformation 2",
            "Transformation 3",
            "IA 1",
            "IA 2",
            "IA 3",
            "Marketing Campaign 1",
            "Marketing Campaign 2",
            "Marketing Campaign 3",
        ]
        prev_col = "Base_Forecast_for_Forecast_Group"
        for field in adjustment_fields:
            new_col = f"Forecast_{field}"
            df_copy[new_col] = df_copy.apply(lambda row: calculate_sequential(row, field, prev_col), axis=1)
            prev_col = new_col
        return df_copy

    processed = apply_sequential_adjustments(df)
    # convenience final column
    if "Forecast_Marketing Campaign 3" in processed.columns:
        processed["Final_Forecast"] = processed["Forecast_Marketing Campaign 3"]
    return processed



@app.callback(
    Output("vs-upload-msg", "children"),
    Output("vs-preview", "data"),
    Output("vs-preview", "columns"),
    Output("vs-data-store", "data"),
    Input("vs-upload", "contents"),
    State("vs-upload", "filename"),
    prevent_initial_call=True,
)
def _on_vs_upload(contents, filename):
    if not contents or not filename:
        raise dash.exceptions.PreventUpdate
    df, msg = _parse_upload(contents, filename)
    preview = df.head(50)
    cols = [{"name": c, "id": c} for c in preview.columns]
    store = df.to_json(date_format="iso", orient="split") if not df.empty else None
    return msg, preview.to_dict("records"), cols, store


@app.callback(
    Output("vs-summary", "data"),
    Output("vs-summary", "columns"),
    Output("vs-alert", "children"),
    Output("vs-alert", "is_open"),
    Output("vs-next-modal", "is_open"),
    Output("vs-category", "options"),
    Output("vs-category", "value"),
    Output("vs-pivot", "data"),
    Output("vs-pivot", "columns"),
    Output("vs-volume-split", "data"),
    Output("vs-volume-split", "columns"),
    Output("vs-results-store", "data"),
    Input("vs-run-btn", "n_clicks"),
    State("vs-data-store", "data"),
    State("vs-next-modal", "is_open"),
    prevent_initial_call=True,
)
def _run_volume_summary(n_clicks, data_json, modal_open):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return ([], [], "Upload data to run the summary.", True, False, [], None, [], [], [], [], None)
    try:
        df = pd.read_json(io.StringIO(data_json), orient="split")
    except Exception:
        return ([], [], "Could not read cached data.", True, False, [], None, [], [], [], [], None)

    summary = _summarize(df)
    cols = [{"name": c, "id": c} for c in summary.columns]
    alert_text = "Summary complete. Review results below."

    # Normalize column names for downstream helpers
    df_norm = df.copy()
    df_norm.columns = [str(c).strip().lower() for c in df_norm.columns]
    cat_col = _pick_col(df_norm, ("category", "forecast_group", "queue_name"))
    if not cat_col:
        df_norm["category"] = "All"
        cat_col = "category"
    categories = sorted(df_norm[cat_col].dropna().astype(str).unique().tolist())
    options = [{"label": c, "value": c} for c in categories]
    chosen = categories[0] if categories else None

    def _safe_pivots(cat: str):
        try:
            piv, split, long_orig, long_monthly, long_daily = forecast_group_pivot_and_long_style(df_norm.rename(columns={cat_col: "category"}), cat)
            return piv, split
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    piv0, split0 = _safe_pivots(chosen) if chosen else (pd.DataFrame(), pd.DataFrame())
    pivot_cols = [{"name": c, "id": c} for c in (piv0.columns if not piv0.empty else [])]
    split_cols = [{"name": c, "id": c} for c in (split0.columns if not split0.empty else [])]

    results_store = {"categories": categories, "cat_col": cat_col, "data": data_json}

    return (
        summary.to_dict("records"),
        cols,
        alert_text,
        True,
        True,
        options,
        chosen,
        (piv0.to_dict("records") if not piv0.empty else []),
        pivot_cols,
        (split0.to_dict("records") if not split0.empty else []),
        split_cols,
        json.dumps(results_store),
    )


def _toggle_modal(btn_id: str, modal_id: str):
    @app.callback(
        Output(modal_id, "is_open"),
        Input(btn_id, "n_clicks"),
        Input(f"{btn_id}-close", "n_clicks"),
        State(modal_id, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(n_open, n_close, is_open):
        if not n_open and not n_close:
            raise dash.exceptions.PreventUpdate
        if dash.callback_context.triggered_id == f"{btn_id}-close":
            return False
        return not bool(is_open)


for _btn, _modal in [
    ("sa-complete", "sa-complete-modal"),
    ("fc-complete", "fc-complete-modal"),
    ("tp-complete", "tp-complete-modal"),
    ("di-complete", "di-complete-modal"),
]:
    _toggle_modal(_btn, _modal)


@app.callback(
    Output("vs-next-modal", "is_open", allow_duplicate=True),
    Input("vs-modal-close", "n_clicks"),
    State("vs-next-modal", "is_open"),
    prevent_initial_call=True,
)
def _close_vs_modal(n, is_open):
    if not n:
        raise dash.exceptions.PreventUpdate
    return False


@app.callback(
    Output("vs-pivot", "data", allow_duplicate=True),
    Output("vs-pivot", "columns", allow_duplicate=True),
    Output("vs-volume-split", "data", allow_duplicate=True),
    Output("vs-volume-split", "columns", allow_duplicate=True),
    Input("vs-category", "value"),
    State("vs-results-store", "data"),
    prevent_initial_call=True,
)
def _on_category_change(cat, store_json):
    if not cat or not store_json:
        raise dash.exceptions.PreventUpdate
    try:
        payload = json.loads(store_json)
        data_json = payload.get("data")
        cat_col = payload.get("cat_col", "category")
        df = pd.read_json(io.StringIO(data_json), orient="split")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={cat_col: "category"})
        piv, split = forecast_group_pivot_and_long_style(df, cat)
        return (
            piv.to_dict("records"),
            [{"name": c, "id": c} for c in piv.columns],
            split.to_dict("records"),
            [{"name": c, "id": c} for c in split.columns],
        )
    except Exception:
        return [], [], [], []


@app.callback(
    Output("sa-upload-msg", "children"),
    Output("sa-preview", "data"),
    Output("sa-preview", "columns"),
    Output("sa-raw-store", "data"),
    Input("sa-upload", "contents"),
    State("sa-upload", "filename"),
    prevent_initial_call=True,
)
def _on_sa_upload(contents, filename):
    if not contents or not filename:
        raise dash.exceptions.PreventUpdate
    df, msg = _parse_upload(contents, filename)
    preview = df.head(50)
    return msg, preview.to_dict("records"), _cols(preview), df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("sa-alert", "children"),
    Output("sa-alert", "is_open"),
    Output("sa-smooth-chart", "figure"),
    Output("sa-anomaly-table", "data"),
    Output("sa-anomaly-table", "columns"),
    Output("sa-smooth-table", "data"),
    Output("sa-smooth-table", "columns"),
    Output("sa-ratio-table", "data"),
    Output("sa-ratio-table", "columns"),
    Output("sa-seasonality-table", "data"),
    Output("sa-seasonality-table", "columns"),
    Output("sa-norm1-chart", "figure"),
    Output("sa-norm2-chart", "figure", allow_duplicate=True),
    Output("sa-results-store", "data"),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("sa-run-smoothing", "n_clicks"),
    Input("sa-run-prophet", "n_clicks"),
    State("sa-raw-store", "data"),
    State("sa-window", "value"),
    State("sa-threshold", "value"),
    State("sa-prophet-order", "value"),
    State("sa-holdout", "value"),
    State("forecast-phase-store", "data"),
    prevent_initial_call=True,
)
def _run_smoothing(n_basic, n_prophet, raw_json, window, threshold, prophet_order, holdout, phase_store):
    if not n_basic and not n_prophet:
        raise dash.exceptions.PreventUpdate
    _ = holdout  # placeholder for future train/test split logic
    if not raw_json:
        return (
            "Upload data to smooth.",
            True,
            _empty_fig("No data"),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            _empty_fig(),
            _empty_fig(),
            None,
            phase_store,
        )

    try:
        df = pd.read_json(io.StringIO(raw_json), orient="split")
    except Exception:
        return (
            "Could not read uploaded data.",
            True,
            _empty_fig("Bad input"),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            _empty_fig(),
            _empty_fig(),
            None,
            phase_store,
        )

    triggered = dash.callback_context.triggered_id
    use_prophet = triggered == "sa-run-prophet"

    try:
        res = _smoothing_core(df, window, threshold, prophet_order if use_prophet else None)
    except Exception as exc:
        return (
            f"Smoothing failed: {exc}",
            True,
            _empty_fig("Error"),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            _empty_fig(),
            _empty_fig(),
            None,
            phase_store,
        )

    smoothed_tbl = res["smoothed"].copy()
    anomalies_tbl = res["anomalies"].copy()
    for tbl in (smoothed_tbl, anomalies_tbl):
        if "ds" in tbl.columns:
            tbl["ds"] = pd.to_datetime(tbl["ds"]).dt.strftime("%Y-%m-%d")

    payload = {
        "smoothed": smoothed_tbl.to_dict("records"),
        "anomalies": anomalies_tbl.to_dict("records"),
        "ratio": res["ratio"].to_dict("records"),
        "capped": res["capped"].to_dict("records"),
        "pivot": res["pivot"].to_dict("records"),
        "source": "prophet" if use_prophet else "ewma",
    }

    phase_data = phase_store or {}
    phase_data["phase1"] = json.dumps(payload)
    phase_data["phase1_meta"] = {"source": payload["source"], "ts": pd.Timestamp.utcnow().isoformat()}

    return (
        f"Smoothing complete ({'Prophet' if use_prophet else 'EWMA'}).",
        True,
        res["fig_series"],
        anomalies_tbl.to_dict("records"),
        _cols(anomalies_tbl),
        smoothed_tbl.to_dict("records"),
        _cols(smoothed_tbl),
        res["ratio"].to_dict("records"),
        _cols(res["ratio"]),
        res["capped"].to_dict("records"),
        _cols(res["capped"]),
        res["fig_ratio1"],
        res["fig_ratio2"],
        json.dumps(payload),
        phase_data,
    )


@app.callback(
    Output("sa-norm2-chart", "figure", allow_duplicate=True),
    Output("sa-seasonality-store", "data"),
    Input("sa-seasonality-table", "data"),
    prevent_initial_call=True,
)
def _edit_seasonality(rows):
    if not rows:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col != "Year":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _ratio_fig(df, "Normalized Ratio 2 (edited)"), df.to_json(date_format="iso", orient="split")


@app.callback(
    Output("sa-download-smoothed", "data"),
    Input("sa-download-btn", "n_clicks"),
    State("sa-results-store", "data"),
    prevent_initial_call=True,
)
def _download_smoothing(n, data_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return dash.no_update
    try:
        payload = json.loads(data_json)
    except Exception:
        return dash.no_update

    smoothed = pd.DataFrame(payload.get("smoothed", []))
    seasonality = pd.DataFrame(payload.get("capped", []))

    def _bundle():
        buf = io.StringIO()
        buf.write("### Smoothed Series\n")
        smoothed.to_csv(buf, index=False)
        buf.write("\n\n### Seasonality (capped)\n")
        seasonality.to_csv(buf, index=False)
        return buf.getvalue()

    return dcc.send_string(_bundle, "smoothing_results.txt")


@app.callback(
    Output("sa-save-status", "children"),
    Input("sa-save-btn", "n_clicks"),
    State("sa-results-store", "data"),
    prevent_initial_call=True,
)
def _save_smoothing(n, data_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return "Run smoothing first."
    try:
        payload = json.loads(data_json)
    except Exception as exc:
        return f"Could not parse cached data: {exc}"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    smoothed = pd.DataFrame(payload.get("smoothed", []))
    seasonality = pd.DataFrame(payload.get("capped", []))
    smoothed_path = outdir / "smoothing_smoothed.csv"
    seasonality_path = outdir / "smoothing_seasonality.csv"
    smoothed.to_csv(smoothed_path, index=False)
    seasonality.to_csv(seasonality_path, index=False)
    return f"Saved to {smoothed_path} and {seasonality_path}."


@app.callback(
    Output("sa-phase-status", "children"),
    Output("forecast-phase-store", "data", allow_duplicate=True),
    Input("sa-send-phase2", "n_clicks"),
    State("sa-results-store", "data"),
    State("forecast-phase-store", "data"),
    prevent_initial_call=True,
)
def _send_phase2(n, data_json, phase_store):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not data_json:
        return "Nothing to stage yet.", phase_store
    payload = phase_store or {}
    payload["phase1"] = data_json
    payload["phase1_meta"] = {"ts": pd.Timestamp.utcnow().isoformat(), "source": "manual-stage"}
    return "Phase 1 staged for Phase 2 forecasting.", payload


def _config_fields():
    return {
        "prophet": {
            "changepoint_prior_scale": "fc-prophet-cps",
            "seasonality_prior_scale": "fc-prophet-sps",
            "holidays_prior_scale": "fc-prophet-hps",
            "monthly_fourier_order": "fc-prophet-fourier",
            "use_holidays": "fc-prophet-holidays",
            "use_iq_value_scaled": "fc-prophet-iq",
        },
        "random_forest": {
            "n_estimators": "fc-rf-n",
            "max_depth": "fc-rf-depth",
            "use_holidays": "fc-rf-holidays",
            "use_iq_value_scaled": "fc-rf-iq",
        },
        "xgboost": {
            "n_estimators": "fc-xgb-n",
            "learning_rate": "fc-xgb-lr",
            "max_depth": "fc-xgb-depth",
            "use_holidays": "fc-xgb-holidays",
            "use_iq_value_scaled": "fc-xgb-iq",
        },
        "var": {
            "lags": "fc-var-lags",
            "use_holidays": "fc-var-holidays",
            "use_iq_value_scaled": "fc-var-iq",
        },
        "sarimax": {
            "order": "fc-sarimax-order",
            "seasonal_order": "fc-sarimax-seasonal",
            "use_holidays": "fc-sarimax-holidays",
            "use_iq_value_scaled": "fc-sarimax-iq",
        },
        "general": {
            "use_seasonality": "fc-general-seasonality",
        },
    }


@app.callback(
    Output("fc-config-store", "data"),
    Output("fc-config-status", "children"),
    Output("fc-config-status", "is_open"),
    Input("fc-config-loader", "n_intervals"),
    Input("fc-config-save", "n_clicks"),
    Input("fc-config-reset", "n_clicks"),
    State("fc-prophet-cps", "value"),
    State("fc-prophet-sps", "value"),
    State("fc-prophet-hps", "value"),
    State("fc-prophet-fourier", "value"),
    State("fc-prophet-holidays", "value"),
    State("fc-prophet-iq", "value"),
    State("fc-rf-n", "value"),
    State("fc-rf-depth", "value"),
    State("fc-rf-holidays", "value"),
    State("fc-rf-iq", "value"),
    State("fc-xgb-n", "value"),
    State("fc-xgb-lr", "value"),
    State("fc-xgb-depth", "value"),
    State("fc-xgb-holidays", "value"),
    State("fc-xgb-iq", "value"),
    State("fc-var-lags", "value"),
    State("fc-var-holidays", "value"),
    State("fc-var-iq", "value"),
    State("fc-sarimax-order", "value"),
    State("fc-sarimax-seasonal", "value"),
    State("fc-sarimax-holidays", "value"),
    State("fc-sarimax-iq", "value"),
    State("fc-general-seasonality", "value"),
    prevent_initial_call=False,
)
def _config_save_load(
    loader,
    save_click,
    reset_click,
    p_cps,
    p_sps,
    p_hps,
    p_fourier,
    p_holidays,
    p_iq,
    rf_n,
    rf_depth,
    rf_holidays,
    rf_iq,
    xgb_n,
    xgb_lr,
    xgb_depth,
    xgb_holidays,
    xgb_iq,
    var_lags,
    var_holidays,
    var_iq,
    sar_order,
    sar_seasonal,
    sar_holidays,
    sar_iq,
    gen_season,
):
    triggered = dash.callback_context.triggered_id
    cfg = config_manager.load_config()
    status = ""
    if triggered == "fc-config-reset":
        cfg = config_manager.reset_to_default()
        status = "Config reset to defaults."
    elif triggered == "fc-config-save":
        try:
            cfg["prophet"].update(
                {
                    "changepoint_prior_scale": float(p_cps or cfg["prophet"]["changepoint_prior_scale"]),
                    "seasonality_prior_scale": float(p_sps or cfg["prophet"]["seasonality_prior_scale"]),
                    "holidays_prior_scale": float(p_hps or cfg["prophet"]["holidays_prior_scale"]),
                    "monthly_fourier_order": int(p_fourier or cfg["prophet"]["monthly_fourier_order"]),
                    "use_holidays": bool(p_holidays),
                    "use_iq_value_scaled": bool(p_iq),
                }
            )
            cfg["random_forest"].update(
                {
                    "n_estimators": int(rf_n or cfg["random_forest"]["n_estimators"]),
                    "max_depth": int(rf_depth or cfg["random_forest"]["max_depth"]),
                    "use_holidays": bool(rf_holidays),
                    "use_iq_value_scaled": bool(rf_iq),
                }
            )
            cfg["xgboost"].update(
                {
                    "n_estimators": int(xgb_n or cfg["xgboost"]["n_estimators"]),
                    "learning_rate": float(xgb_lr or cfg["xgboost"]["learning_rate"]),
                    "max_depth": int(xgb_depth or cfg["xgboost"]["max_depth"]),
                    "use_holidays": bool(xgb_holidays),
                    "use_iq_value_scaled": bool(xgb_iq),
                }
            )
            cfg["var"].update(
                {
                    "lags": int(var_lags or cfg["var"]["lags"]),
                    "use_holidays": bool(var_holidays),
                    "use_iq_value_scaled": bool(var_iq),
                }
            )
            cfg["sarimax"].update(
                {
                    "order": json.loads(sar_order) if sar_order else cfg["sarimax"]["order"],
                    "seasonal_order": json.loads(sar_seasonal) if sar_seasonal else cfg["sarimax"]["seasonal_order"],
                    "use_holidays": bool(sar_holidays),
                    "use_iq_value_scaled": bool(sar_iq),
                }
            )
            cfg["general"].update({"use_seasonality": bool(gen_season)})
            config_manager.save_config(cfg)
            status = "Config saved."
        except Exception as exc:
            status = f"Save failed: {exc}"
    return cfg, status, bool(status)


@app.callback(
    Output("fc-prophet-cps", "value"),
    Output("fc-prophet-sps", "value"),
    Output("fc-prophet-hps", "value"),
    Output("fc-prophet-fourier", "value"),
    Output("fc-prophet-holidays", "value"),
    Output("fc-prophet-iq", "value"),
    Output("fc-rf-n", "value"),
    Output("fc-rf-depth", "value"),
    Output("fc-rf-holidays", "value"),
    Output("fc-rf-iq", "value"),
    Output("fc-xgb-n", "value"),
    Output("fc-xgb-lr", "value"),
    Output("fc-xgb-depth", "value"),
    Output("fc-xgb-holidays", "value"),
    Output("fc-xgb-iq", "value"),
    Output("fc-var-lags", "value"),
    Output("fc-var-holidays", "value"),
    Output("fc-var-iq", "value"),
    Output("fc-sarimax-order", "value"),
    Output("fc-sarimax-seasonal", "value"),
    Output("fc-sarimax-holidays", "value"),
    Output("fc-sarimax-iq", "value"),
    Output("fc-general-seasonality", "value"),
    Input("fc-config-store", "data"),
    prevent_initial_call=False,
)
def _populate_config(cfg):
    cfg = cfg or config_manager.load_config()
    p = cfg.get("prophet", {})
    rf = cfg.get("random_forest", {})
    xgb = cfg.get("xgboost", {})
    var_cfg = cfg.get("var", {})
    sar = cfg.get("sarimax", {})
    gen = cfg.get("general", {})
    return (
        p.get("changepoint_prior_scale"),
        p.get("seasonality_prior_scale"),
        p.get("holidays_prior_scale"),
        p.get("monthly_fourier_order"),
        p.get("use_holidays"),
        p.get("use_iq_value_scaled"),
        rf.get("n_estimators"),
        rf.get("max_depth"),
        rf.get("use_holidays"),
        rf.get("use_iq_value_scaled"),
        xgb.get("n_estimators"),
        xgb.get("learning_rate"),
        xgb.get("max_depth"),
        xgb.get("use_holidays"),
        xgb.get("use_iq_value_scaled"),
        var_cfg.get("lags"),
        var_cfg.get("use_holidays"),
        var_cfg.get("use_iq_value_scaled"),
        json.dumps(sar.get("order")),
        json.dumps(sar.get("seasonal_order")),
        sar.get("use_holidays"),
        sar.get("use_iq_value_scaled"),
        gen.get("use_seasonality"),
    )


@app.callback(
    Output("fc-alert", "children"),
    Output("fc-alert", "is_open"),
    Output("fc-combined", "data"),
    Output("fc-combined", "columns"),
    Output("fc-pivot", "data"),
    Output("fc-pivot", "columns"),
    Output("fc-errors", "children"),
    Output("fc-data-store", "data"),
    Output("fc-line-chart", "figure"),
    Output("fc-ratio1-chart", "figure"),
    Output("fc-ratio2-chart", "figure"),
    Output("fc-accuracy-table", "data"),
    Output("fc-accuracy-table", "columns"),
    Output("fc-accuracy-chart", "figure"),
    Output("fc-phase2-meta", "children"),
    Input("fc-run-btn", "n_clicks"),
    Input("fc-load-phase1", "n_clicks"),
    State("fc-upload", "contents"),
    State("fc-upload", "filename"),
    State("fc-months", "value"),
    State("forecast-phase-store", "data"),
    prevent_initial_call=True,
)
def _run_forecast(n, n_phase, contents, filename, months, phase_store):
    if not n and not n_phase:
        raise dash.exceptions.PreventUpdate
    try:
        months = int(months or 12)
    except Exception:
        months = 12

    ctx = dash.callback_context
    triggered = ctx.triggered_id if ctx.triggered_id else "fc-run-btn"
    use_phase_data = triggered == "fc-load-phase1"
    df = None
    source = ""

    if contents and filename and not use_phase_data:
        try:
            df, msg = _parse_upload(contents, filename)
            source = f"upload:{filename}"
        except Exception as exc:
            return (
                f"Upload failed: {exc}",
                True,
                [],
                [],
                [],
                [],
                "",
                None,
                _empty_fig(),
                _empty_fig(),
                _empty_fig(),
                [],
                [],
                _empty_fig(),
                "",
            )
    if (df is None or df.empty) and phase_store:
        try:
            phase_payload = json.loads(phase_store) if isinstance(phase_store, str) else phase_store
            phase1_json = phase_payload.get("phase1")
            if phase1_json:
                phase1_payload = json.loads(phase1_json) if isinstance(phase1_json, str) else phase1_json
                smoothed_records = phase1_payload.get("smoothed", [])
                df = pd.DataFrame(smoothed_records)
                if "Final_Smoothed_Value" not in df.columns and "smoothed" in df.columns:
                    df["Final_Smoothed_Value"] = df["smoothed"]
                source = "phase1-staged"
        except Exception:
            df = None

    if df is None or df.empty:
        return (
            "Upload smoothed data (or stage Phase 1) to run the forecast.",
            True,
            [],
            [],
            [],
            [],
            "",
            None,
            _empty_fig("No data"),
            _empty_fig(),
            _empty_fig(),
            [],
            [],
            _empty_fig(),
            "",
        )

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if "Final_Smoothed_Value" not in df.columns and "y" in df.columns:
        df["Final_Smoothed_Value"] = df["y"]

    try:
        res = run_phase2_forecast(df, months)
        forecast_results = res.get("forecast_results", {})
        combined, wide, pivot_smoothed = process_forecast_results(forecast_results)
        errors = res.get("errors", {})
        err_txt = "; ".join([f"{k}: {v}" for k, v in errors.items()]) if errors else ""
        line_fig = _empty_fig("No forecast data")
        if not combined.empty:
            line_fig = px.line(
                combined,
                x="Month",
                y="Forecast",
                color="Model",
                markers=True,
                title="Forecast by model",
            )
            if not pivot_smoothed.empty:
                baseline_long = pivot_smoothed.melt(id_vars="Year", var_name="Month", value_name="Final_Smoothed_Value")
                baseline_long["Month"] = pd.to_datetime(
                    "1 " + baseline_long["Month"].astype(str) + " " + baseline_long["Year"].astype(str),
                    errors="coerce",
                )
                line_fig.add_scatter(
                    x=baseline_long["Month"],
                    y=baseline_long["Final_Smoothed_Value"],
                    mode="lines+markers",
                    name="Final Smoothed",
                    line=dict(color="gray", dash="dot"),
                )

        ratio_fig1, ratio_fig2 = _empty_fig("No ratio data"), _empty_fig("No ratio data")
        ratio_df, capped_df = pd.DataFrame(), pd.DataFrame()
        try:
            _raw_fig, capped_df, ratio_df = plot_contact_ratio_seasonality(pivot_smoothed)
            ratio_fig1 = _ratio_fig(ratio_df, "Normalized Ratio 1")
            ratio_fig2 = _ratio_fig(capped_df, "Normalized Ratio 2 (capped)")
        except Exception:
            pass

        accuracy_tbl = pd.DataFrame()
        accuracy_fig = _empty_fig("Accuracy not available")
        try:
            wide_with_base = fill_final_smoothed_row(wide.copy(), pivot_smoothed)
            accuracy_tbl = accuracy_phase1(wide_with_base, pivot_smoothed)
            if not accuracy_tbl.empty:
                plot_df = accuracy_tbl.copy()
                for col in ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"]:
                    if col in plot_df.columns:
                        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
                accuracy_fig = px.bar(
                    plot_df,
                    x="Model",
                    y=[c for c in ["Accuracy(+−5%)", "Accuracy(+−7%)", "Accuracy(+−10%)"] if c in plot_df.columns],
                    barmode="group",
                    title="Accuracy by model (%)",
                )
        except Exception:
            pass

        store = res | {
            "combined": combined.to_dict("records"),
            "wide": wide.to_dict("records"),
            "pivot_smoothed": pivot_smoothed.to_dict("records"),
            "accuracy": accuracy_tbl.to_dict("records") if isinstance(accuracy_tbl, pd.DataFrame) else [],
            "ratio": ratio_df.to_dict("records") if isinstance(ratio_df, pd.DataFrame) else [],
            "capped": capped_df.to_dict("records") if isinstance(capped_df, pd.DataFrame) else [],
        }
        model_names = [m for m in forecast_results.keys() if m not in ("final_smoothed_values",)]
        meta = f"Phase 2 run ({source or 'upload'}) | horizon {months} months | models: {', '.join(model_names)}"
        return (
            "Forecast complete.",
            True,
            combined.to_dict("records"),
            [{"name": c, "id": c} for c in combined.columns],
            wide.to_dict("records"),
            [{"name": c, "id": c} for c in wide.columns],
            err_txt,
            json.dumps(store),
            line_fig,
            ratio_fig1,
            ratio_fig2,
            accuracy_tbl.to_dict("records") if isinstance(accuracy_tbl, pd.DataFrame) else [],
            _cols(accuracy_tbl) if isinstance(accuracy_tbl, pd.DataFrame) else [],
            accuracy_fig,
            meta,
        )
    except Exception as exc:
        return (
            f"Forecast failed: {exc}",
            True,
            [],
            [],
            [],
            [],
            str(exc),
            None,
            _empty_fig("Error"),
            _empty_fig(),
            _empty_fig(),
            [],
            [],
            _empty_fig(),
            "",
        )


@app.callback(
    Output("fc-download-forecast", "data"),
    Input("fc-download-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _download_forecast_results(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return dash.no_update
    try:
        payload = json.loads(store_json)
    except Exception:
        return dash.no_update
    combined = pd.DataFrame(payload.get("combined", []))
    if combined.empty:
        return dash.no_update
    return dcc.send_data_frame(combined.to_csv, "forecast_results.csv", index=False)


@app.callback(
    Output("fc-download-config", "data"),
    Input("fc-download-config-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _download_forecast_config(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return dash.no_update
    try:
        payload = json.loads(store_json)
    except Exception:
        return dash.no_update
    cfg = payload.get("config", {})
    combined = pd.DataFrame(payload.get("combined", []))
    txt = create_download_csv_with_metadata(combined, cfg)
    return dcc.send_string(lambda: txt, "forecast_with_config.txt")


@app.callback(
    Output("fc-save-status", "children"),
    Input("fc-save-btn", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _save_forecast_to_disk(n, store_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not store_json:
        return "Run a forecast first."
    try:
        payload = json.loads(store_json)
    except Exception as exc:
        return f"Could not read cached forecast: {exc}"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    combined = pd.DataFrame(payload.get("combined", []))
    accuracy = pd.DataFrame(payload.get("accuracy", []))
    combined_path = outdir / "forecast_results.csv"
    accuracy_path = outdir / "forecast_accuracy.csv"
    combined.to_csv(combined_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)
    return f"Saved to {combined_path} and {accuracy_path}."


# ---------------------------------------------------------------------------
# Transformation / Projects (Phase 2+ adjustments)
# ---------------------------------------------------------------------------


def _load_latest_forecast_file() -> tuple[pd.DataFrame, str]:
    txt_path = Path(__file__).resolve().parent.parent / "latest_forecast_full_path.txt"
    if txt_path.exists():
        try:
            file_path = txt_path.read_text().strip()
            if file_path and Path(file_path).exists():
                df = pd.read_csv(file_path)
                return df, f"Loaded {Path(file_path).name}."
            return pd.DataFrame(), f"File not found: {file_path}"
        except Exception as exc:
            return pd.DataFrame(), f"Could not load latest forecast: {exc}"
    return pd.DataFrame(), "No latest forecast path found."


def _options_from_df(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    vals = sorted(pd.unique(df[col].dropna()).tolist())
    return [{"label": str(v), "value": v} for v in vals]


@app.callback(
    Output("tp-load-status", "children"),
    Output("tp-raw-store", "data"),
    Output("tp-group", "options"),
    Output("tp-model", "options"),
    Output("tp-year", "options"),
    Output("tp-raw-table", "data"),
    Output("tp-raw-table", "columns"),
    Input("tp-load-latest", "n_clicks"),
    Input("tp-use-phase", "n_clicks"),
    State("fc-data-store", "data"),
    prevent_initial_call=True,
)
def _tp_load_source(n_file, n_phase, fc_store_json):
    if not n_file and not n_phase:
        raise dash.exceptions.PreventUpdate
    ctx = dash.callback_context
    trig = ctx.triggered_id if ctx.triggered_id else "tp-load-latest"

    df = pd.DataFrame()
    msg = ""
    if trig == "tp-load-latest":
        df, msg = _load_latest_forecast_file()
    elif trig == "tp-use-phase":
        if fc_store_json:
            try:
                payload = json.loads(fc_store_json)
                combined = pd.DataFrame(payload.get("combined", []))
                pivot = pd.DataFrame(payload.get("pivot_smoothed", []))
                if not combined.empty:
                    # attempt to reconstruct Month_Year and Year/Month
                    combined["Month_Year"] = pd.to_datetime(combined["Month"]).dt.strftime("%b-%y")
                    combined["Year"] = pd.to_datetime(combined["Month"]).dt.year
                    combined["Month"] = pd.to_datetime(combined["Month"]).dt.strftime("%b")
                    combined.rename(columns={"Forecast": "Base_Forecast_for_Forecast_Group"}, inplace=True)
                    df = combined
                    msg = "Loaded from Phase 2 results."
                elif not pivot.empty:
                    pivot = pivot.copy()
                    df = pivot
                    msg = "Loaded pivot smoothed from Phase 2."
                else:
                    msg = "No Phase 2 data available."
            except Exception as exc:
                msg = f"Could not parse Phase 2 data: {exc}"
        else:
            msg = "Phase 2 results not found."

    opts_group = _options_from_df(df, "forecast_group")
    opts_model = _options_from_df(df, "Model")
    opts_year = _options_from_df(df, "Year")
    preview = df.head(200)
    return msg, df.to_json(date_format="iso", orient="split"), opts_group, opts_model, opts_year, preview.to_dict("records"), _cols(preview)


@app.callback(
    Output("tp-selection-status", "children"),
    Output("tp-selection-status", "is_open"),
    Output("tp-filtered-table", "data"),
    Output("tp-filtered-table", "columns"),
    Output("tp-transform-table", "data"),
    Output("tp-transform-table", "columns"),
    Output("tp-filtered-store", "data"),
    Input("tp-apply-selection", "n_clicks"),
    State("tp-raw-store", "data"),
    State("tp-group", "value"),
    State("tp-model", "value"),
    State("tp-year", "value"),
    prevent_initial_call=True,
)
def _tp_apply_selection(n, raw_json, group, model, year):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not raw_json:
        return "Load data first.", True, [], [], [], [], None
    try:
        df = pd.read_json(io.StringIO(raw_json), orient="split")
    except Exception:
        return "Could not read loaded data.", True, [], [], [], [], None

    filtered = df.copy()
    if group and "forecast_group" in filtered.columns:
        filtered = filtered[filtered["forecast_group"] == group]
    if model and "Model" in filtered.columns:
        filtered = filtered[filtered["Model"] == model]
    if year and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"] >= year]
    if filtered.empty:
        return "No data for that selection.", True, [], [], [], [], None

    # ensure Month_Year present
    if "Month_Year" not in filtered.columns and "Month" in filtered.columns and "Year" in filtered.columns:
        filtered["Month_Year"] = filtered.apply(
            lambda r: f"{str(r['Month'])[:3]}-{str(r['Year'])[-2:]}", axis=1
        )

    filtered = filtered.sort_values(["Year", "Month_Year"]) if "Year" in filtered.columns else filtered

    # add transformation columns
    display_cols = ["Month_Year", "Year", "Month", "Base_Forecast_for_Forecast_Group"]
    base_cols = [c for c in display_cols if c in filtered.columns]
    transform_df = filtered[base_cols].copy()
    for col in _TRANSFORMATION_COLS:
        transform_df[col] = ""

    return (
        f"Loaded {len(filtered)} rows for {group or 'All'} | {model or 'All'} | {year or 'All'}+",
        True,
        filtered.head(200).to_dict("records"),
        _cols(filtered.head(200)),
        transform_df.to_dict("records"),
        _cols(transform_df),
        filtered.to_json(date_format="iso", orient="split"),
    )


@app.callback(
    Output("tp-transform-status", "children"),
    Output("tp-final-store", "data"),
    Output("tp-transposed-table", "data"),
    Output("tp-transposed-table", "columns"),
    Output("tp-final-forecast-table", "data"),
    Output("tp-final-forecast-table", "columns"),
    Output("tp-summary-json", "children"),
    Input("tp-apply-transform", "n_clicks"),
    State("tp-transform-table", "data"),
    State("tp-filtered-store", "data"),
    prevent_initial_call=True,
)
def _tp_apply_transform(n, edited_rows, filtered_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not edited_rows or not filtered_json:
        return "Nothing to transform yet.", None, [], [], [], [], ""
    try:
        base_filtered = pd.read_json(io.StringIO(filtered_json), orient="split")
    except Exception:
        base_filtered = pd.DataFrame()
    edited_df = pd.DataFrame(edited_rows)
    full_df = base_filtered.copy()
    for col in edited_df.columns:
        if col in full_df.columns:
            # keep base cols as-is
            continue
        full_df[col] = edited_df[col]

    processed = _apply_transformations(full_df)

    # transposed view
    transpose_cols = [
        "Month_Year",
        "Base_Forecast_for_Forecast_Group",
        "Transformation 1",
        "Remarks_Tr 1",
        "Forecast_Transformation 1",
        "Transformation 2",
        "Remarks_Tr 2",
        "Forecast_Transformation 2",
        "Transformation 3",
        "Remarks_Tr 3",
        "Forecast_Transformation 3",
        "IA 1",
        "Remarks_IA 1",
        "Forecast_IA 1",
        "IA 2",
        "Remarks_IA 2",
        "Forecast_IA 2",
        "IA 3",
        "Remarks_IA 3",
        "Forecast_IA 3",
        "Marketing Campaign 1",
        "Remarks_Mkt 1",
        "Forecast_Marketing Campaign 1",
        "Marketing Campaign 2",
        "Remarks_Mkt 2",
        "Forecast_Marketing Campaign 2",
        "Marketing Campaign 3",
        "Remarks_Mkt 3",
        "Forecast_Marketing Campaign 3",
    ]
    available_cols = [c for c in transpose_cols if c in processed.columns]
    transposed = pd.DataFrame()
    if available_cols:
        transposed = processed[available_cols].copy()
        if "Month_Year" in transposed.columns:
            t = transposed.set_index("Month_Year").transpose().reset_index()
            t.rename(columns={"index": "Category"}, inplace=True)
            if "Forecast_Marketing Campaign 3" in processed.columns:
                final_forecast_values = processed.set_index("Month_Year")["Forecast_Marketing Campaign 3"]
                t.loc[len(t)] = ["Final Forecast"] + final_forecast_values.tolist()
            transposed = t

    # final forecast table
    final_cols = ["Month_Year", "Forecast_Marketing Campaign 3"]
    if "forecast_group" in processed.columns:
        final_cols.insert(0, "forecast_group")
    if "Model" in processed.columns:
        final_cols.insert(-1, "Model")
    if "Year" in processed.columns:
        final_cols.insert(-1, "Year")
    final_cols = [c for c in final_cols if c in processed.columns]
    final_tbl = processed[final_cols].copy()
    if "Forecast_Marketing Campaign 3" in final_tbl.columns:
        final_tbl = final_tbl.rename(columns={"Forecast_Marketing Campaign 3": "Final_Forecast"})

    # summary
    summary = {}
    try:
        summary = {
            "Forecast Group": processed["forecast_group"].iloc[0] if "forecast_group" in processed.columns else None,
            "Model": processed["Model"].iloc[0] if "Model" in processed.columns else None,
            "Selected Year": processed["Year"].min() if "Year" in processed.columns else None,
            "Years Included": sorted(processed["Year"].unique().tolist()) if "Year" in processed.columns else [],
            "Total Rows": len(processed),
            "Base Forecast Total": float(processed["Base_Forecast_for_Forecast_Group"].sum()) if "Base_Forecast_for_Forecast_Group" in processed.columns else None,
            "Final Forecast Total": float(processed["Final_Forecast"].sum()) if "Final_Forecast" in processed.columns else None,
        }
    except Exception:
        summary = {}

    return (
        "Transformations applied.",
        processed.to_json(date_format="iso", orient="split"),
        transposed.to_dict("records"),
        _cols(transposed),
        final_tbl.to_dict("records"),
        _cols(final_tbl),
        json.dumps(summary, indent=2),
    )


@app.callback(
    Output("tp-download-final", "data"),
    Input("tp-download-final-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_download_final(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return dash.no_update
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception:
        return dash.no_update
    if "Final_Forecast" in df.columns:
        use_df = df
    else:
        use_df = df.rename(columns={"Forecast_Marketing Campaign 3": "Final_Forecast"}) if "Forecast_Marketing Campaign 3" in df.columns else df
    return dcc.send_data_frame(use_df.to_csv, "final_forecast.csv", index=False)


@app.callback(
    Output("tp-download-full", "data"),
    Input("tp-download-full-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_download_full(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return dash.no_update
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception:
        return dash.no_update
    return dcc.send_data_frame(df.to_csv, "transformation_full.csv", index=False)


@app.callback(
    Output("tp-save-status", "children", allow_duplicate=True),
    Input("tp-save-btn", "n_clicks"),
    State("tp-final-store", "data"),
    prevent_initial_call=True,
)
def _tp_save_to_disk(n, final_json):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not final_json:
        return "Run transformations first."
    try:
        df = pd.read_json(io.StringIO(final_json), orient="split")
    except Exception as exc:
        return f"Could not read results: {exc}"

    base_dir_txt = Path(__file__).resolve().parent.parent / "latest_forecast_base_dir.txt"
    outdir = Path(__file__).resolve().parent.parent / "exports"
    outdir.mkdir(exist_ok=True)
    target_dir = outdir
    if base_dir_txt.exists():
        try:
            bd = base_dir_txt.read_text().strip()
            if bd:
                target_dir = Path(bd)
                target_dir.mkdir(exist_ok=True, parents=True)
        except Exception:
            pass
    timestamp = pd.Timestamp.utcnow()
    fname = f"Monthly_Forecast_with_Adjustments_{timestamp:%d_%b_%Y}_{timestamp:%H_%M_%S}_user.csv"
    fpath = target_dir / fname
    try:
        df.to_csv(fpath, index=False)
        return f"Saved to {fpath}"
    except Exception as exc:
        return f"Save failed: {exc}"


@app.callback(
    Output("tp-raw-store", "data", allow_duplicate=True),
    Output("tp-filtered-store", "data", allow_duplicate=True),
    Output("tp-final-store", "data", allow_duplicate=True),
    Output("tp-raw-table", "data", allow_duplicate=True),
    Output("tp-raw-table", "columns", allow_duplicate=True),
    Output("tp-filtered-table", "data", allow_duplicate=True),
    Output("tp-filtered-table", "columns", allow_duplicate=True),
    Output("tp-transform-table", "data", allow_duplicate=True),
    Output("tp-transform-table", "columns", allow_duplicate=True),
    Output("tp-transposed-table", "data", allow_duplicate=True),
    Output("tp-transposed-table", "columns", allow_duplicate=True),
    Output("tp-final-forecast-table", "data", allow_duplicate=True),
    Output("tp-final-forecast-table", "columns", allow_duplicate=True),
    Output("tp-selection-status", "is_open", allow_duplicate=True),
    Output("tp-selection-status", "children", allow_duplicate=True),
    Output("tp-transform-status", "children", allow_duplicate=True),
    Input("tp-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _tp_reset(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    empty_cols = []
    return (None, None, None, [], empty_cols, [], empty_cols, [], empty_cols, [], empty_cols, [], empty_cols, False, "", "")
