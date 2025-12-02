from __future__ import annotations
import base64
import io
import json
from typing import Optional, Tuple

import dash
from dash import Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
from app_instance import app
from forecasting.process_and_IQ_data import forecast_group_pivot_and_long_style


def _parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, str]:
    if not contents or "," not in contents:
        return pd.DataFrame(), "No file supplied."
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            df = pd.read_excel(io.BytesIO(decoded))
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
    Output("vs-pivot", "data"),
    Output("vs-pivot", "columns"),
    Output("vs-volume-split", "data"),
    Output("vs-volume-split", "columns"),
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
