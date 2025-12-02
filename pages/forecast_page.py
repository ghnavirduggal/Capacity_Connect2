from __future__ import annotations
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from common import header_bar

MODELS = [
    {
        "title": "Random Forest",
        "icon": "🌳",
        "content": """
Aggregates many decision trees for prediction.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
f(x) = (1 / B ) Σ Tₖ(x)
</span>
"""
    },
    {
        "title": "Prophet",
        "icon": "📅",
        "content": """
Handles trend, seasonality, and holidays.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
yₜ = gₜ + sₜ + hₜ + eₜ
</span>
"""
    },
    {
        "title": "XGBoost",
        "icon": "⚡",
        "content": """
Gradient boosting framework for high performance.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷᵢ = Σ fₖ(xᵢ) <br>
Obj(θ) = Σ l(yᵢ, ŷᵢ) + Σ Ω(fₖ)
</span>
"""
    },
    {
        "title": "ARIMA",
        "icon": "📘",
        "content": """
Combines autoregression and moving average.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
AR(p): yₜ = c + Φ₁yₜ₋₁ + ... + Φₚyₜ₋ₚ <br>
MA(q): yₜ = α + θ₁eₜ₋₁ + ... + θq eₜ₋q <br>
ARIMA(p,d,q): differencing d times then ARMA(p,q)
</span>
"""
    },
    {
        "title": "Triple Exponential Smoothing (Holt-Winters)",
        "icon": "📉",
        "content": """
Captures level, trend, and seasonality.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
Level: lₜ = βyₜ + (1−β)(lₜ₋₁ + bₜ₋₁)<br>
Trend: bₜ = β(lₜ − lₜ₋₁) + (1 − β)bₜ₋₁<br>
Seasonality: sₜ = γ(yₜ / lₜ) + (1 − γ)sₜ₋ₘ<br>
Forecast: yₜ₊₁ = lₜ + bₜ + sₜ₊₁
</span>
"""
    },
    {
        "title": "Double Exponential Smoothing (Holt’s)",
        "icon": "📊",
        "content": """
Captures level and trend.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
Level: lₜ = βyₜ + (1−β)(lₜ₋₁)<br>
Trend: bₜ = β(lₜ − lₜ₋₁) + (1 − β)bₜ₋₁<br>
Forecast: yₜ₊₁ = lₜ + bₜ
</span>
"""
    },
    {
        "title": "Single Exponential Smoothing",
        "icon": "🔹",
        "content": """
Simple smoothing method.<br><br>
<strong style="color:#000000;">Equations:</strong><br>
<span style="color:#000000; font-size:22px;">
lₜ = βyₜ + (1 − β)lₜ₋₁<br>
Forecast: yₜ₊₁ = lₜ
</span>
"""
    },
    {
        "title": "Linear Regression",
        "icon": "📐",
        "content": """
Predicts using a linear combination of features.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷ = β0 + β1x₁ + β2x₂ + ... + βₖxₖ<br>
RSS = Σ (yᵢ − ŷᵢ)²
</span>
"""
    },
    {
        "title": "Weighted Moving Average",
        "icon": "📘",
        "content": """
Forecasts using average of past observations.<br><br>
<strong style="color:#000000;">Equation:</strong><br>
<span style="color:#000000; font-size:22px;">
ŷₜ = Σ (wᵢ × yₜ₋ᵢ), where Σ wᵢ = 1
</span>
"""
    },
]

FORECAST_NAV = [
    {"slug": "volume-summary", "label": "Volume Summary", "emoji": "📊"},
    {"slug": "smoothing-anomaly", "label": "Smoothing & Anomaly Detection", "emoji": "🧹"},
    {"slug": "forecasting", "label": "Forecasting", "emoji": "🔮"},
    {"slug": "transformation-projects", "label": "Transformation Projects", "emoji": "⚙️"},
    {"slug": "daily-interval", "label": "Daily Interval Forecast", "emoji": "⏱️"},
]

FORECAST_STEPS = [
    {"slug": "volume-summary", "label": "Volume Summary", "emoji": "📊", "next": "smoothing-anomaly"},
    {"slug": "smoothing-anomaly", "label": "Smoothing & Anomaly Detection", "emoji": "🧹", "next": "forecasting"},
    {"slug": "forecasting", "label": "Forecasting", "emoji": "🔮", "next": "transformation-projects"},
    {"slug": "transformation-projects", "label": "Transformation Projects", "emoji": "⚙️", "next": "daily-interval"},
    {"slug": "daily-interval", "label": "Daily Interval Forecast", "emoji": "⏱️", "next": None},
]

def _model_card(title: str, content: str, icon: str):
    return html.Div(
        className="forecast-card",
        children=[
            html.Div(f"{icon} {title}", className="forecast-card-title"),
            dcc.Markdown(
                content,
                dangerously_allow_html=True,
                className="forecast-card-content"
            ),
        ],
    )

def _nav_buttons():
    return dbc.Row(
        [
            dbc.Col(
                dcc.Link(
                    dbc.Button(
                        f"{item['emoji']} {item['label']}",
                        color="secondary",
                        outline=True,
                        className="w-100",
                    ),
                    href=f"/forecast/{item['slug']}",
                    style={"textDecoration": "none"},
                ),
                xs=12,
                sm=6,
                md=4,
                lg=3,
            )
            for item in FORECAST_NAV
        ],
        className="g-2 mb-3",
    )

def page_forecast():
    return html.Div(
        dbc.Container(
            [
                header_bar(),
                html.Div(
                    children=[
                        html.H1(
                            "🔮 Power of 9 Models: A complete suite for Forecasting",
                            className="forecast-heading",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Your guided path through forecasting.",
                                    className="text-muted small",
                                ),
                                dcc.Link(
                                    dbc.Button("→ Volume Summary", color="primary", outline=False, className="float-end"),
                                    href="/forecast/volume-summary",
                                    style={"textDecoration": "none"},
                                ),
                            ],
                            className="d-flex justify-content-between align-items-center mb-3",
                        ),
                        html.Div(
                            children=[
                                _model_card(m["title"], m["content"], m["icon"])
                                for m in MODELS
                            ],
                            className="forecast-grid",
                        ),
                        _nav_buttons(),
                    ],
                    className="forecast-page",
                ),
            ],
            fluid=True,
        )
    )

def page_forecast_section(slug: str):
    """Forecasting workspace wizard-style sections."""
    shared_stores = [
        dcc.Store(id="vs-data-store", storage_type="session"),
        dcc.Store(id="vs-results-store", storage_type="session"),
        dcc.Store(id="vs-iq-store", storage_type="session"),
    ]

    def _stepper(active_slug: str):
        items = []
        for step in FORECAST_STEPS:
            is_active = step["slug"] == active_slug
            items.append(
                html.Div(
                    [
                        dbc.Badge(f"{step['emoji']} {step['label']}", color="primary" if is_active else "secondary", pill=True, className="me-2"),
                        html.Span("→", className="text-muted me-2") if step["next"] else None,
                    ],
                    className="d-flex align-items-center",
                )
            )
        return html.Div(items, className="d-flex flex-wrap gap-2 mb-3")

    def _section_heading(step_meta):
        prev_item = None
        for i, s in enumerate(FORECAST_STEPS):
            if s["slug"] == step_meta["slug"] and i > 0:
                prev_item = FORECAST_STEPS[i - 1]
                break
        prev_href = f"/forecast/{prev_item['slug']}" if prev_item else "/forecast"
        prev_label = f"← {prev_item['label']}" if prev_item else "Back to models"
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(f"{step_meta['emoji']} {step_meta['label']}", className="mb-1"),
                        html.Div("Follow the steps to keep moving forward.", className="text-muted"),
                    ],
                ),
                dcc.Link(
                    dbc.Button(prev_label, color="link", className="px-0"),
                    href=prev_href,
                ),
            ],
            className="d-flex justify-content-between align-items-start mb-3 flex-wrap gap-2",
        )

    def _volume_summary_layout(step_meta):
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Upload data"),
                                dcc.Upload(
                                    id="vs-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select a CSV/XLSX file")]),
                                    multiple=False,
                                    accept=".csv,.xls,.xlsx",
                                    className="border border-secondary rounded p-3 mb-2 bg-light",
                                ),
                                html.Div(id="vs-upload-msg", className="small text-muted"),
                                dbc.Button("Run Volume Summary", id="vs-run-btn", color="primary", className="mt-2"),
                                dbc.Alert(id="vs-alert", color="info", is_open=False, className="mt-2"),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.H4("Preview"),
                                dash_table.DataTable(
                                    id="vs-preview",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Summary"),
                                dash_table.DataTable(
                                    id="vs-summary",
                                    data=[],
                                    columns=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H5("Forecast group view", className="mb-1"),
                                        dcc.Dropdown(id="vs-category", options=[], value=None, placeholder="Select category"),
                                    ],
                                    className="mb-2",
                                ),
                                dash_table.DataTable(
                                    id="vs-pivot",
                                    data=[],
                                    columns=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Volume split (%)"),
                                dash_table.DataTable(
                                    id="vs-volume-split",
                                    data=[],
                                    columns=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                            ],
                            md=8,
                        ),
                    ],
                    className="g-3",
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Proceed to next step"),
                        dbc.ModalBody("Volume summary ran successfully. Move to Smoothing & Anomaly Detection?"),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Stay here", id="vs-modal-close", className="me-2"),
                                dcc.Link(dbc.Button("Go to Smoothing →", color="primary"), href="/forecast/smoothing-anomaly", style={"textDecoration": "none"}),
                            ]
                        ),
                    ],
                    id="vs-next-modal",
                    is_open=False,
                ),
            ],
            className="forecast-page",
        )

    def _smoothing_layout(step_meta):
        return [
            dcc.Store(id="sa-raw-store"),
            dcc.Store(id="sa-results-store"),
            dcc.Store(id="sa-seasonality-store"),
            dcc.Download(id="sa-download-smoothed"),
            dcc.Download(id="sa-download-seasonality"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Upload raw volume data", className="mb-2"),
                                dcc.Upload(
                                    id="sa-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select CSV/XLSX (Date, Volume, IQ_value optional)")]),
                                    multiple=False,
                                    className="border border-secondary rounded p-3 mb-2 bg-light",
                                ),
                                html.Div(id="sa-upload-msg", className="small text-muted mb-2"),
                                html.Div("Smoothing & anomaly configuration", className="fw-semibold mb-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("EWMA span"), dbc.Input(id="sa-window", type="number", value=6, min=1)]), md=6),
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Z-score threshold"), dbc.Input(id="sa-threshold", type="number", value=2.5, step=0.1)]), md=6),
                                    ],
                                    className="g-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Prophet fourier"), dbc.Input(id="sa-prophet-order", type="number", value=5, min=1)]), md=6),
                                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("Hold-out months"), dbc.Input(id="sa-holdout", type="number", value=0, min=0)]), md=6),
                                    ],
                                    className="g-2 mt-1",
                                ),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Run Smoothing", id="sa-run-smoothing", color="primary"),
                                        dbc.Button("Prophet smoothing", id="sa-run-prophet", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Download", id="sa-download-btn", color="secondary", outline=True),
                                        dbc.Button("Save to disk", id="sa-save-btn", color="secondary", outline=True),
                                        dbc.Button("Send to Phase 2", id="sa-send-phase2", color="info"),
                                    ],
                                    className="mt-2",
                                ),
                                dbc.Alert(id="sa-alert", is_open=False, color="info", className="mt-2"),
                                html.Div(id="sa-save-status", className="small text-muted mt-1"),
                                html.Div(id="sa-phase-status", className="small fw-semibold text-primary mt-2"),
                            ],
                            body=True,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Series & anomalies",
                                    tab_id="series",
                                    children=[
                                        dcc.Graph(id="sa-smooth-chart", figure={}, className="mb-2"),
                                        dash_table.DataTable(
                                            id="sa-anomaly-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontSize": 12},
                                        ),
                                        dash_table.DataTable(
                                            id="sa-smooth-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontSize": 12},
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Seasonality & ratios",
                                    tab_id="seasonality",
                                    children=[
                                        html.Div(
                                            "Normalized ratio views. Edit the capped seasonality table to tweak the curve.",
                                            className="small text-muted mb-2",
                                        ),
                                        dcc.Graph(id="sa-norm1-chart", figure={}, className="mb-2"),
                                        dcc.Graph(id="sa-norm2-chart", figure={}, className="mb-2"),
                                        dash_table.DataTable(
                                            id="sa-ratio-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontSize": 12},
                                        ),
                                        dash_table.DataTable(
                                            id="sa-seasonality-table",
                                            data=[],
                                            columns=[],
                                            page_size=6,
                                            editable=True,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontSize": 12},
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Raw preview",
                                    tab_id="raw",
                                    children=[
                                        dash_table.DataTable(
                                            id="sa-preview",
                                            data=[],
                                            columns=[],
                                            page_size=8,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"fontSize": 12},
                                        ),
                                    ],
                                ),
                            ],
                            active_tab="series",
                        ),
                        md=8,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _forecasting_layout(step_meta):
        return [
            dcc.Download(id="fc-download-forecast"),
            dcc.Download(id="fc-download-config"),
            dcc.Download(id="fc-download-accuracy"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Smoothed data upload"),
                            dcc.Upload(
                                id="fc-upload",
                                children=html.Div(["Drag & drop or ", html.Strong("select smoothed CSV/XLSX (ds, Final_Smoothed_Value, IQ_value)")]),
                                multiple=False,
                                className="border border-secondary rounded p-3 mb-2 bg-light",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Forecast months"),
                                    dbc.Input(id="fc-months", type="number", value=12, min=1, step=1),
                                ],
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Run Forecast", id="fc-run-btn", color="primary"),
                                    dbc.Button("Load Phase 1 data", id="fc-load-phase1", color="secondary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Download results", id="fc-download-btn", color="secondary", outline=True),
                                    dbc.Button("Download config", id="fc-download-config-btn", color="secondary", outline=True),
                                    dbc.Button("Save to disk", id="fc-save-btn", color="secondary", outline=True),
                                ],
                                className="mb-2",
                            ),
                            html.Div(id="fc-save-status", className="small text-muted mb-1"),
                            dbc.Alert(id="fc-alert", color="info", is_open=False),
                            html.Div(id="fc-errors", className="text-danger small mt-2"),
                            html.Div(id="fc-phase2-meta", className="small fw-semibold text-primary mt-2"),
                            html.Hr(),
                            html.H5("Model configuration"),
                            dcc.Store(id="fc-config-store"),
                            dcc.Interval(id="fc-config-loader", interval=0, n_intervals=0, max_intervals=1),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("ChangePoint Prior"), dbc.Input(id="fc-prophet-cps", type="number", step=0.01)]), md=6),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Seasonality Prior"), dbc.Input(id="fc-prophet-sps", type="number", step=0.01)]), md=6),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Holiday Prior"), dbc.Input(id="fc-prophet-hps", type="number", step=0.01)]), md=6),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Fourier Order"), dbc.Input(id="fc-prophet-fourier", type="number", step=1)]), md=6),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-prophet-holidays", label="Use holidays"), md=6),
                                                    dbc.Col(dbc.Checkbox(id="fc-prophet-iq", label="Use IQ scaled"), md=6),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="Prophet",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("n_estimators"), dbc.Input(id="fc-rf-n", type="number", step=10)]), md=6),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("max_depth"), dbc.Input(id="fc-rf-depth", type="number", step=1)]), md=6),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-rf-holidays", label="Use holidays"), md=6),
                                                    dbc.Col(dbc.Checkbox(id="fc-rf-iq", label="Use IQ scaled"), md=6),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="Random Forest",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("n_estimators"), dbc.Input(id="fc-xgb-n", type="number", step=10)]), md=4),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("learning_rate"), dbc.Input(id="fc-xgb-lr", type="number", step=0.01)]), md=4),
                                                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("max_depth"), dbc.Input(id="fc-xgb-depth", type="number", step=1)]), md=4),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-xgb-holidays", label="Use holidays"), md=6),
                                                    dbc.Col(dbc.Checkbox(id="fc-xgb-iq", label="Use IQ scaled"), md=6),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="XGBoost",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.InputGroup([dbc.InputGroupText("Lags"), dbc.Input(id="fc-var-lags", type="number", step=1)]),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-var-holidays", label="Use holidays"), md=6),
                                                    dbc.Col(dbc.Checkbox(id="fc-var-iq", label="Use IQ scaled"), md=6),
                                                ],
                                                className="g-2 mt-2",
                                            ),
                                        ],
                                        title="VAR",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.InputGroup([dbc.InputGroupText("order (p,d,q)"), dbc.Input(id="fc-sarimax-order", type="text")], className="mb-2"),
                                            dbc.InputGroup([dbc.InputGroupText("seasonal (P,D,Q,s)"), dbc.Input(id="fc-sarimax-seasonal", type="text")], className="mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(dbc.Checkbox(id="fc-sarimax-holidays", label="Use holidays"), md=6),
                                                    dbc.Col(dbc.Checkbox(id="fc-sarimax-iq", label="Use IQ scaled"), md=6),
                                                ],
                                                className="g-2",
                                            ),
                                        ],
                                        title="SARIMAX",
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            dbc.Checkbox(id="fc-general-seasonality", label="Use seasonality"),
                                        ],
                                        title="General",
                                    ),
                                ],
                                start_collapsed=True,
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Save Config", id="fc-config-save", color="success"),
                                    dbc.Button("Reset Defaults", id="fc-config-reset", color="secondary"),
                                ],
                                className="mb-2",
                            ),
                            dbc.Alert(id="fc-config-status", color="info", is_open=False),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Forecast tables",
                                        tab_id="tables",
                                        children=[
                                            dcc.Graph(id="fc-line-chart", figure={}, className="mb-2"),
                                            dash_table.DataTable(
                                                id="fc-combined",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                style_table={"overflowX": "auto"},
                                                style_cell={"fontSize": 12},
                                            ),
                                            html.Hr(),
                                            dash_table.DataTable(
                                                id="fc-pivot",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                style_table={"overflowX": "auto"},
                                                style_cell={"fontSize": 12},
                                            ),
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Normalized ratios",
                                        tab_id="ratios",
                                        children=[
                                            dcc.Graph(id="fc-ratio1-chart", figure={}, className="mb-2"),
                                            dcc.Graph(id="fc-ratio2-chart", figure={}, className="mb-2"),
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Accuracy",
                                        tab_id="accuracy",
                                        children=[
                                            dcc.Graph(id="fc-accuracy-chart", figure={}, className="mb-2"),
                                            dash_table.DataTable(
                                                id="fc-accuracy-table",
                                                data=[],
                                                columns=[],
                                                page_size=8,
                                                style_table={"overflowX": "auto"},
                                                style_cell={"fontSize": 12},
                                            ),
                                        ],
                                    ),
                                ],
                                active_tab="tables",
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="g-3",
            ),
            dcc.Store(id="fc-data-store"),
        ]

    def _daily_interval_layout(step_meta):
        return [
            dcc.Store(id="di-transform-store"),
            dcc.Store(id="di-interval-store"),
            dcc.Store(id="di-results-store"),
            dcc.Download(id="di-download-daily"),
            dcc.Download(id="di-download-interval"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Use transformed forecast", className="mb-2"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Load latest saved file", id="di-load-transform", color="secondary", outline=True),
                                        dcc.Upload(
                                            id="di-upload-transform",
                                            children=html.Div(["Upload CSV"], className="px-2"),
                                            multiple=False,
                                            accept=".csv",
                                            className="border border-secondary rounded p-2 bg-light",
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.Div(id="di-transform-msg", className="small text-muted mb-2"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-transform-group",
                                                options=[],
                                                placeholder="Select forecast group",
                                            ),
                                            md=6,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id="di-transform-month",
                                                options=[],
                                                placeholder="Select month",
                                            ),
                                            md=6,
                                        ),
                                    ],
                                    className="g-2 mb-2",
                                ),
                                html.Hr(),
                                html.H5("Interval history (last 3 months)", className="mb-2"),
                                dcc.Upload(
                                    id="di-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select interval CSV/XLSX")]),
                                    multiple=False,
                                    accept=".csv,.xls,.xlsx",
                                    className="border border-secondary rounded p-3 mb-2 bg-light",
                                ),
                                html.Div(id="di-upload-msg", className="small text-muted"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Run interval forecast", id="di-run-btn", color="primary"),
                                        dbc.Button("Download daily", id="di-download-daily-btn", color="secondary", outline=True),
                                        dbc.Button("Download interval", id="di-download-interval-btn", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                html.Div(id="di-run-status", className="small fw-semibold text-primary mt-2"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Save to disk", id="di-save-btn", color="secondary", outline=True),
                                    ],
                                    className="mt-2",
                                ),
                                html.Div(id="di-save-status", className="small text-muted mt-2"),
                                html.Div(
                                    [
                                        html.Div("Tips:", className="fw-semibold mb-1"),
                                        html.Ul(
                                            [
                                                html.Li("Date column is auto-detected (Date/ds/timestamp)."),
                                                html.Li("Interval like 09:00-09:30 or 09:00 works."),
                                                html.Li("Volume column aliases: volume, calls, items, count."),
                                                html.Li("Optional AHT column: aht, aht_sec, avg_handle_time."),
                                            ],
                                            className="small mb-0",
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                            ],
                            body=True,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                html.H5("Preview (first 200 rows)", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-transform-preview",
                                    data=[],
                                    columns=[],
                                    page_size=6,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Interval history preview (first 200)", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-preview",
                                    data=[],
                                    columns=[],
                                    page_size=6,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Daily forecast", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-daily-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                                html.Hr(),
                                html.H5("Interval forecast", className="mb-2"),
                                dash_table.DataTable(
                                    id="di-interval-forecast-table",
                                    data=[],
                                    columns=[],
                                    page_size=8,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12},
                                ),
                            ],
                            body=True,
                        ),
                        md=8,
                    ),
                ],
                className="g-3",
            ),
        ]

    def _generic_step_layout(step_meta, body, button_id, next_slug):
        has_next = bool(next_slug)
        next_label = next_slug.replace("-", " ").title() if has_next else "Forecast Home"
        next_href = f"/forecast/{next_slug}" if has_next else "/forecast"
        modal_body = f"Move to {next_label}?" if has_next else "Return to the Forecast overview?"
        next_btn_text = "Next →" if has_next else "Back to Forecast"
        save_btn_text = "Save & Continue" if has_next else "Save"
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                html.Div(body, className="mb-3"),
                dbc.Button(save_btn_text, id=button_id, color="primary"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Ready for the next step"),
                        dbc.ModalBody(modal_body),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Stay", id=f"{button_id}-close", className="me-2"),
                                dcc.Link(
                                    dbc.Button(next_btn_text, color="primary"),
                                    href=next_href,
                                    style={"textDecoration": "none"},
                                ),
                            ]
                        ),
                    ],
                    id=f"{button_id}-modal",
                    is_open=False,
                ),
            ],
            className="forecast-page",
        )

    step_meta = next((s for s in FORECAST_STEPS if s["slug"] == slug), None)
    if not step_meta:
        item = next((i for i in FORECAST_NAV if i["slug"] == slug), None)
        title = item["label"] if item else slug.replace("-", " ").title()
        emoji = item["emoji"] if item else "🧭"
        return html.Div(
            dbc.Container(
                [
                    header_bar(),
                    html.Div(
                        children=[
                            _nav_buttons(),
                            html.H1(f"{emoji} {title}", className="forecast-heading"),
                            dbc.Alert(
                                "This forecasting workspace page is under construction. "
                                "Once available, forecasts saved here can be pushed directly into planning.",
                                color="info",
                            ),
                        ],
                        className="forecast-page",
                    ),
                ],
                fluid=True,
            )
        )

    next_slug = step_meta.get("next")
    body_placeholder = html.Div(
        [
            html.P("This step is coming soon. You can still save and move forward.", className="text-muted"),
        ]
    )

    if slug == "volume-summary":
        content = _volume_summary_layout(step_meta)
    elif slug == "smoothing-anomaly":
        content = _generic_step_layout(step_meta, _smoothing_layout(step_meta), "sa-complete", next_slug)
    elif slug == "forecasting":
        content = _generic_step_layout(step_meta, _forecasting_layout(step_meta), "fc-complete", next_slug)
    elif slug == "transformation-projects":
        body = [
            html.H4("Transformation projects"),
            html.P("Track uplift, savings, and impact on your forecast pipeline."),
        ]
        content = _generic_step_layout(step_meta, body, "tp-complete", next_slug)
    elif slug == "daily-interval":
        content = _generic_step_layout(step_meta, _daily_interval_layout(step_meta), "di-complete", None)
    else:
        content = body_placeholder

    return html.Div(
        dbc.Container(
            [
                header_bar(),
                *shared_stores,
                content,
            ],
            fluid=True,
        )
    )
