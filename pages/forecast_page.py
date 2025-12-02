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
                                    dbc.Button("Start → Volume Summary", color="primary", outline=False, className="float-end"),
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
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(f"{step_meta['emoji']} {step_meta['label']}", className="mb-1"),
                        html.Div("Follow the steps to keep moving forward.", className="text-muted"),
                    ],
                ),
                dcc.Link(
                    dbc.Button("Back to models", color="link", className="px-0"),
                    href="/forecast",
                ),
            ],
            className="d-flex justify-content-between align-items-start mb-3 flex-wrap gap-2",
        )

    def _volume_summary_layout(step_meta):
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                dcc.Store(id="vs-data-store"),
                dcc.Store(id="vs-results-store"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Upload data"),
                                dcc.Upload(
                                    id="vs-upload",
                                    children=html.Div(["Drag & drop or ", html.Strong("select a CSV/XLSX file")]),
                                    multiple=False,
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

    def _generic_step_layout(step_meta, body, button_id, next_slug):
        return html.Div(
            [
                _section_heading(step_meta),
                _stepper(step_meta["slug"]),
                html.Div(body, className="mb-3"),
                dbc.Button("Save & Continue", id=button_id, color="primary"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Ready for the next step"),
                        dbc.ModalBody(f"Move to {next_slug.replace('-', ' ').title()}?"),
                        dbc.ModalFooter(
                            [
                                dbc.Button("Stay", id=f"{button_id}-close", className="me-2"),
                                dcc.Link(
                                    dbc.Button("Next →", color="primary"),
                                    href=f"/forecast/{next_slug}" if next_slug else "/forecast",
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
        body = [
            html.H4("Run smoothing & anomaly detection"),
            html.P("Use your uploaded volume summary data to smooth trends and flag anomalies. A guided workflow will appear here."),
        ]
        content = _generic_step_layout(step_meta, body, "sa-complete", next_slug)
    elif slug == "forecasting":
        body = [
            html.H4("Select models & generate forecasts"),
            html.P("Choose the best-performing models for your demand curves."),
        ]
        content = _generic_step_layout(step_meta, body, "fc-complete", next_slug)
    elif slug == "transformation-projects":
        body = [
            html.H4("Transformation projects"),
            html.P("Track uplift, savings, and impact on your forecast pipeline."),
        ]
        content = _generic_step_layout(step_meta, body, "tp-complete", next_slug)
    elif slug == "daily-interval":
        body = [
            html.H4("Daily interval forecast"),
            html.P("Push your final results into interval planning and requirements."),
        ]
        content = _generic_step_layout(step_meta, body, "di-complete", None)
    else:
        content = body_placeholder

    return html.Div(
        dbc.Container(
            [
                header_bar(),
                content,
            ],
            fluid=True,
        )
    )
