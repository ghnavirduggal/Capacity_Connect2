from __future__ import annotations
from dash import dcc, html
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
    """Placeholder subpage layout for forecasting workspace routes."""
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
