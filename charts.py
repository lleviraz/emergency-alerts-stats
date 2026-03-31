"""
Plotly figure factories for the Israel Alerts dashboard.
All functions return a go.Figure. No Streamlit imports here.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Colour palette – consistent across charts
CATEGORY_COLORS = {
    "Rocket / Missile Fire": "#e63946",
    "Hostile Aircraft Intrusion": "#f4a261",
    "Unconventional Rocket": "#c9184a",
    "Warning": "#ffb703",
    "Terror Warning": "#ff6b6b",
    "Hazardous Materials": "#9b5de5",
    "Earthquake": "#8338ec",
    "Tsunami": "#3a86ff",
    "Radiological Event": "#06d6a0",
    "Terrorist Infiltration": "#d62828",
    "Hostile Vehicle": "#fb8500",
    "All Clear": "#2dc653",
    "Pre-Alert": "#ffd166",
    "Drill — Rockets": "#adb5bd",
    "Drill — Earthquake": "#adb5bd",
    "Drill — Hazardous Materials": "#adb5bd",
}

_PLOTLY_TEMPLATE = "plotly_dark"


def timeline_chart(weekly_df: pd.DataFrame) -> go.Figure:
    """
    Stacked area chart: weekly event counts coloured by category.
    weekly_df columns: week_start, category_en, count.
    """
    if weekly_df.empty:
        return _empty_figure("No data for selected range")

    color_map = {cat: CATEGORY_COLORS.get(cat, "#888") for cat in weekly_df["category_en"].unique()}

    fig = px.area(
        weekly_df,
        x="week_start",
        y="count",
        color="category_en",
        color_discrete_map=color_map,
        labels={"week_start": "Week", "count": "Events", "category_en": "Type"},
        title="Weekly Events Over Time",
        template=_PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        legend_title_text="Alert Type",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def category_breakdown_chart(cat_totals: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of events per alert category, sorted descending.
    cat_totals columns: category_en, count.
    """
    if cat_totals.empty:
        return _empty_figure("No data")

    df = cat_totals.sort_values("count", ascending=True)
    colors = [CATEGORY_COLORS.get(c, "#888") for c in df["category_en"]]

    fig = go.Figure(
        go.Bar(
            x=df["count"],
            y=df["category_en"],
            orientation="h",
            marker_color=colors,
            text=df["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Events by Alert Type",
        xaxis_title="Count",
        yaxis_title=None,
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=180, r=60, t=50, b=40),
        height=max(300, len(df) * 40),
    )
    return fig


def top_locations_chart(loc_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top N locations.
    loc_df columns: location (or index name), count.
    """
    if loc_df.empty:
        return _empty_figure("No location data")

    # Handle value_counts output — first column is location name
    loc_col = loc_df.columns[0]
    df = loc_df.sort_values("count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["count"],
            y=df[loc_col],
            orientation="h",
            marker_color="#e63946",
            text=df["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top Locations by Alert Count",
        xaxis_title="Events",
        yaxis_title=None,
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=200, r=60, t=50, b=40),
        height=max(300, len(df) * 28),
    )
    return fig


def monthly_trend_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of total siren-type events per month.
    monthly_df columns: month_dt (datetime), count.
    """
    if monthly_df.empty:
        return _empty_figure("No siren data")

    fig = go.Figure(
        go.Scatter(
            x=monthly_df["month_dt"],
            y=monthly_df["count"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#e63946", width=2),
            name="Siren Events",
        )
    )
    fig.update_layout(
        title="Monthly Siren / Missile Events",
        xaxis_title="Month",
        yaxis_title="Events",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def hourly_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """
    Activity heatmap: day-of-week (rows) × hour-of-day (columns).
    heatmap_df: index = day labels (Mon…Sun), columns = 0..23.
    """
    if heatmap_df.empty:
        return _empty_figure("No data")

    fig = go.Figure(
        go.Heatmap(
            z=heatmap_df.values,
            x=[f"{h:02d}:00" for h in heatmap_df.columns],
            y=heatmap_df.index.tolist(),
            colorscale="Reds",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Events: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Alert Activity by Day & Hour (UTC+2)",
        xaxis_title="Hour of Day",
        yaxis_title=None,
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def daily_pre_alert_siren_chart(daily_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: each day has up to 2 bars — Pre-Alert and Siren.
    daily_df columns: parsed_date, type ('Pre-Alert' or 'Siren'), count.
    """
    if daily_df.empty:
        return _empty_figure("No pre-alert / siren data for selected range")

    colors = {
        "Pre-Alert": CATEGORY_COLORS["Pre-Alert"],
        "Siren": CATEGORY_COLORS["Rocket / Missile Fire"],
    }

    fig = go.Figure()
    for event_type in ["Siren", "Pre-Alert"]:
        subset = daily_df[daily_df["type"] == event_type]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["parsed_date"],
                y=subset["count"],
                name=event_type,
                marker_color=colors.get(event_type, "#888"),
            )
        )

    fig.update_layout(
        title="Daily Pre-Alerts vs Sirens",
        xaxis_title="Date",
        yaxis_title="Events",
        barmode="group",
        template=_PLOTLY_TEMPLATE,
        legend_title_text="Type",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── helpers ──────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
