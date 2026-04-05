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


def prediction_distribution_chart(model_state: dict, predicted_min: float) -> go.Figure:
    """
    Histogram of historical pre-alert → siren times with vertical lines for
    min, max, mean (±1σ band), median, and the model prediction.
    """
    import numpy as np

    y = np.array(model_state["y_values"])
    mean = model_state["historical_avg_min"]
    std = model_state["historical_std_min"]
    median = model_state["historical_median_min"]
    y_min = model_state["historical_min_min"]
    y_max = model_state["historical_max_min"]

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=y,
            nbinsx=min(30, max(10, len(y) // 2)),
            name="Observed",
            marker_color="#4a4e69",
            opacity=0.75,
        )
    )

    # ±1σ band
    fig.add_vrect(
        x0=max(0, mean - std),
        x1=mean + std,
        fillcolor="rgba(255, 209, 102, 0.15)",
        line_width=0,
        annotation_text="±1σ",
        annotation_position="top left",
        annotation_font_size=11,
    )

    # Min line
    fig.add_vline(
        x=y_min, line_dash="dot", line_color="#adb5bd", line_width=1.5,
        annotation_text=f"Min {y_min:.1f}m", annotation_position="top right",
        annotation_font_size=10,
    )
    # Max line
    fig.add_vline(
        x=y_max, line_dash="dot", line_color="#adb5bd", line_width=1.5,
        annotation_text=f"Max {y_max:.1f}m", annotation_position="top left",
        annotation_font_size=10,
    )
    # Mean line
    fig.add_vline(
        x=mean, line_dash="dash", line_color="#ffd166", line_width=2,
        annotation_text=f"Mean {mean:.1f}m", annotation_position="top right",
        annotation_font_size=11,
    )
    # Median line
    fig.add_vline(
        x=median, line_dash="dashdot", line_color="#06d6a0", line_width=2,
        annotation_text=f"Median {median:.1f}m", annotation_position="top left",
        annotation_font_size=11,
    )
    # Prediction line
    fig.add_vline(
        x=predicted_min, line_color="#e63946", line_width=3,
        annotation_text=f"Prediction {predicted_min:.1f}m",
        annotation_position="top right",
        annotation_font_size=12,
    )

    fig.update_layout(
        title=f"Pre-Alert → Siren Time Distribution ({model_state['n_samples']} events)",
        xaxis_title="Minutes to Siren",
        yaxis_title="Count",
        template=_PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def convergence_rate_chart(conv_df: pd.DataFrame, area: str) -> go.Figure:
    """
    Line chart of daily convergence rate (% of pre-alerts followed by siren)
    over time, with a rolling 7-day average smoothing.
    """
    if conv_df.empty:
        return _empty_figure("No convergence data available")

    import numpy as np

    # Rolling 7-day mean (pad with NaN for short series)
    rolling = (
        conv_df.set_index("date")["convergence_rate"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index()
    )

    fig = go.Figure()

    # Raw daily dots
    fig.add_trace(
        go.Scatter(
            x=conv_df["date"],
            y=conv_df["convergence_rate"] * 100,
            mode="markers",
            marker=dict(color="#adb5bd", size=5, opacity=0.5),
            name="Daily",
            hovertemplate="%{x}<br>%{y:.0f}%<extra></extra>",
        )
    )

    # Smoothed line
    fig.add_trace(
        go.Scatter(
            x=rolling["date"],
            y=rolling["convergence_rate"] * 100,
            mode="lines",
            line=dict(color="#e63946", width=2),
            name="7-day avg",
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Convergence Rate Over Time — {area}",
        xaxis_title="Date",
        yaxis_title="% Pre-Alerts → Siren",
        yaxis=dict(range=[0, 105]),
        template=_PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def interactive_risk_windows_chart(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart of siren event counts per time-of-day bucket.
    Bars are stacked by day-of-week.
    A Plotly dropdown lets the user switch between 2 h, 4 h, and 6 h bucket sizes.
    """
    from transforms import SIREN_CATEGORIES

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    DAY_COLORS = ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#a8dadc", "#457b9d"]

    sirens = df[df["category"].isin(SIREN_CATEGORIES)].dropna(subset=["hour", "day_of_week"])

    # Build one set of 7 traces per window size (3 sizes × 7 days = 21 traces total)
    all_traces: list[go.Bar] = []
    visibility: dict[int, list[bool]] = {2: [], 4: [], 6: []}

    for window_hours in (2, 4, 6):
        n_buckets = 24 // window_hours
        labels = [
            f"{b * window_hours:02d}:00–{(b + 1) * window_hours:02d}:00"
            for b in range(n_buckets)
        ]

        for dow in range(7):
            dow_sirens = sirens[sirens["day_of_week"] == dow]
            counts = [
                int((dow_sirens["hour"] // window_hours == b).sum())
                for b in range(n_buckets)
            ]
            all_traces.append(
                go.Bar(
                    x=labels,
                    y=counts,
                    name=DAY_NAMES[dow],
                    marker_color=DAY_COLORS[dow],
                    visible=(window_hours == 2),
                    legendgroup=DAY_NAMES[dow],
                    showlegend=(window_hours == 2),  # only show legend once
                )
            )

        n_prev = len(all_traces)
        for w, vis_list in visibility.items():
            # The 7 traces just added belong to window_hours
            is_this_window = w == window_hours
            vis_list.extend([is_this_window] * 7)

    fig = go.Figure(data=all_traces)

    fig.update_layout(
        barmode="stack",
        title="Siren Events by Time-of-Day Window",
        xaxis_title="Time of Day",
        yaxis_title="Event Count",
        template=_PLOTLY_TEMPLATE,
        legend_title="Day of Week",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=100, b=40),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="2-hour windows",
                        method="update",
                        args=[
                            {"visible": visibility[2]},
                            {"title": "Siren Events by Time-of-Day Window (2 h buckets)"},
                        ],
                    ),
                    dict(
                        label="4-hour windows",
                        method="update",
                        args=[
                            {"visible": visibility[4]},
                            {"title": "Siren Events by Time-of-Day Window (4 h buckets)"},
                        ],
                    ),
                    dict(
                        label="6-hour windows",
                        method="update",
                        args=[
                            {"visible": visibility[6]},
                            {"title": "Siren Events by Time-of-Day Window (6 h buckets)"},
                        ],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.18,
                yanchor="top",
            )
        ],
    )
    return fig


def high_risk_windows_chart(risk_df: pd.DataFrame, area: str) -> go.Figure:
    """
    Horizontal bar chart of the top N highest-risk day+hour combinations,
    sorted descending by siren event count.
    risk_df columns: label, count.
    """
    if risk_df.empty:
        return _empty_figure("Not enough siren data for this area")

    df = risk_df.sort_values("count", ascending=True)  # ascending so top is at chart top

    # Colour bars by count intensity
    fig = go.Figure(
        go.Bar(
            x=df["count"],
            y=df["label"],
            orientation="h",
            marker=dict(
                color=df["count"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Events"),
            ),
            text=df["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top High-Risk Time Windows — {area}",
        xaxis_title="Siren Events",
        yaxis_title=None,
        template=_PLOTLY_TEMPLATE,
        margin=dict(l=200, r=80, t=50, b=40),
        height=max(300, len(df) * 38),
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


def comparison_convergence_chart(conv_data: dict) -> go.Figure:
    """
    Multi-line 7-day rolling convergence rate chart for 2-5 areas.
    conv_data: {area_name: DataFrame(date, convergence_rate)}
    """
    COLORS = ["#e63946", "#2dc653", "#ffd166", "#a8dadc", "#f4a261"]
    fig = go.Figure()

    for i, (area, df) in enumerate(conv_data.items()):
        if df.empty:
            continue
        rolling = (
            df.set_index("date")["convergence_rate"]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index()
        )
        # Raw dots (faint)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["convergence_rate"] * 100,
                mode="markers",
                marker=dict(color=COLORS[i % len(COLORS)], size=4, opacity=0.25),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # 7-day rolling line
        fig.add_trace(
            go.Scatter(
                x=rolling["date"],
                y=rolling["convergence_rate"] * 100,
                mode="lines",
                name=area,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"<b>{area}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )

    if not fig.data:
        return _empty_figure("No convergence data for selected areas")

    fig.update_layout(
        title="Convergence Rate Comparison — 7-day rolling average",
        xaxis_title="Date",
        yaxis_title="% Pre-Alerts → Siren",
        yaxis=dict(range=[0, 105]),
        template=_PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=80, b=40),
    )
    return fig


def comparison_summary_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    Grouped horizontal bar chart comparing key metrics across areas.
    summary_df columns: area, metric, value.
    """
    if summary_df.empty:
        return _empty_figure("No data")

    metrics = summary_df["metric"].unique().tolist()
    areas = summary_df["area"].unique().tolist()
    COLORS = ["#e63946", "#2dc653", "#ffd166", "#a8dadc", "#f4a261"]

    fig = go.Figure()
    for i, area in enumerate(areas):
        sub = summary_df[summary_df["area"] == area]
        fig.add_trace(
            go.Bar(
                y=sub["metric"],
                x=sub["value"],
                name=area,
                orientation="h",
                marker_color=COLORS[i % len(COLORS)],
            )
        )

    fig.update_layout(
        title="Summary Statistics by Area",
        xaxis_title="Value",
        yaxis_title=None,
        barmode="group",
        template=_PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=200, r=40, t=80, b=40),
        height=max(300, len(metrics) * 60),
    )
    return fig


# ── helpers ──────────────────────────────────────────────────────────────────

def siren_heatmap_chart(counts_df: pd.DataFrame) -> go.Figure:
    """
    Density heatmap of siren events overlaid on a zoomable OpenStreetMap of Israel.
    counts_df must have columns: location, lat, lon, count.
    """
    if counts_df.empty:
        return _empty_figure("No geocoded siren data available.")

    fig = px.density_mapbox(
        counts_df,
        lat="lat",
        lon="lon",
        z="count",
        hover_name="location",
        hover_data={"count": True, "lat": False, "lon": False},
        radius=28,
        center={"lat": 31.5, "lon": 34.9},
        zoom=6.5,
        mapbox_style="open-street-map",
        color_continuous_scale="YlOrRd",
        opacity=0.65,
        labels={"count": "Siren events"},
        title="Siren Events Heatmap",
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="Sirens", thickness=14, len=0.6),
    )
    return fig


def risk_correlation_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: event frequency score (x) vs convergence rate (y) per area.
    Bubble size ∝ n_pre_alerts.  Colour = current 60/40 risk score.
    Pearson r between the two components is shown in the title.

    Interpretation: high r → the two signals are redundant (weights barely
    matter); low r → they capture independent dimensions (weights matter more).
    """
    if summary_df.empty or len(summary_df) < 3:
        return _empty_figure("Not enough areas for correlation analysis (need ≥ 3)")

    df = summary_df.copy()
    df["risk_score"] = (
        (0.6 * df["convergence_rate"] + 0.4 * df["freq_score"]) * 100
    ).round(1)

    r = df["freq_score"].corr(df["convergence_rate"])   # Pearson r (pandas default)

    fig = px.scatter(
        df,
        x="freq_score",
        y="convergence_rate",
        size="n_pre_alerts",
        color="risk_score",
        hover_name="area",
        hover_data={"n_pre_alerts": True, "n_sirens": True,
                    "freq_score": True, "convergence_rate": True, "risk_score": True},
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        title=f"Event Frequency vs Convergence Rate — Pearson r = {r:.2f}",
        labels={
            "freq_score": "Event Frequency score (pre-alerts / day, capped at 1)",
            "convergence_rate": "Convergence Rate (0 – 1)",
            "risk_score": "Risk Score (60/40)",
        },
        template=_PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=460,
        coloraxis_colorbar=dict(title="Risk", thickness=14, len=0.7),
    )
    return fig


def risk_sensitivity_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of Spearman rank-correlation (ρ) between the area ranking at
    each convergence-rate weight (0 % … 100 %) and the baseline 60/40 ranking.

    High ρ across the full sweep → the ranking is stable regardless of the
    exact weight choice.  A dip below ~0.90 signals that the ordering is
    sensitive to the weight and the 60/40 choice matters more.
    """
    if summary_df.empty or len(summary_df) < 3:
        return _empty_figure("Not enough areas for sensitivity analysis (need ≥ 3)")

    df = summary_df.copy()
    baseline_w = 0.60
    baseline_scores = (
        baseline_w * df["convergence_rate"]
        + (1 - baseline_w) * df["freq_score"]
    )

    weights = [w / 100 for w in range(0, 101, 5)]   # 0.00 … 1.00 in steps of 0.05
    # Spearman ρ = Pearson r of ranks — no scipy dependency needed.
    baseline_ranks = baseline_scores.rank()
    rhos = []
    for w in weights:
        scores = w * df["convergence_rate"] + (1 - w) * df["freq_score"]
        rho = baseline_ranks.corr(scores.rank())   # Pearson of ranks = Spearman ρ
        rhos.append(rho)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[w * 100 for w in weights],
        y=rhos,
        mode="lines+markers",
        line=dict(color="#ffd166", width=2),
        marker=dict(size=7),
        name="Spearman ρ vs 60/40 baseline",
        hovertemplate="Conv. weight = %{x}%<br>ρ = %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(
        x=60,
        line_dash="dash",
        line_color="#e63946",
        annotation_text="Current (60 / 40)",
        annotation_position="top right",
    )
    fig.add_hrect(
        y0=0.90,
        y1=1.02,
        fillcolor="rgba(45,198,83,0.08)",
        line_width=0,
        annotation_text="Stable zone (ρ ≥ 0.90)",
        annotation_position="top left",
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title="Rank Stability: how much does the weight choice affect area rankings?",
        xaxis_title="Convergence-rate weight (%)",
        yaxis_title="Spearman ρ vs 60/40 baseline",
        yaxis=dict(range=[max(0.0, min(rhos) - 0.05), 1.02]),
        height=360,
    )
    return fig


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
