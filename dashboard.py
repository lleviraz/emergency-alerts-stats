"""
Israel Civil Defense Alerts — Live Dashboard
Run with:  streamlit run dashboard.py
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from charts import (
    category_breakdown_chart,
    daily_pre_alert_siren_chart,
    hourly_heatmap,
    monthly_trend_chart,
    timeline_chart,
    top_locations_chart,
)
from data_loader import load_dashboard_df, stream_download
from transforms import (
    CATEGORY_MAP,
    DRILL_CATEGORIES,
    apply_english_labels,
    apply_english_locations,
    area_timings,
    category_totals,
    daily_counts,
    daily_pre_alert_siren_counts,
    filter_by_date_range,
    filter_by_location,
    filter_categories,
    get_all_locations,
    hourly_heatmap_data,
    kpi_delta,
    kpi_summary,
    monthly_counts,
    top_locations,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Israel Alerts Dashboard",
    page_icon="🚨",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "loaded_at" not in st.session_state:
    st.session_state["loaded_at"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")

    # ── Load / Refresh ──────────────────────────────────────────────────────
    if st.button("⬇ Load / Refresh Data", use_container_width=True, type="primary"):
        progress_bar = st.progress(0.0, text="Connecting…")
        status_text = st.empty()

        def _on_progress(fraction: float) -> None:
            progress_bar.progress(fraction, text=f"Downloading… {fraction * 100:.0f}%")

        try:
            csv_bytes = stream_download(_on_progress)
            status_text.text("Parsing data…")
            raw = load_dashboard_df(csv_bytes)
            raw = apply_english_labels(raw)
            raw = apply_english_locations(raw)
            st.session_state["df"] = raw
            st.session_state["loaded_at"] = datetime.now()
            progress_bar.empty()
            status_text.empty()
            st.rerun()
        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Failed to load data: {exc}")

    if st.session_state["loaded_at"]:
        st.caption(
            f"Last loaded: {st.session_state['loaded_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    st.divider()

    df_full: pd.DataFrame | None = st.session_state["df"]

    if df_full is not None:
        # ── Last X days slider ──────────────────────────────────────────────
        st.subheader("Time Range")
        max_date = df_full["parsed_date"].max().date()
        min_date = df_full["parsed_date"].min().date()
        total_days = (max_date - min_date).days

        DAY_OPTIONS = [7, 14, 30, 60, 90, 180, 365, total_days]
        DAY_LABELS = {
            7: "Last 7 days",
            14: "Last 14 days",
            30: "Last 30 days",
            60: "Last 60 days",
            90: "Last 90 days",
            180: "Last 6 months",
            365: "Last year",
            total_days: "All time",
        }
        last_x_days = st.select_slider(
            "Show last N days",
            options=DAY_OPTIONS,
            value=90,
            format_func=lambda v: DAY_LABELS.get(v, f"Last {v} days"),
        )
        start_date = max(min_date, max_date - timedelta(days=last_x_days))
        end_date = max_date

        st.divider()

        # ── Area filter ─────────────────────────────────────────────────────
        st.subheader("Area Filter")
        all_locations = get_all_locations(df_full)
        selected_area = st.selectbox(
            "Focus on area",
            options=["(All areas)"] + all_locations,
            index=0,
        )

        st.divider()

        # ── Category filter ─────────────────────────────────────────────────
        st.subheader("Alert Types")
        all_categories = sorted(
            df_full["category"].dropna().unique().astype(int).tolist()
        )
        category_options = {
            cat: CATEGORY_MAP.get(cat, f"Unknown ({cat})")
            for cat in all_categories
        }
        default_selected = [c for c in all_categories if c not in DRILL_CATEGORIES]
        selected_ids = st.multiselect(
            "Show categories",
            options=list(category_options.keys()),
            default=default_selected,
            format_func=lambda c: category_options[c],
        )
        include_drills = st.checkbox("Include drill events", value=False)

        st.divider()

        # ── Top N locations ─────────────────────────────────────────────────
        st.subheader("Top Locations")
        top_n = st.slider("Show top N locations", min_value=5, max_value=50, value=20, step=5)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🚨 Israel Civil Defense Alerts Dashboard")
st.caption(
    "Data source: [github.com/dleshem/israel-alerts-data]"
    "(https://github.com/dleshem/israel-alerts-data)"
)

if st.session_state["df"] is None:
    st.info(
        "No data loaded yet. Click **⬇ Load / Refresh Data** in the sidebar to begin.\n\n"
        "The dataset is ~50 MB so the first download may take a moment."
    )
    st.stop()

# Apply date + category filters
df_full = st.session_state["df"]
df = filter_by_date_range(df_full, start_date, end_date)
df = filter_categories(df, include_drills=include_drills, selected_category_ids=selected_ids)

# Apply area filter (for area-specific views)
area_active = selected_area != "(All areas)"
df_area = filter_by_location(df, selected_area) if area_active else df

if df.empty:
    st.warning("No events match the current filters. Try a wider time range or more alert types.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_area, tab_overview = st.tabs(["📍 Area Analysis", "📊 Overview"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Area Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_area:

    if not area_active:
        st.info(
            "Select an **area** in the sidebar to see timing statistics and "
            "pre-alert → siren analytics for that location."
        )
        # Still show the daily pre-alert/siren chart for all areas
        st.subheader("Daily Pre-Alerts vs Sirens (all areas)")
        daily_ps = daily_pre_alert_siren_counts(df)
        st.plotly_chart(daily_pre_alert_siren_chart(daily_ps), width="stretch")
    else:
        st.subheader(f"Area: {selected_area}")

        if df_area.empty:
            st.warning(f"No events found for **{selected_area}** in the selected time range.")
        else:
            # ── KPI row for area ────────────────────────────────────────────
            kpi = kpi_summary(df_area)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Events", f"{kpi['total']:,}")
            c2.metric("Sirens", f"{kpi['sirens']:,}")
            c3.metric("Pre-Alerts", f"{kpi['pre_alerts']:,}")
            c4.metric("All-Clear Sent", f"{kpi['all_clear']:,}")

            st.divider()

            # ── Timing analytics ────────────────────────────────────────────
            st.subheader("Pre-Alert Timing Analysis")

            with st.spinner("Computing timing statistics…"):
                timings = area_timings(df_area, selected_area, window_minutes=60)

            if timings["n_pre_alerts"] == 0:
                st.info("No pre-alerts found for this area in the selected time range.")
            else:
                t1, t2, t3 = st.columns(3)

                # 3.1 — Pre-alert → Siren
                avg_ps = timings["avg_pre_to_siren_min"]
                t1.metric(
                    "Avg Pre-Alert → Siren",
                    f"{avg_ps:.1f} min" if avg_ps is not None else "N/A",
                    help="Average minutes from a pre-alert to the next siren in this area (within 60-min window)",
                )

                # 3.2 — Pre-alert → All-Clear
                avg_pc = timings["avg_pre_to_clear_min"]
                t2.metric(
                    "Avg Pre-Alert → All-Clear",
                    f"{avg_pc:.1f} min" if avg_pc is not None else "N/A",
                    help="Average minutes from a pre-alert to the next all-clear in this area (within 60-min window)",
                )

                # 3.3 — Convergence rate
                rate = timings["convergence_rate"]
                t3.metric(
                    "Convergence Rate",
                    f"{rate * 100:.1f}%" if rate is not None else "N/A",
                    help=(
                        f"{timings['n_converged']} of {timings['n_pre_alerts']} pre-alerts "
                        "were followed by an actual siren within 60 minutes"
                    ),
                )

                st.caption(
                    f"Based on **{timings['n_pre_alerts']}** pre-alerts and "
                    f"**{timings['n_sirens']}** sirens in **{selected_area}** "
                    f"from {start_date} to {end_date}."
                )

            st.divider()

            # ── 3.5 — Daily bar chart ───────────────────────────────────────
            st.subheader("Daily Pre-Alerts vs Sirens")
            daily_ps = daily_pre_alert_siren_counts(df_area)
            st.plotly_chart(daily_pre_alert_siren_chart(daily_ps), width="stretch")

            st.divider()

            # ── Top sub-locations within area ───────────────────────────────
            st.subheader("Event Breakdown by Type")
            cat_df_area = category_totals(df_area)
            st.plotly_chart(category_breakdown_chart(cat_df_area), width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Overview
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI row ─────────────────────────────────────────────────────────────
    kpi = kpi_summary(df)
    deltas = kpi_delta(df_full, start_date, end_date)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Events", f"{kpi['total']:,}", delta=int(deltas.get("total", 0)))
    col2.metric("Sirens / Alerts", f"{kpi['sirens']:,}", delta=int(deltas.get("sirens", 0)))
    col3.metric("Pre-Alerts", f"{kpi['pre_alerts']:,}", delta=int(deltas.get("pre_alerts", 0)))
    col4.metric("All-Clear Sent", f"{kpi['all_clear']:,}", delta=int(deltas.get("all_clear", 0)))
    col5.metric("Unique Locations", f"{kpi['unique_locations']:,}")

    st.caption(
        f"Showing **{kpi['total']:,}** events from **{start_date}** to **{end_date}**  |  "
        f"Last 24 h: **{kpi['last_24h']:,}**  |  Last 7 days: **{kpi['last_7d']:,}**"
    )

    st.divider()

    # ── Timeline ─────────────────────────────────────────────────────────────
    weekly = daily_counts(df)
    st.plotly_chart(timeline_chart(weekly), width="stretch")

    # ── Locations + category breakdown ───────────────────────────────────────
    left, right = st.columns([3, 2])
    with left:
        loc_df = top_locations(df, n=top_n)
        st.plotly_chart(top_locations_chart(loc_df), width="stretch")
    with right:
        cat_df = category_totals(df)
        st.plotly_chart(category_breakdown_chart(cat_df), width="stretch")

    # ── Monthly trend + heatmap ───────────────────────────────────────────────
    left2, right2 = st.columns(2)
    with left2:
        monthly_df = monthly_counts(df)
        st.plotly_chart(monthly_trend_chart(monthly_df), width="stretch")
    with right2:
        heatmap_df = hourly_heatmap_data(df)
        st.plotly_chart(hourly_heatmap(heatmap_df), width="stretch")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Raw data (filtered)"):
        display_cols = ["alertDate", "location_en", "category_en", "category_desc"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("alertDate", ascending=False).reset_index(drop=True),
            width="stretch",
            height=400,
        )
        csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_bytes,
            file_name=f"israel_alerts_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
