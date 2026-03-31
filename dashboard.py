"""
Israel Civil Defense Alerts — Live Dashboard
Run with:  streamlit run dashboard.py
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from charts import (
    category_breakdown_chart,
    convergence_rate_chart,
    daily_pre_alert_siren_chart,
    hourly_heatmap,
    monthly_trend_chart,
    prediction_distribution_chart,
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
    convergence_rate_over_time,
    daily_counts,
    daily_pre_alert_siren_counts,
    filter_by_date_range,
    filter_by_location,
    filter_categories,
    get_all_locations,
    heatmap_insights,
    hourly_heatmap_data,
    kpi_delta,
    kpi_summary,
    monthly_counts,
    predict_time_to_siren_now,
    top_locations,
    train_area_model,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Israel Alerts Dashboard",
    page_icon="🚨",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
for _key, _default in [("df", None), ("loaded_at", None), ("model_state", None)]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

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
            st.session_state["model_state"] = None  # invalidate stale model
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
            value=30,
            format_func=lambda v: DAY_LABELS.get(v, f"Last {v} days"),
        )
        start_date = max(min_date, max_date - timedelta(days=last_x_days))
        end_date = max_date

        st.divider()

        # ── Area filter ─────────────────────────────────────────────────────
        st.subheader("Area Filter")
        all_locations = get_all_locations(df_full)
        area_options = ["(All areas)"] + all_locations
        default_idx = area_options.index("Kfar Netter") if "Kfar Netter" in area_options else 0
        selected_area = st.selectbox(
            "Focus on area",
            options=area_options,
            index=default_idx,
        )

        st.divider()

        # ── Model training (sidebar) ─────────────────────────────────────────
        st.subheader("Prediction Model")
        area_active_sidebar = selected_area != "(All areas)"

        ms = st.session_state["model_state"]
        if ms:
            trained_for = ms["location"]
            stale = trained_for != selected_area
            status_icon = "⚠️" if stale else "✅"
            st.caption(
                f"{status_icon} Model: **{trained_for}**  \n"
                f"Trained at {ms['trained_at']} · {ms['n_samples']} samples"
            )
            if stale:
                st.caption("Area changed — retrain for accurate predictions.")

        train_label = (
            f"🧠 Train model for {selected_area}"
            if area_active_sidebar
            else "🧠 Train model  *(select an area first)*"
        )
        if st.button(train_label, use_container_width=True, disabled=not area_active_sidebar):
            df_history_train = filter_categories(
                df_full, include_drills=False, selected_category_ids=None
            )
            df_history_area_train = filter_by_location(df_history_train, selected_area)
            with st.spinner(f"Training on {selected_area} history…"):
                model_state, err = train_area_model(df_history_area_train, selected_area)
            if err:
                st.warning(err)
            else:
                st.session_state["model_state"] = model_state
                st.success(
                    f"Model ready · {model_state['n_samples']} paired events · "
                    f"avg {model_state['historical_avg_min']:.1f} min"
                )
                st.rerun()

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

# ── Derived DataFrames ────────────────────────────────────────────────────────
df_full = st.session_state["df"]
area_active = selected_area != "(All areas)"

df = filter_by_date_range(df_full, start_date, end_date)
df = filter_categories(df, include_drills=include_drills, selected_category_ids=selected_ids)

df_area = filter_by_location(df, selected_area) if area_active else df

df_history = filter_categories(df_full, include_drills=include_drills, selected_category_ids=selected_ids)
df_history_area = filter_by_location(df_history, selected_area) if area_active else df_history

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
        st.subheader("Daily Pre-Alerts vs Sirens (all areas)")
        daily_ps = daily_pre_alert_siren_counts(df)
        st.plotly_chart(daily_pre_alert_siren_chart(daily_ps), width="stretch", key="daily_ps_all")

    else:
        st.subheader(f"Area: {selected_area}")

        if df_area.empty:
            st.warning(f"No events found for **{selected_area}** in the selected time range.")
        else:
            # ── Pre-alert now button (inference only) ───────────────────────
            ms = st.session_state["model_state"]
            model_ready = ms is not None and ms.get("location") == selected_area

            if st.button("🚨 Pre-alert now...", type="primary", use_container_width=True):
                if not model_ready:
                    st.warning(
                        "No trained model for this area yet. "
                        "Click **🧠 Train model** in the sidebar first."
                    )
                else:
                    predicted = predict_time_to_siren_now(ms)
                    st.session_state["last_prediction"] = predicted

            if model_ready and "last_prediction" in st.session_state:
                predicted = st.session_state["last_prediction"]
                st.error(
                    f"### ⚠️ Pre-Alert Detected — {selected_area}\n\n"
                    f"**Estimated time to siren: {predicted:.1f} min**  "
                    f"· CI [{ms['historical_min_min']:.1f} – {ms['historical_max_min']:.1f}] min  \n"
                    f"Mean {ms['historical_avg_min']:.1f} ± {ms['historical_std_min']:.1f} min  "
                    f"· Median {ms['historical_median_min']:.1f} min  "
                    f"· {ms['n_samples']} training events"
                )
                st.plotly_chart(
                    prediction_distribution_chart(ms, predicted),
                    width="stretch",
                    key="pred_dist",
                )
            elif not model_ready:
                st.caption("⚠️ Train the model in the sidebar for siren-time predictions.")

            st.divider()

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
                timings = area_timings(df_history_area, selected_area, window_minutes=15)

            if timings["n_pre_alerts"] == 0:
                st.info("No pre-alerts found for this area in the full dataset.")
            else:
                t1, t2, t3 = st.columns(3)

                avg_ps = timings["avg_pre_to_siren_min"]
                t1.metric(
                    "Avg Pre-Alert → Siren",
                    f"{avg_ps:.1f} min" if avg_ps is not None else "N/A",
                    help="Average minutes from a pre-alert to the next siren (within 15-min window)",
                )

                avg_pc = timings["avg_pre_to_clear_min"]
                t2.metric(
                    "Avg Pre-Alert → All-Clear",
                    f"{avg_pc:.1f} min" if avg_pc is not None else "N/A",
                    help="Average minutes from a pre-alert to the next all-clear (within 15-min window)",
                )

                rate = timings["convergence_rate"]
                t3.metric(
                    "Convergence Rate",
                    f"{rate * 100:.1f}%" if rate is not None else "N/A",
                    help=(
                        f"{timings['n_converged']} of {timings['n_pre_alerts']} pre-alerts "
                        "were followed by an actual siren within 15 minutes"
                    ),
                )

                st.caption(
                    f"Based on **{timings['n_pre_alerts']}** pre-alerts and "
                    f"**{timings['n_sirens']}** sirens in **{selected_area}** (full dataset)."
                )

            st.divider()

            # ── Daily bar chart ─────────────────────────────────────────────
            st.subheader("Daily Pre-Alerts vs Sirens")
            daily_ps = daily_pre_alert_siren_counts(df_area)
            st.plotly_chart(
                daily_pre_alert_siren_chart(daily_ps), width="stretch", key="daily_ps_area"
            )

            st.divider()

            # ── Category breakdown for area ─────────────────────────────────
            st.subheader("Event Breakdown by Type")
            cat_df_area = category_totals(df_area)
            st.plotly_chart(
                category_breakdown_chart(cat_df_area), width="stretch", key="cat_breakdown_area"
            )

            st.divider()

            # ── Convergence rate over time ───────────────────────────────────
            st.subheader("Convergence Rate Over Time")
            conv_df = convergence_rate_over_time(df_history_area, selected_area, window_minutes=15)
            st.plotly_chart(
                convergence_rate_chart(conv_df, selected_area),
                width="stretch",
                key="conv_rate_chart",
            )
            st.caption(
                "% of pre-alerts followed by an actual siren within 15 minutes.  "
                "Red line = 7-day rolling average."
            )

            st.divider()

            # ── High-risk days & times ───────────────────────────────────────
            st.subheader("High-Risk Days & Times")
            risk_insights = heatmap_insights(df_history_area, top_n=10)
            if risk_insights:
                for i, insight in enumerate(risk_insights, 1):
                    st.markdown(f"{i}. {insight}")
            else:
                st.info("Not enough siren data to compute high-risk windows for this area.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Overview (scoped to area when active)
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # Overview is always unfiltered by area
    df_view = df
    df_view_history = df_history

    # ── KPI row ─────────────────────────────────────────────────────────────
    kpi = kpi_summary(df_view)
    deltas = kpi_delta(df_full, start_date, end_date)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Events", f"{kpi['total']:,}", delta=int(deltas.get("total", 0)))
    col2.metric("Sirens / Alerts", f"{kpi['sirens']:,}", delta=int(deltas.get("sirens", 0)))
    col3.metric("Pre-Alerts", f"{kpi['pre_alerts']:,}", delta=int(deltas.get("pre_alerts", 0)))
    col4.metric("All-Clear Sent", f"{kpi['all_clear']:,}", delta=int(deltas.get("all_clear", 0)))
    col5.metric("Unique Locations", f"{kpi['unique_locations']:,}")

    st.caption(
        f"Showing **{kpi['total']:,}** events from **{start_date}** to **{end_date}** (all areas)  |  "
        f"Last 24 h: **{kpi['last_24h']:,}**  |  Last 7 days: **{kpi['last_7d']:,}**"
    )

    st.divider()

    # ── Timeline ─────────────────────────────────────────────────────────────
    weekly = daily_counts(df_view)
    st.plotly_chart(timeline_chart(weekly), width="stretch", key="timeline_overview")

    # ── Locations + category breakdown ───────────────────────────────────────
    left, right = st.columns([3, 2])
    with left:
        loc_df = top_locations(df_view, n=top_n)
        st.plotly_chart(top_locations_chart(loc_df), width="stretch", key="top_locations_overview")
    with right:
        cat_df = category_totals(df_view)
        st.plotly_chart(
            category_breakdown_chart(cat_df), width="stretch", key="cat_breakdown_overview"
        )

    # ── Monthly trend + heatmap ───────────────────────────────────────────────
    left2, right2 = st.columns(2)
    with left2:
        monthly_df = monthly_counts(df_view_history)
        st.plotly_chart(monthly_trend_chart(monthly_df), width="stretch", key="monthly_overview")
    with right2:
        heatmap_df = hourly_heatmap_data(df_view_history)
        st.plotly_chart(hourly_heatmap(heatmap_df), width="stretch", key="heatmap_overview")

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Raw data (filtered)"):
        display_cols = ["alertDate", "location_en", "category_en", "category_desc"]
        display_cols = [c for c in display_cols if c in df_view.columns]
        st.dataframe(
            df_view[display_cols]
            .sort_values("alertDate", ascending=False)
            .reset_index(drop=True),
            width="stretch",
            height=400,
            key="raw_data_table",
        )
        csv_bytes_dl = df_view[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_bytes_dl,
            file_name=f"israel_alerts_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
