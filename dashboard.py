"""
Israel Civil Defense Alerts — Live Dashboard
Run with:  streamlit run dashboard.py
"""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from charts import (
    category_breakdown_chart,
    hourly_heatmap,
    monthly_trend_chart,
    timeline_chart,
    top_locations_chart,
)
from data_loader import load_dashboard_df
from transforms import (
    CATEGORY_MAP,
    DRILL_CATEGORIES,
    apply_english_labels,
    apply_english_locations,
    category_totals,
    daily_counts,
    filter_by_date_range,
    filter_categories,
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

    # Load / Refresh button
    if st.button("⬇ Load / Refresh Data", use_container_width=True, type="primary"):
        with st.spinner("Downloading ~50 MB dataset…"):
            try:
                st.cache_data.clear()
                raw = load_dashboard_df()
                raw = apply_english_labels(raw)
                raw = apply_english_locations(raw)
                st.session_state["df"] = raw
                st.session_state["loaded_at"] = datetime.now()
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load data: {exc}")

    if st.session_state["loaded_at"]:
        st.caption(
            f"Last loaded: {st.session_state['loaded_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    st.divider()

    df_full: pd.DataFrame | None = st.session_state["df"]

    if df_full is not None:
        # Date range
        min_date = df_full["parsed_date"].min().date()
        max_date = df_full["parsed_date"].max().date()
        default_start = max(min_date, max_date - timedelta(days=365))

        st.subheader("Date Range")
        start_date = st.date_input("From", value=default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.error("'From' must be before 'To'.")
            st.stop()

        st.divider()

        # Category filter
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

        # Top N locations
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

# Apply filters
df_full = st.session_state["df"]
df = filter_by_date_range(df_full, start_date, end_date)
df = filter_categories(df, include_drills=include_drills, selected_category_ids=selected_ids)

if df.empty:
    st.warning("No events match the current filters. Try widening the date range or selecting more alert types.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
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

# ── Timeline (full width) ─────────────────────────────────────────────────────
weekly = daily_counts(df)
st.plotly_chart(timeline_chart(weekly), use_container_width=True)

# ── Row 2: locations + category breakdown ─────────────────────────────────────
left, right = st.columns([3, 2])
with left:
    loc_df = top_locations(df, n=top_n)
    st.plotly_chart(top_locations_chart(loc_df), use_container_width=True)
with right:
    cat_df = category_totals(df)
    st.plotly_chart(category_breakdown_chart(cat_df), use_container_width=True)

# ── Row 3: monthly trend + heatmap ────────────────────────────────────────────
left2, right2 = st.columns(2)
with left2:
    monthly_df = monthly_counts(df)
    st.plotly_chart(monthly_trend_chart(monthly_df), use_container_width=True)
with right2:
    heatmap_df = hourly_heatmap_data(df)
    st.plotly_chart(hourly_heatmap(heatmap_df), use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Raw data (filtered)"):
    display_cols = ["alertDate", "location_en", "category_en", "category_desc"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("alertDate", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name=f"israel_alerts_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
