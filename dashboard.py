"""
Israel Civil Defense Alerts — Live Dashboard
Run with:  streamlit run dashboard.py
"""

import os
import threading
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from charts import (
    category_breakdown_chart,
    comparison_convergence_chart,
    comparison_summary_chart,
    convergence_rate_chart,
    daily_pre_alert_siren_chart,
    high_risk_windows_chart,
    hourly_heatmap,
    interactive_risk_windows_chart,
    monthly_trend_chart,
    prediction_distribution_chart,
    risk_correlation_chart,
    risk_sensitivity_chart,
    siren_heatmap_chart,
    timeline_chart,
    top_locations_chart,
)
from data_loader import (
    load_dashboard_df,
    stream_download,
)
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
    high_risk_windows_data,
    hourly_heatmap_data,
    kpi_delta,
    kpi_summary,
    monthly_counts,
    predict_siren_probability,
    predict_time_to_siren_now,
    all_areas_risk_summary,
    siren_counts_by_location,
    top_locations,
    train_area_model,
    train_siren_classifier,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Israel Alerts Dashboard",
    page_icon="🚨",
    initial_sidebar_state="expanded",
)

# ── Deployment mode ───────────────────────────────────────────────────────────
# Set LOCAL_MODE=true in your shell (or .env) for unrestricted local dev.
# On Streamlit Community Cloud the variable is absent → cloud mode enforced.
IS_LOCAL: bool = os.getenv("LOCAL_MODE", "false").lower() == "true"
COOLDOWN_MINUTES: int = 60   # minimum minutes between manual refreshes / trains
AUTO_REFRESH_HOURS: int = 4  # cloud-mode automatic data refresh interval
# Default area shown on first load.
# Override locally via DEFAULT_AREA="Tel Aviv" in .env (gitignored).
# Cloud deployments fall back to "Tel Aviv" when the variable is absent.
DEFAULT_AREA: str = os.getenv("DEFAULT_AREA", "Tel Aviv")

# ── Process-wide shared state ─────────────────────────────────────────────────
# IMPORTANT: plain module-level variables are RESET on every Streamlit script
# re-run.  @st.cache_resource stores data in Streamlit's own cache manager,
# which survives re-runs and is shared across all user sessions.
# Each dict holds a threading.Lock (for atomicity) plus mutable payload.

@st.cache_resource
def _make_download_state() -> dict:
    """One-time init of download coordination state."""
    return {"lock": threading.Lock(), "last_time": None}  # last_time: datetime|None

@st.cache_resource
def _make_training_state() -> dict:
    """One-time init of training coordination state."""
    return {"lock": threading.Lock(), "timestamps": {}}   # timestamps: {area: datetime}

_dl_state = _make_download_state()
_tr_state = _make_training_state()


# ── Shared data loader (cached across ALL users, TTL = 1 hour) ───────────────
@st.cache_data(ttl=COOLDOWN_MINUTES * 60, show_spinner=False)
def _cached_download() -> pd.DataFrame:
    """
    Download, parse and label the CSV.  Runs at most once per hour across all
    connected users.  Streamlit serialises concurrent callers automatically —
    only the first caller downloads; the rest receive the cached DataFrame.
    """
    csv_bytes = stream_download(lambda _: None)
    df = load_dashboard_df(csv_bytes)
    df = apply_english_labels(df)
    df = apply_english_locations(df)
    return df


# ── Cached compare-tab computations ──────────────────────────────────────────
# Keyed by (df content, area) — once computed for a user, all other users
# sharing the same data get the result instantly from cache.
@st.cache_data(show_spinner=False)
def _cached_area_timings(df: pd.DataFrame, area: str) -> dict:
    # clear_window_minutes=60: all-clears arrive after siren + shelter period,
    # so a 15-min window (same as siren pairing) was too tight and understated the average.
    return area_timings(df, area, window_minutes=15, clear_window_minutes=60)

@st.cache_data(show_spinner=False)
def _cached_convergence_rate(df: pd.DataFrame, area: str) -> pd.DataFrame:
    return convergence_rate_over_time(df, area, window_minutes=15)

@st.cache_data(show_spinner=False)
def _cached_all_areas_risk_summary(df: pd.DataFrame) -> "pd.DataFrame":
    return all_areas_risk_summary(df, window_minutes=15)

@st.cache_data(show_spinner=False)
def _cached_top_locations(df: pd.DataFrame, n: int) -> "pd.DataFrame":
    return top_locations(df, n=n)

# ── Session state init ────────────────────────────────────────────────────────
for _key, _default in [
    ("df", None),
    ("loaded_at", None),
    ("model_state", None),
    ("classifier_state", None),
    ("alert_active", False),
    ("alert_started_at", None),
    ("last_prediction", None),
    ("last_p_siren", None),
    ("active_area", None),
    ("area_model_cache", {}),
    ("recent_areas", []),
    ("pending_area", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Apply pending area selection BEFORE any widget is instantiated ────────────
if st.session_state["pending_area"] is not None:
    st.session_state["area_select"] = st.session_state["pending_area"]
    st.session_state["pending_area"] = None

# ── Auto-refresh data every AUTO_REFRESH_HOURS (cloud mode only) ──────────────
# Fires on the first user interaction after the interval has elapsed.
# After refresh, existing per-area models are automatically marked stale
# (their cached loaded_at won't match the new session loaded_at) so the
# Train button lights up, prompting users to retrain if they want fresh models.
if not IS_LOCAL and st.session_state["df"] is not None:
    with _dl_state["lock"]:
        _auto_last = _dl_state["last_time"]
        _auto_age_h = (
            (datetime.now() - _auto_last).total_seconds() / 3600
            if _auto_last else None
        )
        _do_auto = _auto_age_h is not None and _auto_age_h >= AUTO_REFRESH_HOURS
        if _do_auto:
            _dl_state["last_time"] = datetime.now()  # optimistic stamp under lock

    if _do_auto:
        try:
            _cached_download.clear()
            _auto_fresh = _cached_download()
            st.session_state["df"] = _auto_fresh
            st.session_state["loaded_at"] = _dl_state["last_time"]
            st.toast("🔄 Data refreshed automatically (4-hour schedule)", icon="🔄")
            st.rerun()
        except Exception:
            pass  # silent — never disrupt the user on a background refresh failure

# ── Auto-populate new sessions from shared cache ──────────────────────────────
# When a user opens a new tab or hits F5, their session_state["df"] is None
# but _cached_download() may already hold fresh data in Streamlit's cache.
# Load it immediately so they never see "No data" when data exists.
if st.session_state["df"] is None:
    try:
        with st.spinner("Loading data…"):
            _session_df = _cached_download()
        st.session_state["df"] = _session_df
        if _dl_state["last_time"] is None:
            _dl_state["last_time"] = datetime.now()
        st.session_state["loaded_at"] = _dl_state["last_time"]
        st.rerun()
    except Exception:
        pass  # No cached data yet — fall through to show the manual load button

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")

    # ── Load / Refresh ──────────────────────────────────────────────────────
    # Thread-safe read of last download time
    with _dl_state["lock"]:
        _dl_last = _dl_state["last_time"]
    _dl_age_s = (datetime.now() - _dl_last).total_seconds() if _dl_last else None
    # Always allow refresh when this session has no data (new tab / F5)
    _no_data = st.session_state["df"] is None
    _can_refresh = _no_data or IS_LOCAL or _dl_age_s is None or _dl_age_s >= COOLDOWN_MINUTES * 60
    _refresh_help = None
    if not _can_refresh:
        _mins_left = int(COOLDOWN_MINUTES - _dl_age_s // 60)
        _refresh_help = f"⏳ Next refresh available in ~{_mins_left} min."

    if st.button(
        "⬇ Load / Refresh Data",
        use_container_width=True,
        type="primary",
        disabled=not _can_refresh,
        help=_refresh_help,
    ):
        # Atomic check-and-set under lock — prevents concurrent double-download
        with _dl_state["lock"]:
            _dl_last2 = _dl_state["last_time"]
            _dl_age_s2 = (datetime.now() - _dl_last2).total_seconds() if _dl_last2 else None
            _proceed = IS_LOCAL or _dl_age_s2 is None or _dl_age_s2 >= COOLDOWN_MINUTES * 60
            if _proceed:
                _dl_state["last_time"] = datetime.now()  # optimistic stamp

        if _proceed:
            try:
                _cached_download.clear()
                with st.spinner("Downloading fresh data…"):
                    _fresh = _cached_download()
                st.session_state["df"] = _fresh
                st.session_state["loaded_at"] = _dl_state["last_time"]
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load data: {exc}")
        else:
            st.warning("Another refresh just started — please wait a moment.")

    # ── Data freshness ──────────────────────────────────────────────────────
    _loaded_at = st.session_state.get("loaded_at") or _dl_state["last_time"]
    if _loaded_at:
        _age_h = (datetime.now() - _loaded_at).total_seconds() / 3600
        _color = "🟢" if _age_h < 24 else ("🟡" if _age_h < 72 else "🔴")
        _age_str = f"{_age_h:.0f}h ago" if _age_h < 24 else f"{_age_h / 24:.0f}d ago"
        st.caption(
            f"{_color} Data: **{_loaded_at.strftime('%Y-%m-%d %H:%M')}** ({_age_str})"
        )
        if not IS_LOCAL and _dl_age_s is not None and _dl_age_s < COOLDOWN_MINUTES * 60:
            _mins_left2 = int(COOLDOWN_MINUTES - _dl_age_s // 60)
            st.caption(f"Next refresh in ~{_mins_left2} min")

    st.divider()

    df_full: pd.DataFrame | None = st.session_state["df"]

    if df_full is not None:
        # ── Last X days slider ──────────────────────────────────────────────
        st.subheader("Time Range")
        max_date = df_full["parsed_date"].max().date()
        min_date = df_full["parsed_date"].min().date()
        total_days = (max_date - min_date).days

        last_x_days = st.slider(
            "Show last N days",
            min_value=7,
            max_value=90,
            value=30,
            step=1,
            key="days_slider_v2",
        )
        start_date = max(min_date, max_date - timedelta(days=last_x_days))
        end_date = max_date

        st.divider()

        # ── Area filter ─────────────────────────────────────────────────────
        st.subheader("Area Filter")
        all_locations = get_all_locations(df_full)
        area_options = ["(All areas)"] + all_locations
        # Priority: 1) user already picked something (session_state, from a
        #   previous interaction this session), 2) ?area= URL param (bookmark /
        #   shared link), 3) DEFAULT_AREA env var, 4) first option "(All areas)".
        # Note: index= only applies on the very first render — once the widget
        # is in session_state Streamlit ignores it, which is the correct behaviour.
        _area_pref = st.query_params.get("area", DEFAULT_AREA)
        default_idx = (
            area_options.index(_area_pref) if _area_pref in area_options else 0
        )
        selected_area = st.selectbox(
            "Focus on area",
            options=area_options,
            index=default_idx,
            key="area_select",
        )

        # ── Recent areas quick-select ────────────────────────────────────────
        recents = [a for a in st.session_state["recent_areas"] if a != selected_area]
        if recents:
            st.caption("Recent areas:")
            for _r in recents:
                _has_model = _r in st.session_state["area_model_cache"]
                _btn_label = f"{'🧠 ' if _has_model else ''}{_r}"
                if st.button(_btn_label, key=f"recent_{_r}", use_container_width=True):
                    st.session_state["pending_area"] = _r
                    st.rerun()

        st.divider()

        # ── Model training (sidebar) ─────────────────────────────────────────
        st.subheader("Prediction Model")
        area_active_sidebar = selected_area != "(All areas)"

        ms = st.session_state["model_state"]
        cs = st.session_state["classifier_state"]
        if ms:
            trained_for = ms["location"]
            stale = trained_for != selected_area
            icon = "⚠️" if stale else "✅"
            st.caption(
                f"{icon} Regression: **{trained_for}**  \n"
                f"Trained {ms['trained_at']} · {ms['n_samples']} pairs · "
                f"avg {ms['historical_avg_min']:.1f} min"
            )
            if cs and cs.get("location") == trained_for:
                clf_icon = "⚠️" if stale else "✅"
                st.caption(
                    f"{clf_icon} Classifier: base rate "
                    f"**{cs['base_rate'] * 100:.0f}%** · {cs['n_samples']} pre-alerts"
                )
            if stale:
                st.caption("Area changed — retrain for accurate predictions.")

        train_label = (
            f"🧠 Train models for {selected_area}"
            if area_active_sidebar
            else "🧠 Train models  *(select an area first)*"
        )
        # Light up as primary when the area needs training.
        _cache_entry = st.session_state["area_model_cache"].get(selected_area)
        _needs_training = area_active_sidebar and (
            _cache_entry is None
            or _cache_entry.get("loaded_at") != st.session_state.get("loaded_at")
        )
        # Cooldown: in cloud mode, block re-training within COOLDOWN_MINUTES.
        with _tr_state["lock"]:
            _train_last = _tr_state["timestamps"].get(selected_area)
            _train_age_s = (
                (datetime.now() - _train_last).total_seconds() if _train_last else None
            )
        # Always allow training when this session has no model yet (new tab / F5)
        # — same guard as the data-refresh button to avoid the stuck-state bug.
        _session_has_model = st.session_state["model_state"] is not None
        _can_train = area_active_sidebar and (
            not _session_has_model
            or IS_LOCAL
            or _train_age_s is None
            or _train_age_s >= COOLDOWN_MINUTES * 60
        )
        _train_help = None
        if area_active_sidebar and not _can_train:
            _train_mins_left = int(COOLDOWN_MINUTES - _train_age_s // 60)
            _train_help = f"⏳ Models were just trained. Next training in ~{_train_mins_left} min."

        if st.button(
            train_label,
            use_container_width=True,
            disabled=not _can_train,
            type="primary" if (_needs_training and _can_train) else "secondary",
            help=_train_help,
        ):
            df_history_train = filter_categories(
                df_full, include_drills=False, selected_category_ids=None
            )
            df_history_area_train = filter_by_location(df_history_train, selected_area)
            _pbar = st.progress(0.0, text="Starting…")

            def _reg_progress(fraction: float, text: str) -> None:
                _pbar.progress(fraction * 0.5, text=f"Regression: {text}")

            def _clf_progress(fraction: float, text: str) -> None:
                _pbar.progress(0.5 + fraction * 0.5, text=f"Classifier: {text}")

            model_state, err = train_area_model(
                df_history_area_train, selected_area, progress_cb=_reg_progress
            )
            classifier_state, clf_err = train_siren_classifier(
                df_history_area_train, selected_area, progress_cb=_clf_progress
            )
            _pbar.empty()
            if err:
                st.warning(err)
            else:
                st.session_state["model_state"] = model_state
                if clf_err:
                    st.warning(clf_err)
                else:
                    st.session_state["classifier_state"] = classifier_state
                # Persist both models in the per-area cache
                st.session_state["area_model_cache"][selected_area] = {
                    "model_state": model_state,
                    "classifier_state": None if clf_err else classifier_state,
                    "loaded_at": st.session_state.get("loaded_at"),
                }
                # Stamp training time (thread-safe) to enforce cloud cooldown
                with _tr_state["lock"]:
                    _tr_state["timestamps"][selected_area] = datetime.now()
                st.success(
                    f"Models ready · {model_state['n_samples']} pairs · "
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

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🚨 Israel Civil Defense Alerts Dashboard")
_updated_str = ""
if st.session_state.get("loaded_at"):
    _updated_str = f" · Data updated: **{st.session_state['loaded_at'].strftime('%Y-%m-%d %H:%M')}**"
st.caption(
    "Data source: [github.com/dleshem/israel-alerts-data]"
    f"(https://github.com/dleshem/israel-alerts-data){_updated_str}"
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

# ── Handle area change: stop event, update recents, restore cached model ──────
if st.session_state["active_area"] != selected_area:
    # Keep URL in sync so the user can bookmark / share their current area.
    if selected_area != "(All areas)":
        st.query_params["area"] = selected_area
    elif "area" in st.query_params:
        del st.query_params["area"]

    prev = st.session_state["active_area"]
    # Push previous area into recent list (skip None / all-areas placeholder)
    if prev and prev != "(All areas)":
        recents = [prev] + [
            a for a in st.session_state["recent_areas"]
            if a != prev and a != selected_area
        ]
        st.session_state["recent_areas"] = recents[:3]
    st.session_state["active_area"] = selected_area
    # Stop any running alert
    if st.session_state["alert_active"]:
        st.session_state["alert_active"] = False
        st.session_state["alert_started_at"] = None
    # Restore cached models if data hasn't changed since they were trained
    cache = st.session_state["area_model_cache"]
    if selected_area in cache:
        entry = cache[selected_area]
        if entry.get("loaded_at") == st.session_state.get("loaded_at"):
            st.session_state["model_state"] = entry["model_state"]
            st.session_state["classifier_state"] = entry["classifier_state"]
        else:
            # Data was refreshed — cached model is stale, clear it
            st.session_state["model_state"] = None
            st.session_state["classifier_state"] = None
    else:
        # No model cached for this area yet — clear any leftover from previous area
        st.session_state["model_state"] = None
        st.session_state["classifier_state"] = None
    # Rerun so the sidebar re-renders immediately with the correct model state.
    # Safe: active_area is already updated above so this block won't fire again.
    st.rerun()

df = filter_by_date_range(df_full, start_date, end_date)
df = filter_categories(df, include_drills=include_drills, selected_category_ids=selected_ids)

df_area = filter_by_location(df, selected_area) if area_active else df

df_history = filter_categories(df_full, include_drills=include_drills, selected_category_ids=selected_ids)
df_history_area = filter_by_location(df_history, selected_area) if area_active else df_history

if df.empty:
    st.warning("No events match the current filters. Try a wider time range or more alert types.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_area, tab_overview, tab_compare = st.tabs(["📍 Area Analysis", "📊 Overview", "⚖️ Compare Areas"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Area Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_area:

    if not area_active:
        with st.container(border=True):
            st.markdown("### 👋 Getting started")
            st.markdown(
                "**Step 1 →** Pick an area from the **Area** drop-down in the sidebar.  \n"
                "**Step 2 →** Click **🧠 Train models** in the sidebar to enable timing "
                "predictions and the live simulation timer.  \n"
                "**Step 3 →** Press **🚨 Simulate a pre-alert event now** to start a "
                "colour-coded elapsed-time display based on that area's historical statistics."
            )
        st.subheader("Daily Pre-Alerts vs Sirens (all areas)")
        daily_ps = daily_pre_alert_siren_counts(df)
        st.plotly_chart(daily_pre_alert_siren_chart(daily_ps), width="stretch", key="daily_ps_all")

    else:
        st.subheader(f"Area: {selected_area}")

        if df_area.empty:
            st.warning(f"No events found for **{selected_area}** in the selected time range.")
        else:
            # ── Model state ─────────────────────────────────────────────────
            ms = st.session_state["model_state"]
            cs = st.session_state["classifier_state"]
            model_ready = ms is not None and ms.get("location") == selected_area
            classifier_ready = cs is not None and cs.get("location") == selected_area
            alert_active_now = st.session_state["alert_active"]

            # ── Area summary card / onboarding prompt ───────────────────────
            # Compute stats from the date-filtered area data so the summary
            # always reflects the currently selected time range.
            _n_days = max(1, (end_date - start_date).days + 1)
            _sum_t   = _cached_area_timings(df_area, selected_area)
            _sum_n_pa   = _sum_t.get("n_pre_alerts", 0)
            _sum_n_sir  = _sum_t.get("n_sirens", 0)
            _sum_conv   = _sum_t.get("convergence_rate")      # may be None
            _sum_avg    = _sum_t.get("avg_pre_to_siren_min")  # may be None

            if model_ready and not alert_active_now:
                # ── Build readable convergence sentence ─────────────────────
                if _sum_conv is not None and _sum_conv > 0 and _sum_n_pa > 0:
                    _ratio = round(1 / _sum_conv)
                    _conv_sentence = (
                        f"About **1 in every {_ratio}** pre-alerts led to a siren "
                        f"within 15 minutes (**{_sum_conv * 100:.0f}% convergence rate**)"
                    )
                    if _sum_avg is not None:
                        _conv_sentence += f", with an average lead time of **{_sum_avg:.1f} min**."
                    else:
                        _conv_sentence += "."
                elif _sum_conv == 0.0:
                    _conv_sentence = "None of the pre-alerts in this period led to a siren within 15 minutes."
                else:
                    _conv_sentence = "No pre-alerts recorded in this period."

                # ── Cross-area context (same date window as the card) ───────
                # Using df (date-filtered) so daily rates are comparable.
                _pa_per_day  = _sum_n_pa  / _n_days
                _sir_per_day = _sum_n_sir / _n_days

                _all_smry = _cached_all_areas_risk_summary(df)
                if not _all_smry.empty and _all_smry["n_days"].iloc[0] > 0:
                    _smry_days = _all_smry["n_days"].iloc[0]
                    _top_act  = _all_smry.loc[_all_smry["n_pre_alerts"].idxmax()]
                    # Highest convergence: only areas with ≥20 pre-alerts (avoid flukes)
                    _conv_pool = _all_smry[_all_smry["n_pre_alerts"] >= 20]
                    _top_conv  = (
                        _conv_pool.loc[_conv_pool["convergence_rate"].idxmax()]
                        if not _conv_pool.empty
                        else _all_smry.loc[_all_smry["convergence_rate"].idxmax()]
                    )
                    _ta_name = str(_top_act["area"])
                    _ta_rate = float(_top_act["n_pre_alerts"]) / _smry_days
                    _tc_name = str(_top_conv["area"])
                    _tc_rate = float(_top_conv["convergence_rate"]) * 100
                    _ref_caption = (
                        f"Most active: **{_ta_name}** ({_ta_rate:.1f}/day)  \n"
                        f"Highest conv ⚠️: **{_tc_name}** ({_tc_rate:.0f}%)"
                    )
                else:
                    _ref_caption = None

                with st.container(border=True):
                    _col_txt, _col_rates = st.columns([3, 2])
                    with _col_txt:
                        st.markdown(
                            f"**{selected_area}** — last **{_n_days} days**  \n"
                            f"📢 **{_sum_n_pa:,}** pre-alerts &nbsp;·&nbsp; "
                            f"🚨 **{_sum_n_sir:,}** sirens  \n"
                            + _conv_sentence
                        )
                    with _col_rates:
                        def _fmt_rate(r):
                            return f"~{round(r)}" if round(r) >= 1 else "< 1"
                        def _tip(icon, name, r, total, days):
                            display = _fmt_rate(r)
                            tip = f"Exact: {r:.2f} / day ({total} total over {days} days)"
                            return (
                                f'<span title="{tip}" style="cursor:help;font-size:0.9em;">'
                                f"{icon} {name} <b>{display}</b> / day</span>"
                            )
                        st.markdown(
                            _tip("📢", "Pre-alerts", _pa_per_day, _sum_n_pa, _n_days) + "<br>" +
                            _tip("🚨", "Sirens", _sir_per_day, _sum_n_sir, _n_days),
                            unsafe_allow_html=True,
                        )
                        if _ref_caption:
                            st.caption(_ref_caption)
            elif not model_ready:
                # ── Onboarding prompt (model not yet trained) ───────────────
                st.info(
                    f"📊 **{selected_area}** is selected.  \n"
                    "Click **🧠 Train models** in the sidebar to enable timing "
                    "predictions, the live simulation timer, and the area risk summary."
                )

            if not alert_active_now:
                if st.button(
                    "🚨 Simulate a pre-alert event now",
                    type="primary",
                    use_container_width=True,
                    help=(
                        "⚠️ This is a visual simulation only — it is not connected to any "
                        "real alert system. "
                        "Starts a live elapsed-time timer from this moment. "
                        "The timer background colour changes based on the historical "
                        "pre-alert → siren statistics for this area: "
                        "green = well within the expected window, yellow = approaching, "
                        "orange = near the predicted time, red = past predicted time. "
                        "The regression model predicts minutes until a siren is expected "
                        "(based on time of day and day of week); "
                        "the classifier estimates the probability a siren will follow. "
                        "The timer auto-ends after 2× the match window has elapsed. "
                        "Click ✅ to stop it manually at any time."
                    ),
                ):
                    if not model_ready:
                        st.warning(
                            "No trained model for this area yet. "
                            "Click **🧠 Train models** in the sidebar first."
                        )
                    else:
                        predicted = predict_time_to_siren_now(ms)
                        p_siren = predict_siren_probability(cs) if classifier_ready else None
                        st.session_state["last_prediction"] = predicted
                        st.session_state["last_p_siren"] = p_siren
                        st.session_state["alert_active"] = True
                        st.session_state["alert_started_at"] = datetime.now()
                        st.rerun()
                if not model_ready:
                    st.caption("⚠️ Train the models in the sidebar for predictions.")
                else:
                    _wm       = ms.get("window_minutes", 15)
                    _hist_avg = ms.get("historical_avg_min") or _wm
                    _auto_min = _hist_avg + 10
                    st.caption(
                        f"Visual timer only — not a real alert system. "
                        f"Auto-ends {_auto_min:.0f} min from start "
                        f"(mean {_hist_avg:.1f} min + 10 min buffer)."
                    )
            else:
                # ── Auto-end at historical mean + 10 min ────────────────────
                started_at = st.session_state["alert_started_at"]
                _wm       = ms.get("window_minutes", 15) if ms else 15
                _hist_avg = (ms.get("historical_avg_min") or _wm) if ms else _wm
                _auto_end_s = int((_hist_avg + 10) * 60)
                if started_at is not None:
                    _elapsed_s = (datetime.now() - started_at).total_seconds()
                    if _elapsed_s >= _auto_end_s:
                        st.session_state["alert_active"] = False
                        st.session_state["alert_started_at"] = None
                        st.rerun()

                if st.button("✅ Click to mark the end of the event", type="secondary", use_container_width=True):
                    st.session_state["alert_active"] = False
                    st.session_state["alert_started_at"] = None
                    st.rerun()

                started_at = st.session_state["alert_started_at"]
                predicted = st.session_state.get("last_prediction", 5.0)
                p_siren = st.session_state.get("last_p_siren")

                # Live elapsed timer (JS-driven, updates every 500 ms in browser).
                # Colour reflects position relative to the predicted siren time.
                # After 2× the match window the display greys out to signal auto-end.
                if started_at is not None:
                    start_ms = int(started_at.timestamp() * 1000)
                    pred_ms  = int(predicted * 60 * 1000)
                    max_ms   = int(_auto_end_s * 1000)
                    p_siren_html = ""
                    if p_siren is not None:
                        pct = f"{p_siren * 100:.0f}%"
                        clr = "#e63946" if p_siren > 0.7 else "#f4a261" if p_siren > 0.4 else "#2dc653"
                        p_siren_html = (
                            f" &nbsp;·&nbsp; P(siren): "
                            f'<span style="color:{clr};font-weight:bold;font-size:1.1em">{pct}</span>'
                        )
                    # Zone-boundary positions as % of bar width
                    def _bar_pct(ms_val):
                        return round(min(98.5, ms_val / max_ms * 100), 1)
                    _p50  = _bar_pct(pred_ms * 0.50)   # green → yellow
                    _p80  = _bar_pct(pred_ms * 0.80)   # yellow → orange
                    _p100 = _bar_pct(pred_ms * 1.00)   # orange → red  (predicted)
                    _p135 = _bar_pct(pred_ms * 1.35)   # red → dark-red
                    _auto_min_label = f"{_auto_end_s / 60:.0f}"
                    timer_html = f"""
                    <div style="font-family:monospace;text-align:center;padding:16px 20px;
                                border-radius:12px;border:1px solid #444;background:#0e1117;">

                      <!-- status label -->
                      <div id="tmr-label" style="font-size:0.85em;color:#aaa;margin-bottom:10px;
                                  text-transform:uppercase;letter-spacing:2px;">
                        Time since pre-alert{p_siren_html}
                      </div>

                      <!-- clock -->
                      <div id="tmr" style="font-size:4.2em;font-weight:bold;padding:8px 28px;
                           border-radius:10px;display:inline-block;min-width:180px;
                           transition:background-color 0.5s ease;color:#fff;">0:00</div>

                      <!-- progress bar -->
                      <div style="position:relative;width:100%;height:22px;background:#1a1d27;
                                  border-radius:11px;overflow:visible;margin-top:16px;">
                        <!-- coloured fill -->
                        <div id="prog" style="position:absolute;top:0;left:0;height:100%;width:0%;
                             background:#2dc653;border-radius:11px;
                             transition:width 0.5s linear,background-color 0.5s ease;"></div>
                        <!-- green→yellow boundary -->
                        <div title="50% of predicted — entering caution zone"
                             style="position:absolute;top:-3px;left:{_p50}%;
                                    width:2px;height:28px;background:#ffd166;
                                    opacity:0.5;border-radius:1px;"></div>
                        <!-- yellow→orange boundary -->
                        <div title="80% of predicted — approaching"
                             style="position:absolute;top:-3px;left:{_p80}%;
                                    width:2px;height:28px;background:#f4a261;
                                    opacity:0.65;border-radius:1px;"></div>
                        <!-- predicted time — glowing anchor -->
                        <div title="Predicted siren time"
                             style="position:absolute;top:-5px;left:{_p100}%;
                                    width:3px;height:32px;background:#ffd166;
                                    border-radius:2px;box-shadow:0 0 7px #ffd166;"></div>
                        <!-- red→dark-red boundary (only if it fits) -->
                        {"" if _p135 >= 98 else f'''
                        <div title="135% of predicted — well past expected"
                             style="position:absolute;top:-3px;left:{_p135}%;
                                    width:2px;height:28px;background:#e63946;
                                    opacity:0.6;border-radius:1px;"></div>'''}
                        <!-- auto-stop line -->
                        <div title="Auto-stop"
                             style="position:absolute;top:-5px;right:1px;
                                    width:3px;height:32px;background:#888;
                                    border-radius:2px;border-left:2px dashed #555;"></div>
                      </div>

                      <!-- bar axis labels -->
                      <div style="position:relative;width:100%;margin-top:5px;
                                  font-size:0.72em;color:#666;">
                        <span style="float:left;">0</span>
                        <span style="position:absolute;left:{_p100}%;
                                     transform:translateX(-50%);color:#ffd166;white-space:nowrap;">
                          ▲ {predicted:.1f} min
                        </span>
                        <span style="float:right;color:#777;">⏹ {_auto_min_label} min</span>
                      </div>

                    </div>
                    <script>
                      (function(){{
                        var startMs={start_ms}, predMs={pred_ms}, maxMs={max_ms};
                        var el  = document.getElementById('tmr');
                        var lbl = document.getElementById('tmr-label');
                        var prg = document.getElementById('prog');
                        function zoneColor(e){{
                          var r = e / predMs;
                          if(r < 0.5)  return '#2dc653';
                          if(r < 0.8)  return '#ffd166';
                          if(r < 1.0)  return '#f4a261';
                          if(r < 1.35) return '#e63946';
                          return '#c0392b';
                        }}
                        function tick(){{
                          var e = Date.now() - startMs;
                          if(e >= maxMs){{
                            el.textContent = '--:--';
                            el.style.backgroundColor = '#2a2a2a';
                            if(prg){{ prg.style.width='100%'; prg.style.background='#444'; }}
                            if(lbl) lbl.textContent = 'AUTO-STOPPED \u2014 interact with the page to dismiss';
                            return;
                          }}
                          var s   = Math.floor(e / 1000);
                          var col = zoneColor(e);
                          el.textContent = Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
                          el.style.backgroundColor = col;
                          if(prg){{ prg.style.width=(e/maxMs*100)+'%'; prg.style.background=col; }}
                        }}
                        tick(); setInterval(tick, 500);
                      }})();
                    </script>
                    """
                    components.html(timer_html, height=240)

                if model_ready and predicted is not None:
                    st.error(
                        f"### ⚠️ Pre-Alert Reported — {selected_area}\n\n"
                        f"**Estimated time to siren: {predicted:.1f} min**  "
                        f"· CI [{ms['historical_min_min']:.1f} – {ms['historical_max_min']:.1f}] min  \n"
                        f"Mean {ms['historical_avg_min']:.1f} ± {ms['historical_std_min']:.1f} min  "
                        f"· Median {ms['historical_median_min']:.1f} min  "
                        f"· {ms['n_samples']} training events"
                    )

            # ── Distribution chart — always visible when model is trained ───
            if model_ready:
                # During an active event use the live prediction; otherwise
                # fall back to the historical mean so the chart is always useful.
                _chart_predicted = (
                    st.session_state.get("last_prediction") or ms["historical_avg_min"]
                    if alert_active_now
                    else ms["historical_avg_min"]
                )
                st.plotly_chart(
                    prediction_distribution_chart(ms, _chart_predicted),
                    width="stretch",
                    key="pred_dist",
                )
                if not alert_active_now:
                    st.caption(
                        "Historical pre-alert → siren time distribution for this area.  "
                        "Red line = historical mean. Press **🚨 Simulate a pre-alert event now** for a "
                        "time-adjusted prediction."
                    )

            st.divider()

            # ── KPI row for area ────────────────────────────────────────────
            kpi = kpi_summary(df_area)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Total Events", f"{kpi['total']:,}",
                help="All alert events in this area for the selected time range.",
            )
            c2.metric(
                "Sirens", f"{kpi['sirens']:,}",
                help="Rocket/missile fire, hostile aircraft, infiltration, and other active-threat sirens.",
            )
            c3.metric(
                "Pre-Alerts", f"{kpi['pre_alerts']:,}",
                help="Early-warning notifications (category 14) issued before a potential siren.",
            )
            c4.metric(
                "All-Clear Sent", f"{kpi['all_clear']:,}",
                help="All-clear / relax notifications (category 13) indicating the threat has passed.",
            )

            st.divider()

            # ── Timing analytics ────────────────────────────────────────────
            st.subheader("Pre-Alert Timing Analysis")
            with st.spinner("Computing timing statistics…"):
                timings = area_timings(
                    df_history_area, selected_area,
                    window_minutes=15, clear_window_minutes=60,
                )

            if timings["n_pre_alerts"] == 0:
                st.info("No pre-alerts found for this area in the full dataset.")
            else:
                def _fmt_ci(ci):
                    lo, hi = ci
                    if lo is None:
                        return "n < 3, CI not available"
                    return f"95% CI: {lo:.1f} – {hi:.1f} min  (bootstrap, non-parametric)"

                t1, t2, t3, t4 = st.columns(4)

                avg_ps = timings["avg_pre_to_siren_min"]
                t1.metric(
                    "Pre-Alert → Siren",
                    f"{avg_ps:.1f} min" if avg_ps is not None else "N/A",
                    help=(
                        "Mean minutes from a pre-alert to the next siren "
                        f"(within {15}-min window, {timings['n_converged']} pairs).  \n"
                        + _fmt_ci(timings["ci_pre_to_siren_min"])
                    ),
                )

                avg_pc = timings["avg_pre_to_clear_min"]
                t2.metric(
                    "Pre-Alert → Release",
                    f"{avg_pc:.1f} min" if avg_pc is not None else "N/A",
                    help=(
                        "Mean minutes from a pre-alert to the next all-clear / release "
                        "(within 60-min window).  \n"
                        + _fmt_ci(timings["ci_pre_to_clear_min"])
                    ),
                )

                avg_sc = timings["avg_siren_to_clear_min"]
                t3.metric(
                    "Siren → Release",
                    f"{avg_sc:.1f} min" if avg_sc is not None else "N/A",
                    help=(
                        "Mean minutes from a siren to the next all-clear / release — "
                        f"the actual shelter duration ({timings['n_siren_to_clear']} pairs).  \n"
                        + _fmt_ci(timings["ci_siren_to_clear_min"])
                    ),
                )

                rate = timings["convergence_rate"]
                t4.metric(
                    "Convergence Rate",
                    f"{rate * 100:.1f}%" if rate is not None else "N/A",
                    help=(
                        f"{timings['n_converged']} of {timings['n_pre_alerts']} pre-alerts "
                        "were followed by an actual siren within 15 minutes."
                    ),
                )

                st.caption(
                    f"Based on **{timings['n_pre_alerts']}** pre-alerts and "
                    f"**{timings['n_sirens']}** sirens in **{selected_area}** (full dataset).  "
                    "Hover each metric for 95% bootstrap CI (distribution-free)."
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

            # ── Convergence rate over time (respects N-days slider) ──────────
            st.subheader("Convergence Rate Over Time")
            conv_df = convergence_rate_over_time(df_area, selected_area, window_minutes=15)
            st.plotly_chart(
                convergence_rate_chart(conv_df, selected_area),
                width="stretch",
                key="conv_rate_chart",
            )
            st.caption(
                "% of pre-alerts followed by an actual siren within 15 minutes.  "
                "Red line = 7-day rolling average. Affected by the time-range slider."
            )

            st.divider()

            # ── Interactive risk windows ─────────────────────────────────────
            st.subheader("Siren Events by Time-of-Day Window")
            st.caption("Use the dropdown inside the chart to switch between 2 h, 4 h, and 6 h buckets.")
            st.plotly_chart(
                interactive_risk_windows_chart(df_history_area),
                width="stretch",
                key="risk_windows_chart",
            )

            st.divider()

            # ── High-risk days & times ───────────────────────────────────────
            st.subheader("High-Risk Days & Times")
            risk_df = high_risk_windows_data(df_history_area, top_n=10)
            if risk_df.empty:
                st.info("Not enough siren data to compute high-risk windows for this area.")
            else:
                st.plotly_chart(
                    high_risk_windows_chart(risk_df, selected_area),
                    width="stretch",
                    key="high_risk_chart",
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Overview (scoped to area when active)
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # Overview always shows all areas, filtered by the selected date range
    df_view = df
    df_view_history = df  # use date-filtered df (not full history)

    # ── KPI row ─────────────────────────────────────────────────────────────
    kpi = kpi_summary(df_view)
    deltas = kpi_delta(df_full, start_date, end_date)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Total Events", f"{kpi['total']:,}", delta=int(deltas.get("total", 0)),
        help="All alert events across all areas for the selected time range. Delta = change vs the same-length period immediately before.",
    )
    col2.metric(
        "Sirens / Alerts", f"{kpi['sirens']:,}", delta=int(deltas.get("sirens", 0)),
        help="Active-threat sirens (rockets, missiles, aircraft, infiltration, etc.). Delta = change vs the previous period.",
    )
    col3.metric(
        "Pre-Alerts", f"{kpi['pre_alerts']:,}", delta=int(deltas.get("pre_alerts", 0)),
        help="Early-warning notifications (category 14) issued before a potential siren. Delta = change vs the previous period.",
    )
    col4.metric(
        "All-Clear Sent", f"{kpi['all_clear']:,}", delta=int(deltas.get("all_clear", 0)),
        help="All-clear / relax notifications (category 13) indicating the immediate threat has passed. Delta = change vs the previous period.",
    )
    col5.metric(
        "Unique Locations", f"{kpi['unique_locations']:,}",
        help="Number of distinct cities / zones that received at least one alert in the selected time range.",
    )

    st.caption(
        f"Showing **{kpi['total']:,}** events from **{start_date}** to **{end_date}** (all areas)  |  "
        f"Last 24 h: **{kpi['last_24h']:,}**  |  Last 7 days: **{kpi['last_7d']:,}**"
    )

    st.divider()

    # ── Timeline ─────────────────────────────────────────────────────────────
    weekly = daily_counts(df_view)
    st.plotly_chart(timeline_chart(weekly), width="stretch", key="timeline_overview")

    # ── Locations + category breakdown ───────────────────────────────────────
    top_n = st.slider("Show top N locations", min_value=5, max_value=50, value=20, step=5,
                      key="top_n_overview")
    left, right = st.columns([3, 2])
    with left:
        loc_df = _cached_top_locations(df_view, n=top_n)
        st.plotly_chart(
            top_locations_chart(loc_df, start_date=start_date, end_date=end_date),
            width="stretch", key="top_locations_overview",
        )
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

    # ── Siren heatmap ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📍 Siren Activity Map")
    st.caption("Density of siren events by location. Zoom and pan to explore.")
    _heatmap_counts = siren_counts_by_location(df_view_history)
    st.plotly_chart(
        siren_heatmap_chart(_heatmap_counts), use_container_width=True, key="siren_heatmap"
    )

    # ── Risk score weight validation ─────────────────────────────────────────
    with st.expander("🔬 Risk Score Weight Validation", expanded=False):
        st.markdown(
            "The area **Risk Score** shown in the Area Analysis tab combines two signals: "
            "**convergence rate** (how reliably pre-alerts lead to sirens) and "
            "**event frequency** (how active an area is), weighted 60 / 40. "
            "This section checks whether that split is reasonable."
        )
        with st.spinner("Computing per-area stats…"):
            _risk_df = _cached_all_areas_risk_summary(df_view)

        if _risk_df.empty or len(_risk_df) < 3:
            st.info("Not enough areas with pre-alert data to run the validation.")
        else:
            _r_pearson = _risk_df["freq_score"].corr(_risk_df["convergence_rate"])

            # ── Interpretation text ──────────────────────────────────────────
            if abs(_r_pearson) < 0.3:
                _corr_msg = (
                    f"**Pearson r = {_r_pearson:.2f}** — the two components are largely "
                    "**independent**. They capture different dimensions of risk, so the "
                    "weight split matters: shifting it significantly would re-order areas."
                )
            elif abs(_r_pearson) < 0.6:
                _corr_msg = (
                    f"**Pearson r = {_r_pearson:.2f}** — moderate correlation. "
                    "The two components share some information but are not redundant. "
                    "The 60/40 split is a reasonable blend."
                )
            else:
                _corr_msg = (
                    f"**Pearson r = {_r_pearson:.2f}** — the two components are "
                    "**strongly correlated**: areas with many events also tend to have "
                    "higher convergence. The exact weight split matters less in this case."
                )
            st.info(_corr_msg)

            # ── Stability conclusion from Spearman sweep ─────────────────────
            # Spearman ρ = Pearson r of ranks — no scipy needed.
            _baseline_w = 0.60
            _bs = _baseline_w * _risk_df["convergence_rate"] + (1 - _baseline_w) * _risk_df["freq_score"]
            _bs_ranks = _bs.rank()
            _rho_0   = _bs_ranks.corr(_risk_df["freq_score"].rank())        # w=0
            _rho_100 = _bs_ranks.corr(_risk_df["convergence_rate"].rank())  # w=1
            _min_rho = min(_rho_0, _rho_100)

            if _min_rho >= 0.90:
                _stab_msg = (
                    f"Rank stability is **high** (ρ ≥ {_min_rho:.2f} even at the extremes). "
                    "Area rankings barely change across the full weight sweep — the 60/40 "
                    "choice is well-justified and can be kept as-is."
                )
                _stab_color = "success"
            elif _min_rho >= 0.75:
                _stab_msg = (
                    f"Rank stability is **moderate** (ρ ≥ {_min_rho:.2f}). "
                    "Most area rankings are preserved, but a few areas shift meaningfully "
                    "near the extremes. The 60/40 split is reasonable."
                )
                _stab_color = "warning"
            else:
                _stab_msg = (
                    f"Rank stability is **low** (ρ drops to {_min_rho:.2f}). "
                    "Area rankings change significantly with different weights. "
                    "Consider domain expertise or regression-derived weights."
                )
                _stab_color = "error"

            getattr(st, _stab_color)(_stab_msg)

            # ── Charts side by side ──────────────────────────────────────────
            _vc_left, _vc_right = st.columns(2)
            with _vc_left:
                st.plotly_chart(
                    risk_correlation_chart(_risk_df),
                    use_container_width=True,
                    key="risk_corr_chart",
                )
                st.caption(
                    "Each bubble is one area. Size = number of pre-alerts. "
                    "Colour = risk score (red = high). "
                    "Pearson r measures how correlated the two risk components are."
                )
            with _vc_right:
                st.plotly_chart(
                    risk_sensitivity_chart(_risk_df),
                    use_container_width=True,
                    key="risk_sens_chart",
                )
                st.caption(
                    "Spearman ρ vs the 60/40 baseline as the convergence-rate weight "
                    "sweeps from 0 % to 100 %. "
                    "Green band = stable zone (ρ ≥ 0.90)."
                )

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

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare Areas
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare Areas")
    st.caption(
        "Select 2–5 areas to compare their summary statistics and convergence "
        "rate over time. The convergence chart respects the time-range slider."
    )

    all_locations_cmp = get_all_locations(df_full)

    # Default: pre-fill with current area + first recent area (if any)
    _cmp_defaults = []
    if area_active:
        _cmp_defaults.append(selected_area)
    for _r in st.session_state["recent_areas"]:
        if _r not in _cmp_defaults:
            _cmp_defaults.append(_r)
        if len(_cmp_defaults) >= 2:
            break

    compare_areas = st.multiselect(
        "Choose areas to compare (2–5)",
        options=all_locations_cmp,
        default=[a for a in _cmp_defaults if a in all_locations_cmp],
        max_selections=5,
        key="compare_areas_select",
    )

    if len(compare_areas) < 2:
        st.info("Select at least **2 areas** to see the comparison.")
    else:
        # ── Summary statistics table ─────────────────────────────────────────
        st.subheader("Summary Statistics (full dataset)")
        with st.spinner("Computing statistics…"):
            rows = []
            for area in compare_areas:
                t = _cached_area_timings(df_history, area)
                def _r(v): return round(v, 1) if v is not None else None
                def _ci_str(ci):
                    lo, hi = ci
                    return f"[{lo:.1f}–{hi:.1f}]" if lo is not None else "—"
                rows.append(
                    {
                        "Area": area,
                        "Pre-Alerts": t["n_pre_alerts"],
                        "Sirens": t["n_sirens"],
                        "Pre→Siren (min)": _r(t["avg_pre_to_siren_min"]),
                        "  95% CI": _ci_str(t["ci_pre_to_siren_min"]),
                        "Pre→Release (min)": _r(t["avg_pre_to_clear_min"]),
                        " 95% CI": _ci_str(t["ci_pre_to_clear_min"]),
                        "Siren→Release (min)": _r(t["avg_siren_to_clear_min"]),
                        "95% CI": _ci_str(t["ci_siren_to_clear_min"]),
                        "Convergence %": (
                            round(t["convergence_rate"] * 100, 1)
                            if t["convergence_rate"] is not None else None
                        ),
                    }
                )
            summary_df = pd.DataFrame(rows).set_index("Area")

        st.dataframe(summary_df, width="stretch", key="compare_summary_table")

        st.divider()

        # ── Convergence rate comparison chart ────────────────────────────────
        st.subheader("Convergence Rate Over Time")
        st.caption(
            "7-day rolling average per area. Affected by the time-range slider in the sidebar."
        )
        with st.spinner("Computing convergence timelines…"):
            conv_data = {}
            for area in compare_areas:
                conv_data[area] = _cached_convergence_rate(df, area)

        st.plotly_chart(
            comparison_convergence_chart(conv_data),
            width="stretch",
            key="compare_conv_chart",
        )
