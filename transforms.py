"""
Pure-pandas business logic for the Israel Alerts dashboard.
No Streamlit imports here.
"""

import re
from datetime import date, datetime, timedelta

import pandas as pd

from location_map import LOCATION_MAP

# ── Category mapping ────────────────────────────────────────────────────────

CATEGORY_MAP: dict[int, str] = {
    1: "Rocket / Missile Fire",
    2: "Hostile Aircraft Intrusion",
    3: "Unconventional Rocket",
    4: "Warning",
    5: "Terror Warning",
    6: "Hazardous Materials",
    7: "Earthquake",
    8: "Tsunami",
    9: "Radiological Event",
    10: "Terrorist Infiltration",
    11: "Hostile Vehicle",
    13: "All Clear",
    14: "Pre-Alert",
    15: "Drill — Rockets",
    21: "Drill — Earthquake",
    26: "Drill — Hazardous Materials",
}

SIREN_CATEGORIES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
PRE_ALERT_CATEGORIES = {14}
ALL_CLEAR_CATEGORIES = {13}
DRILL_CATEGORIES = {15, 21, 26}


# ── Category helpers ─────────────────────────────────────────────────────────

def apply_english_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'category_en' column with English alert type labels."""
    df = df.copy()
    df["category_en"] = df["category"].map(
        lambda c: CATEGORY_MAP.get(int(c), f"Unknown ({c})")
    )
    return df


# ── Location helpers ─────────────────────────────────────────────────────────

_TRAILING_DIGITS = re.compile(r"\s+\d+$")


def _translate_location(name: str) -> str:
    """Translate a single Hebrew location name to English using LOCATION_MAP."""
    name = name.strip()
    # 1. Exact match
    if name in LOCATION_MAP:
        return LOCATION_MAP[name]
    # 2. Strip trailing digits (e.g. "באר שבע 288")
    stripped = _TRAILING_DIGITS.sub("", name)
    if stripped in LOCATION_MAP:
        return LOCATION_MAP[stripped]
    # 3. Strip " - sub-area" suffix and try base city
    if " - " in stripped:
        base = stripped.split(" - ")[0]
        if base in LOCATION_MAP:
            return LOCATION_MAP[base] + " area"
    # 4. Fall back to Hebrew text
    return name


def _explode_locations(series: pd.Series) -> pd.Series:
    """
    The 'data' field can contain multiple comma-separated locations.
    Return a flat Series with one location per element (still associated with
    the original row index via repeat).
    """
    return series.str.split(",").explode().str.strip()


def apply_english_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'location_en' column with English location names."""
    df = df.copy()
    df["location_en"] = df["data"].apply(
        lambda cell: ", ".join(
            _translate_location(loc)
            for loc in str(cell).split(",")
        )
    )
    return df


# ── Date filtering ───────────────────────────────────────────────────────────

def filter_by_date_range(
    df: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Return rows where parsed_date falls in [start, end] inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df["parsed_date"] >= start_ts) & (df["parsed_date"] <= end_ts)
    return df[mask]


def filter_categories(
    df: pd.DataFrame,
    include_drills: bool = False,
    selected_category_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Filter by category. Drills are excluded unless include_drills=True."""
    if not include_drills:
        df = df[~df["category"].isin(DRILL_CATEGORIES)]
    if selected_category_ids is not None:
        df = df[df["category"].isin(selected_category_ids)]
    return df


# ── KPI summary ──────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame) -> dict:
    """
    Returns a dict with counts for the KPI metric row.
    """
    now = pd.Timestamp.now()
    last_24h = df[df["parsed_alertDate"] >= now - pd.Timedelta(hours=24)]
    last_7d = df[df["parsed_alertDate"] >= now - pd.Timedelta(days=7)]

    all_locs = (
        df["data"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .nunique()
    )

    return {
        "total": len(df),
        "sirens": int(df["category"].isin(SIREN_CATEGORIES).sum()),
        "pre_alerts": int(df["category"].isin(PRE_ALERT_CATEGORIES).sum()),
        "all_clear": int(df["category"].isin(ALL_CLEAR_CATEGORIES).sum()),
        "drills": int(df["category"].isin(DRILL_CATEGORIES).sum()),
        "unique_locations": int(all_locs),
        "last_24h": len(last_24h),
        "last_7d": len(last_7d),
    }


def kpi_delta(df: pd.DataFrame, current_start: date, current_end: date) -> dict:
    """
    Compute delta counts: current window vs the same-length window immediately before.
    Returns dict with same keys as kpi_summary but holding delta integers.
    """
    length = (
        pd.Timestamp(current_end) - pd.Timestamp(current_start)
    ) + pd.Timedelta(days=1)
    prev_end = pd.Timestamp(current_start) - pd.Timedelta(days=1)
    prev_start = prev_end - length + pd.Timedelta(days=1)

    current_df = filter_by_date_range(df, current_start, current_end)
    previous_df = filter_by_date_range(df, prev_start.date(), prev_end.date())

    def _count(d):
        return {
            "total": len(d),
            "sirens": int(d["category"].isin(SIREN_CATEGORIES).sum()),
            "pre_alerts": int(d["category"].isin(PRE_ALERT_CATEGORIES).sum()),
            "all_clear": int(d["category"].isin(ALL_CLEAR_CATEGORIES).sum()),
        }

    cur = _count(current_df)
    prev = _count(previous_df)
    return {k: cur[k] - prev[k] for k in cur}


# ── Aggregations for charts ──────────────────────────────────────────────────

def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame grouped by week and category_en.
    Columns: week_start (datetime), category_en, count.
    Grouped by ISO week to keep point count manageable over 10+ years.
    """
    d = df.copy()
    d["week_start"] = d["parsed_date"] - pd.to_timedelta(
        d["parsed_date"].dt.dayofweek, unit="D"
    )
    grouped = (
        d.groupby(["week_start", "category_en"])
        .size()
        .reset_index(name="count")
    )
    return grouped.sort_values("week_start")


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns total siren-type events per month.
    Columns: month (Period), count.
    """
    sirens_df = df[df["category"].isin(SIREN_CATEGORIES)].copy()
    sirens_df["month"] = sirens_df["parsed_date"].dt.to_period("M")
    grouped = (
        sirens_df.groupby("month")
        .size()
        .reset_index(name="count")
    )
    grouped["month_dt"] = grouped["month"].dt.to_timestamp()
    return grouped.sort_values("month_dt")


def category_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Returns category_en vs count, sorted descending."""
    return (
        df.groupby("category_en")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def top_locations(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Explodes the 'data' field, translates to English, and returns top N
    locations by event count.
    Columns: location (str), count (int).
    """
    locs = _explode_locations(df["data"].dropna())
    locs = locs[locs != ""].apply(_translate_location)
    return (
        locs.value_counts()
        .head(n)
        .reset_index()
        .rename(columns={"index": "location", "data": "location", "count": "count"})
        .rename(columns={locs.name if locs.name else "data": "location"})
    )


def hourly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table suitable for a day-of-week × hour heatmap.
    Rows = day of week (0=Mon … 6=Sun), columns = hour (0..23).
    """
    DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = (
        df.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    # Ensure all 24 hours are present
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    pivot = pivot.reindex(range(7), fill_value=0)
    pivot.index = [DAY_LABELS[i] for i in pivot.index]
    return pivot
