"""
Pure-pandas business logic for the Israel Alerts dashboard.
No Streamlit imports here.
"""

import re
from datetime import date, datetime, timedelta

import numpy as np
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
    Columns: month_dt (datetime), count.
    """
    sirens_df = df[df["category"].isin(SIREN_CATEGORIES)].copy()
    sirens_df = sirens_df.dropna(subset=["parsed_date"])
    # Convert to month-start timestamp directly — avoids Period groupby issues
    sirens_df["month_dt"] = sirens_df["parsed_date"].values.astype("datetime64[M]")
    grouped = (
        sirens_df.groupby("month_dt")
        .size()
        .reset_index(name="count")
    )
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
    Returns top N locations by event count.
    Uses the pre-translated 'location_en' column if present; falls back to
    exploding the raw 'data' column and translating on the fly.
    Columns: location (str), count (int).
    """
    if "location_en" in df.columns:
        # location_en may contain comma-separated names (one row = one alert
        # that spans multiple areas); explode them so each area is counted once.
        locs = (
            df["location_en"]
            .dropna()
            .str.split(",")
            .explode()
            .str.strip()
        )
        locs = locs[locs != ""]
    else:
        locs = _explode_locations(df["data"].dropna())
        locs = locs[locs != ""].apply(_translate_location)
    vc = locs.value_counts().head(n).reset_index()
    # pandas ≥2.0 names the columns [series_name, "count"]; older versions use
    # ["index", series_name].  Normalise to ["location", "count"] by position.
    vc.columns = ["location", "count"]
    return vc


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


# ── Area helpers ─────────────────────────────────────────────────────────────

def get_all_locations(df: pd.DataFrame) -> list[str]:
    """Return a sorted list of all unique English location names in the dataset."""
    return sorted(
        df["location_en"]
        .dropna()
        .str.split(", ")
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )


def filter_by_location(df: pd.DataFrame, location_en: str) -> pd.DataFrame:
    """Return only rows that include location_en in their location_en field."""
    mask = df["location_en"].str.contains(location_en, na=False, regex=False)
    return df[mask]


def _bootstrap_ci(
    values: list, n_boot: int = 500, ci: float = 95
) -> "tuple[float, float] | tuple[None, None]":
    """
    Non-parametric bootstrap confidence interval for the mean.
    Makes no normality assumption — safe for right-skewed timing data.
    Returns (lo, hi) rounded to 1 dp, or (None, None) when n < 3.
    """
    n = len(values)
    if n < 3:
        return (None, None)
    rng = np.random.default_rng(42)
    a = np.asarray(values, dtype=float)
    means = rng.choice(a, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(means, alpha))
    hi = float(np.percentile(means, 100.0 - alpha))
    return (round(lo, 1), round(hi, 1))


def area_timings(
    df: pd.DataFrame,
    location_en: str,
    window_minutes: int = 15,
    clear_window_minutes: int = 60,
) -> dict:
    """
    For a given area, compute timing statistics between event types.

    Returns (all timing fields are None when insufficient data):
      n_pre_alerts            : total pre-alerts in area
      n_sirens                : total sirens in area
      convergence_rate        : fraction of pre-alerts followed by siren within window
      n_converged             : count of pre-alerts that preceded a siren
      avg_pre_to_siren_min    : mean minutes pre-alert → next siren
      ci_pre_to_siren_min     : (lo, hi) 95% bootstrap CI, or (None, None)
      avg_pre_to_clear_min    : mean minutes pre-alert → next release
      ci_pre_to_clear_min     : (lo, hi) 95% bootstrap CI, or (None, None)
      avg_siren_to_clear_min  : mean minutes siren → next release
      ci_siren_to_clear_min   : (lo, hi) 95% bootstrap CI, or (None, None)
      n_siren_to_clear        : number of siren→release pairs found

    window_minutes       — tolerance for pre-alert → siren (default 15 min)
    clear_window_minutes — tolerance for *→release pairings (default 60 min);
                           wider because releases arrive after siren + shelter.

    Uses pd.merge_asof (O(n log n)) instead of iterrows (O(n²)).
    """
    area_df = filter_by_location(df, location_en)

    pre_alerts = (
        area_df[area_df["category"].isin(PRE_ALERT_CATEGORIES)][["parsed_alertDate"]]
        .dropna().sort_values("parsed_alertDate").reset_index(drop=True)
    )
    sirens = (
        area_df[area_df["category"].isin(SIREN_CATEGORIES)][["parsed_alertDate"]]
        .dropna().sort_values("parsed_alertDate").reset_index(drop=True)
    )
    all_clears = (
        area_df[area_df["category"].isin(ALL_CLEAR_CATEGORIES)][["parsed_alertDate"]]
        .dropna().sort_values("parsed_alertDate").reset_index(drop=True)
    )

    n_pre = len(pre_alerts)
    tol      = pd.Timedelta(minutes=window_minutes)
    tol_clr  = pd.Timedelta(minutes=clear_window_minutes)

    _none_ci = (None, None)
    _empty = {
        "n_pre_alerts": n_pre, "n_sirens": len(sirens),
        "convergence_rate": None, "n_converged": 0,
        "avg_pre_to_siren_min": None,   "ci_pre_to_siren_min": _none_ci,
        "avg_pre_to_clear_min": None,   "ci_pre_to_clear_min": _none_ci,
        "avg_siren_to_clear_min": None, "ci_siren_to_clear_min": _none_ci,
        "n_siren_to_clear": 0,
    }
    if n_pre == 0:
        return _empty

    pre_s = pre_alerts.rename(columns={"parsed_alertDate": "pre_time"})

    # ── pre-alert → siren ─────────────────────────────────────────────────────
    n_converged = 0
    avg_pre_to_siren = None
    ci_pre_to_siren  = _none_ci
    if not sirens.empty:
        m_s = pd.merge_asof(
            pre_s, sirens.rename(columns={"parsed_alertDate": "siren_time"}),
            left_on="pre_time", right_on="siren_time",
            direction="forward", tolerance=tol,
        )
        hit_s = m_s["siren_time"].notna()
        n_converged = int(hit_s.sum())
        d_s = (m_s.loc[hit_s, "siren_time"] - m_s.loc[hit_s, "pre_time"]).dt.total_seconds() / 60
        if not d_s.empty:
            avg_pre_to_siren = float(d_s.mean())
            ci_pre_to_siren  = _bootstrap_ci(d_s.tolist())

    # ── pre-alert → release ───────────────────────────────────────────────────
    avg_pre_to_clear = None
    ci_pre_to_clear  = _none_ci
    if not all_clears.empty:
        m_c = pd.merge_asof(
            pre_s, all_clears.rename(columns={"parsed_alertDate": "clear_time"}),
            left_on="pre_time", right_on="clear_time",
            direction="forward", tolerance=tol_clr,
        )
        hit_c = m_c["clear_time"].notna()
        d_c = (m_c.loc[hit_c, "clear_time"] - m_c.loc[hit_c, "pre_time"]).dt.total_seconds() / 60
        if not d_c.empty:
            avg_pre_to_clear = float(d_c.mean())
            ci_pre_to_clear  = _bootstrap_ci(d_c.tolist())

    # ── siren → release ───────────────────────────────────────────────────────
    avg_siren_to_clear = None
    ci_siren_to_clear  = _none_ci
    n_siren_to_clear   = 0
    if not sirens.empty and not all_clears.empty:
        siren_s = sirens.rename(columns={"parsed_alertDate": "siren_time"})
        m_sc = pd.merge_asof(
            siren_s, all_clears.rename(columns={"parsed_alertDate": "clear_time"}),
            left_on="siren_time", right_on="clear_time",
            direction="forward", tolerance=tol_clr,
        )
        hit_sc = m_sc["clear_time"].notna()
        n_siren_to_clear = int(hit_sc.sum())
        d_sc = (m_sc.loc[hit_sc, "clear_time"] - m_sc.loc[hit_sc, "siren_time"]).dt.total_seconds() / 60
        if not d_sc.empty:
            avg_siren_to_clear = float(d_sc.mean())
            ci_siren_to_clear  = _bootstrap_ci(d_sc.tolist())

    return {
        "n_pre_alerts": n_pre,
        "n_sirens": len(sirens),
        "convergence_rate": (n_converged / n_pre) if n_pre > 0 else None,
        "n_converged": n_converged,
        "avg_pre_to_siren_min":   avg_pre_to_siren,
        "ci_pre_to_siren_min":    ci_pre_to_siren,
        "avg_pre_to_clear_min":   avg_pre_to_clear,
        "ci_pre_to_clear_min":    ci_pre_to_clear,
        "avg_siren_to_clear_min": avg_siren_to_clear,
        "ci_siren_to_clear_min":  ci_siren_to_clear,
        "n_siren_to_clear":       n_siren_to_clear,
    }


def daily_pre_alert_siren_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily counts of Pre-Alert vs Siren events.
    Returns columns: parsed_date, type ('Pre-Alert' or 'Siren'), count.
    """
    relevant = df[
        df["category"].isin(PRE_ALERT_CATEGORIES | SIREN_CATEGORIES)
    ].copy()
    relevant["type"] = relevant["category"].apply(
        lambda c: "Pre-Alert" if c in PRE_ALERT_CATEGORIES else "Siren"
    )
    grouped = (
        relevant.groupby(["parsed_date", "type"])
        .size()
        .reset_index(name="count")
    )
    return grouped.sort_values("parsed_date")


# ── Heatmap insights ─────────────────────────────────────────────────────────

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def heatmap_insights(df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """
    Return descriptive strings for the top N busiest day+hour combinations
    for siren-type events, suitable for display below the heatmap.
    """
    data = high_risk_windows_data(df, top_n)
    return [
        f"**{row['label']}** · {row['count']:,} siren events"
        for _, row in data.iterrows()
    ]


def high_risk_windows_data(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return a DataFrame of the top N busiest day+hour combinations for siren events.
    Columns: label (str), day_of_week (int), hour (int), count (int).
    Sorted descending by count.
    """
    sirens_df = df[df["category"].isin(SIREN_CATEGORIES)].dropna(
        subset=["day_of_week", "hour"]
    )
    if sirens_df.empty:
        return pd.DataFrame(columns=["label", "day_of_week", "hour", "count"])

    counts = (
        sirens_df.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    counts["label"] = counts.apply(
        lambda r: f"{_DAY_NAMES[int(r['day_of_week'])]} {int(r['hour']):02d}:00–{int(r['hour']) + 1:02d}:00",
        axis=1,
    )
    return counts[["label", "day_of_week", "hour", "count"]].reset_index(drop=True)


# ── ML prediction ─────────────────────────────────────────────────────────────

def _make_features(hour: int, dow: int):
    """Build the 6-element feature vector for a given hour + day-of-week."""
    import numpy as np
    return [
        hour, dow,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ]


def train_area_model(
    df: pd.DataFrame,
    location_en: str,
    window_minutes: int = 15,
    progress_cb=None,
) -> tuple:
    """
    Train a Ridge regression on historical pre-alert → siren pairs for the area.

    progress_cb(fraction: float, text: str) is called at key stages if provided.
    Returns (model_state_dict, error_str).
    model_state_dict is None when there is insufficient training data.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    def _cb(f, t):
        if progress_cb:
            progress_cb(f, t)

    _cb(0.05, "Filtering area data…")
    area_df = filter_by_location(df, location_en).sort_values("parsed_alertDate")

    _cb(0.15, "Identifying pre-alerts & sirens…")
    pre_alerts = area_df[area_df["category"].isin(PRE_ALERT_CATEGORIES)].reset_index(drop=True)
    sirens = area_df[area_df["category"].isin(SIREN_CATEGORIES)].reset_index(drop=True)

    window = pd.Timedelta(minutes=window_minutes)
    X: list[list[float]] = []
    y: list[float] = []
    n_pre = len(pre_alerts)

    for i, (_, pre) in enumerate(pre_alerts.iterrows()):
        if n_pre > 0:
            _cb(0.20 + 0.55 * (i / n_pre), f"Matching pairs… {i}/{n_pre}")
        t = pre["parsed_alertDate"]
        if pd.isnull(t):
            continue
        next_sirens = sirens[
            (sirens["parsed_alertDate"] > t) & (sirens["parsed_alertDate"] <= t + window)
        ]
        if not next_sirens.empty:
            delta_min = (next_sirens["parsed_alertDate"].min() - t).total_seconds() / 60
            X.append(_make_features(t.hour, t.dayofweek))
            y.append(delta_min)

    n = len(X)
    if n < 5:
        _cb(1.0, "Insufficient data")
        return None, (
            f"Only {n} matched pre→siren pair(s) found for '{location_en}' "
            "(need ≥ 5). Try a busier area or load more data."
        )

    _cb(0.80, "Building feature matrix…")
    X_arr, y_arr = np.array(X), np.array(y)

    _cb(0.90, "Fitting Ridge regression…")
    model = Ridge()
    model.fit(X_arr, y_arr)
    _cb(1.0, "Complete")

    return {
        "model": model,
        "location": location_en,
        "window_minutes": window_minutes,
        "n_samples": n,
        "historical_avg_min": float(np.mean(y_arr)),
        "historical_std_min": float(np.std(y_arr)),
        "historical_median_min": float(np.median(y_arr)),
        "historical_min_min": float(y_arr.min()),
        "historical_max_min": float(y_arr.max()),
        "y_values": y_arr.tolist(),
        "trained_at": pd.Timestamp.now().strftime("%H:%M:%S"),
    }, None


def convergence_rate_over_time(
    df: pd.DataFrame, location_en: str, window_minutes: int = 15
) -> pd.DataFrame:
    """
    Daily convergence rate for a given area: fraction of pre-alerts
    that were followed by a siren within window_minutes.

    Returns columns: date, convergence_rate (0–1), n_pre_alerts.
    Uses pd.merge_asof (O(n log n)) instead of iterrows (O(n²)).
    """
    area_df = filter_by_location(df, location_en)
    pre_alerts = (
        area_df[area_df["category"].isin(PRE_ALERT_CATEGORIES)][
            ["parsed_alertDate", "parsed_date"]
        ]
        .dropna(subset=["parsed_alertDate"])
        .sort_values("parsed_alertDate")
        .reset_index(drop=True)
    )
    sirens = (
        area_df[area_df["category"].isin(SIREN_CATEGORIES)][["parsed_alertDate"]]
        .dropna()
        .sort_values("parsed_alertDate")
        .reset_index(drop=True)
    )

    if pre_alerts.empty:
        return pd.DataFrame(columns=["date", "convergence_rate", "n_pre_alerts"])

    if sirens.empty:
        pre_alerts["converged"] = 0
    else:
        merged = pd.merge_asof(
            pre_alerts,
            sirens.rename(columns={"parsed_alertDate": "siren_time"}),
            left_on="parsed_alertDate", right_on="siren_time",
            direction="forward",
            tolerance=pd.Timedelta(minutes=window_minutes),
        )
        pre_alerts = merged.copy()
        pre_alerts["converged"] = pre_alerts["siren_time"].notna().astype(int)

    daily = (
        pre_alerts.groupby("parsed_date")
        .agg(convergence_rate=("converged", "mean"), n_pre_alerts=("converged", "count"))
        .reset_index()
        .rename(columns={"parsed_date": "date"})
    )
    return daily.sort_values("date")


def predict_time_to_siren_now(model_state: dict) -> float:
    """
    Instant inference using a previously trained model_state.
    Returns predicted minutes (clamped to [0.3, window_minutes]).
    """
    import numpy as np

    now = pd.Timestamp.now()
    X = np.array([_make_features(now.hour, now.dayofweek)])
    raw = float(model_state["model"].predict(X)[0])
    return float(np.clip(raw, 0.3, model_state["window_minutes"]))


def train_siren_classifier(
    df: pd.DataFrame,
    location_en: str,
    window_minutes: int = 15,
    progress_cb=None,
) -> tuple:
    """
    Train a logistic regression classifier to predict P(siren | pre-alert in area).

    progress_cb(fraction: float, text: str) is called at key stages if provided.
    Returns (classifier_state_dict, error_str).
    classifier_state_dict is None when there is insufficient training data.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    def _cb(f, t):
        if progress_cb:
            progress_cb(f, t)

    _cb(0.05, "Filtering area data…")
    area_df = filter_by_location(df, location_en).sort_values("parsed_alertDate")

    _cb(0.15, "Identifying pre-alerts & sirens…")
    pre_alerts = area_df[area_df["category"].isin(PRE_ALERT_CATEGORIES)].reset_index(drop=True)
    sirens = area_df[area_df["category"].isin(SIREN_CATEGORIES)].reset_index(drop=True)

    window = pd.Timedelta(minutes=window_minutes)
    X: list[list[float]] = []
    y: list[int] = []
    n_pre = len(pre_alerts)

    for i, (_, pre) in enumerate(pre_alerts.iterrows()):
        if n_pre > 0:
            _cb(0.20 + 0.65 * (i / n_pre), f"Building events… {i}/{n_pre}")
        t = pre["parsed_alertDate"]
        if pd.isnull(t):
            continue
        followed_by_siren = not sirens[
            (sirens["parsed_alertDate"] > t) & (sirens["parsed_alertDate"] <= t + window)
        ].empty
        X.append(_make_features(t.hour, t.dayofweek))
        y.append(int(followed_by_siren))

    n = len(X)
    n_pos = sum(y)

    if n < 10 or n_pos < 2 or (n - n_pos) < 2:
        _cb(1.0, "Insufficient data")
        return None, (
            f"Classifier: only {n} pre-alerts, {n_pos} with siren. "
            "Need ≥10 events with both outcomes."
        )

    _cb(0.90, "Fitting logistic regression…")
    clf = LogisticRegression(max_iter=200)
    clf.fit(np.array(X), np.array(y))
    _cb(1.0, "Complete")

    return {
        "model": clf,
        "location": location_en,
        "n_samples": n,
        "n_with_siren": n_pos,
        "base_rate": float(n_pos / n),
        "trained_at": pd.Timestamp.now().strftime("%H:%M:%S"),
    }, None


def predict_siren_probability(classifier_state: dict) -> float:
    """
    Predict P(siren | pre-alert now) for the current hour and day of week.
    Returns probability in [0, 1].
    """
    import numpy as np

    now = pd.Timestamp.now()
    X = np.array([_make_features(now.hour, now.dayofweek)])
    return float(classifier_state["model"].predict_proba(X)[0][1])


# ── Risk score validation helpers ───────────────────────────────────────────

def all_areas_risk_summary(df: pd.DataFrame, window_minutes: int = 15) -> pd.DataFrame:
    """
    Compute per-area convergence_rate and event-frequency score for every area
    that has at least one pre-alert in *df*.

    Returns a DataFrame with columns:
        area, n_pre_alerts, n_sirens, n_converged,
        convergence_rate, n_days, freq_score

    where:
        convergence_rate = fraction of pre-alerts followed by a siren
                           within *window_minutes*
        freq_score       = min(1.0, pre_alerts_per_day)  — capped at 1
    """
    _empty = pd.DataFrame(columns=[
        "area", "n_pre_alerts", "n_sirens", "n_converged",
        "convergence_rate", "n_days", "freq_score",
    ])

    pre = df[df["category"].isin(PRE_ALERT_CATEGORIES)].copy()
    if pre.empty:
        return _empty

    sir = df[df["category"].isin(SIREN_CATEGORIES)].copy()

    n_days = max(
        1,
        (df["parsed_alertDate"].max() - df["parsed_alertDate"].min()).days + 1,
    )
    tol = pd.Timedelta(minutes=window_minutes)

    pre = pre.sort_values("parsed_alertDate")
    sir = sir.sort_values("parsed_alertDate")

    records = []
    for area in pre["location_en"].dropna().unique():
        pre_a = pre[pre["location_en"] == area]
        n_pa  = len(pre_a)
        sir_a = sir[sir["location_en"] == area]
        n_sir = len(sir_a)

        if n_sir == 0:
            n_conv = 0
        else:
            m = pd.merge_asof(
                pre_a[["parsed_alertDate"]].rename(columns={"parsed_alertDate": "pre_time"}),
                sir_a[["parsed_alertDate"]].rename(columns={"parsed_alertDate": "siren_time"}),
                left_on="pre_time",
                right_on="siren_time",
                direction="forward",
                tolerance=tol,
            )
            n_conv = int(m["siren_time"].notna().sum())

        records.append({
            "area": area,
            "n_pre_alerts": n_pa,
            "n_sirens": n_sir,
            "n_converged": n_conv,
            "convergence_rate": n_conv / n_pa,
        })

    if not records:
        return _empty

    result = pd.DataFrame(records)
    result["n_days"]     = n_days
    result["freq_score"] = (result["n_pre_alerts"] / n_days).clip(upper=1.0)

    return result.sort_values("n_pre_alerts", ascending=False).reset_index(drop=True)


# ── Siren heatmap data ───────────────────────────────────────────────────────

def siren_counts_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns [location, lat, lon, count] for all
    siren-category events that can be geocoded via LOCATION_COORDS.
    """
    from location_coords import LOCATION_COORDS

    sirens = df[df["category"].isin(SIREN_CATEGORIES)]
    if sirens.empty:
        return pd.DataFrame(columns=["location", "lat", "lon", "count"])

    counts = (
        sirens["location_en"]
        .str.split(", ")
        .explode()
        .str.strip()
        .dropna()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["location", "count"]

    coords = counts["location"].map(LOCATION_COORDS)
    counts["lat"] = coords.map(lambda c: c[0] if isinstance(c, tuple) else None)
    counts["lon"] = coords.map(lambda c: c[1] if isinstance(c, tuple) else None)

    return counts.dropna(subset=["lat", "lon"]).reset_index(drop=True)
