"""
Data fetching and caching for the Israel Alerts dashboard.

Loading is always manual (triggered by the user clicking "Load / Refresh data").
@st.cache_data with no TTL means the result is reused across rerenders until
the cache is explicitly cleared via st.cache_data.clear().
"""

import io

import pandas as pd
import requests
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/dleshem/israel-alerts-data/main/israel-alerts.csv"
)


@st.cache_data(show_spinner=False)
def load_raw_df() -> pd.DataFrame:
    """Download the full CSV from GitHub. Cached until explicitly cleared."""
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add typed datetime columns:
      - parsed_date       : from the 'date' column  (DD.MM.YYYY)
      - parsed_alertDate  : from the 'alertDate' column (ISO 8601)
    """
    df = df.copy()
    df["parsed_date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df["parsed_alertDate"] = pd.to_datetime(df["alertDate"], errors="coerce")
    df["hour"] = df["parsed_alertDate"].dt.hour
    df["day_of_week"] = df["parsed_alertDate"].dt.dayofweek  # 0=Mon … 6=Sun
    return df


def load_dashboard_df() -> pd.DataFrame:
    """Public entry point. Returns a fully parsed DataFrame ready for transforms."""
    df = load_raw_df()
    df = parse_dates(df)
    return df
