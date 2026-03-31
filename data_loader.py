"""
Data fetching and parsing for the Israel Alerts dashboard.

On startup the app checks for a local cache file and loads it automatically.
"Load / Refresh Data" downloads fresh data from GitHub and overwrites the cache.
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/dleshem/israel-alerts-data/main/israel-alerts.csv"
)
LOCAL_CACHE_PATH = Path("israel-alerts.csv")


# ── Local cache ───────────────────────────────────────────────────────────────

def local_cache_info() -> tuple[bool, datetime | None]:
    """
    Returns (exists, mtime).
    mtime is the file's last-modified time as a datetime, or None if missing.
    """
    if LOCAL_CACHE_PATH.exists():
        mtime = datetime.fromtimestamp(LOCAL_CACHE_PATH.stat().st_mtime)
        return True, mtime
    return False, None


def load_from_local() -> bytes | None:
    """Read the local cache file. Returns raw bytes or None if not found."""
    if LOCAL_CACHE_PATH.exists():
        return LOCAL_CACHE_PATH.read_bytes()
    return None


def _save_to_local(csv_bytes: bytes) -> None:
    """Persist downloaded bytes to the local cache file."""
    LOCAL_CACHE_PATH.write_bytes(csv_bytes)


# ── Remote download ───────────────────────────────────────────────────────────

def stream_download(on_progress: Callable[[float], None]) -> bytes:
    """
    Stream-download the CSV, call on_progress(0.0–1.0) as data arrives,
    save to local cache, and return raw bytes.
    """
    with requests.get(DATA_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        received = 0
        chunks: list[bytes] = []
        for chunk in resp.iter_content(chunk_size=65_536):
            if chunk:
                chunks.append(chunk)
                received += len(chunk)
                if total:
                    on_progress(min(received / total, 1.0))
        on_progress(1.0)
        csv_bytes = b"".join(chunks)
    _save_to_local(csv_bytes)
    return csv_bytes


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add typed datetime columns:
      - parsed_date       : from the 'date' column  (DD.MM.YYYY)
      - parsed_alertDate  : from the 'alertDate' column (ISO 8601)
      - hour              : hour of day (0-23)
      - day_of_week       : 0=Mon … 6=Sun
    """
    df = df.copy()
    df["parsed_date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
    df["parsed_alertDate"] = pd.to_datetime(df["alertDate"], errors="coerce")
    df["hour"] = df["parsed_alertDate"].dt.hour
    df["day_of_week"] = df["parsed_alertDate"].dt.dayofweek
    return df


def load_dashboard_df(csv_bytes: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes into a fully typed DataFrame ready for transforms."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = parse_dates(df)
    return df
