"""
Tests for data_loader.py — CSV parsing and local cache helpers.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_loader


# ── parse_dates ───────────────────────────────────────────────────────────────

class TestParseDates:
    def _make_raw_df(self):
        """A minimal DataFrame that mimics raw CSV columns before parse_dates."""
        return pd.DataFrame([
            {
                "date":      "15.03.2024",
                "alertDate": "2024-03-15T10:30:45",
                "category":  1,
                "data":      "Test Area",
            },
            {
                "date":      "20.07.2023",
                "alertDate": "2023-07-20T23:59:00",
                "category":  14,
                "data":      "Another Area",
            },
        ])

    def test_parse_dates_format(self):
        df = data_loader.parse_dates(self._make_raw_df())
        # First row: 15 March 2024
        assert df["parsed_date"].iloc[0] == pd.Timestamp("2024-03-15")

    def test_parse_dates_adds_columns(self):
        df = data_loader.parse_dates(self._make_raw_df())
        for col in ("parsed_date", "parsed_alertDate", "hour", "day_of_week"):
            assert col in df.columns, f"Missing column: {col}"

    def test_parse_dates_hour_range(self):
        df = data_loader.parse_dates(self._make_raw_df())
        assert df["hour"].between(0, 23).all()

    def test_parse_dates_dow_range(self):
        df = data_loader.parse_dates(self._make_raw_df())
        assert df["day_of_week"].between(0, 6).all()

    def test_parse_dates_hour_value(self):
        df = data_loader.parse_dates(self._make_raw_df())
        # "2024-03-15T10:30:45" → hour 10
        assert df["hour"].iloc[0] == 10

    def test_parse_dates_does_not_mutate_input(self):
        raw = self._make_raw_df()
        original_cols = list(raw.columns)
        data_loader.parse_dates(raw)
        assert list(raw.columns) == original_cols


# ── load_dashboard_df ─────────────────────────────────────────────────────────

class TestLoadDashboardDf:
    def test_load_dashboard_df_shape(self, raw_csv_bytes):
        df = data_loader.load_dashboard_df(raw_csv_bytes)
        # raw_csv_bytes fixture has 3 data rows
        assert len(df) == 3

    def test_load_dashboard_df_columns(self, raw_csv_bytes):
        df = data_loader.load_dashboard_df(raw_csv_bytes)
        assert "parsed_date" in df.columns
        assert "parsed_alertDate" in df.columns

    def test_load_dashboard_df_parsed_date_type(self, raw_csv_bytes):
        df = data_loader.load_dashboard_df(raw_csv_bytes)
        assert pd.api.types.is_datetime64_any_dtype(df["parsed_date"])

    def test_load_dashboard_df_parsed_alertdate_type(self, raw_csv_bytes):
        df = data_loader.load_dashboard_df(raw_csv_bytes)
        assert pd.api.types.is_datetime64_any_dtype(df["parsed_alertDate"])

    def test_load_dashboard_df_categories_preserved(self, raw_csv_bytes):
        df = data_loader.load_dashboard_df(raw_csv_bytes)
        assert set(df["category"].tolist()) == {1, 14, 13}


# ── local_cache_info ──────────────────────────────────────────────────────────

class TestLocalCacheInfo:
    def test_local_cache_info_missing(self, tmp_path):
        """Returns (False, None) when the cache file does not exist."""
        non_existent = tmp_path / "does_not_exist.csv"
        with patch.object(data_loader, "LOCAL_CACHE_PATH", non_existent):
            exists, mtime = data_loader.local_cache_info()
        assert exists is False
        assert mtime is None

    def test_local_cache_info_present(self, tmp_path):
        """Returns (True, datetime) when the cache file exists."""
        cache_file = tmp_path / "israel-alerts.csv"
        cache_file.write_bytes(b"date,alertDate,category,data\n")
        with patch.object(data_loader, "LOCAL_CACHE_PATH", cache_file):
            exists, mtime = data_loader.local_cache_info()
        assert exists is True
        assert mtime is not None
