"""
Tests for transforms.py — pure-pandas business logic.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import transforms as T
from tests.helpers import (
    AREA,
    CAT_ALL_CLEAR,
    CAT_DRILL,
    CAT_PRE_ALERT,
    CAT_SIREN,
    _make_row,
)


# ── Category / label helpers ─────────────────────────────────────────────────

class TestApplyEnglishLabels:
    def test_apply_english_labels_known(self):
        df = pd.DataFrame([_make_row(datetime(2024, 1, 1, 8, 0), CAT_SIREN)])
        result = T.apply_english_labels(df)
        assert result["category_en"].iloc[0] == "Rocket / Missile Fire"

    def test_apply_english_labels_unknown(self):
        row = _make_row(datetime(2024, 1, 1, 8, 0), CAT_SIREN)
        row["category"] = 99
        df = pd.DataFrame([row])
        result = T.apply_english_labels(df)
        assert result["category_en"].iloc[0] == "Unknown (99)"


class TestFilterCategories:
    def _make_mixed_cats(self):
        rows = [
            _make_row(datetime(2024, 1, 1, 8, 0), CAT_SIREN),
            _make_row(datetime(2024, 1, 1, 9, 0), CAT_PRE_ALERT),
            _make_row(datetime(2024, 1, 1, 10, 0), CAT_DRILL),
        ]
        return pd.DataFrame(rows)

    def test_filter_categories_excludes_drills_by_default(self):
        df = self._make_mixed_cats()
        result = T.filter_categories(df)
        assert CAT_DRILL not in result["category"].values

    def test_filter_categories_keeps_drills_when_requested(self):
        df = self._make_mixed_cats()
        result = T.filter_categories(df, include_drills=True)
        assert CAT_DRILL in result["category"].values

    def test_filter_categories_by_specific_ids(self):
        df = self._make_mixed_cats()
        result = T.filter_categories(df, include_drills=True, selected_category_ids=[CAT_SIREN])
        assert list(result["category"].unique()) == [CAT_SIREN]


# ── Date filtering ────────────────────────────────────────────────────────────

class TestFilterByDateRange:
    def _make_date_df(self):
        rows = [
            _make_row(datetime(2024, 3, 10, 8, 0), CAT_SIREN),
            _make_row(datetime(2024, 3, 15, 8, 0), CAT_SIREN),
            _make_row(datetime(2024, 3, 20, 8, 0), CAT_SIREN),
        ]
        return pd.DataFrame(rows)

    def test_filter_by_date_range_inclusive(self):
        df = self._make_date_df()
        result = T.filter_by_date_range(df, date(2024, 3, 10), date(2024, 3, 20))
        assert len(result) == 3

    def test_filter_by_date_range_inclusive_boundaries(self):
        df = self._make_date_df()
        # Exactly one row falls on March 10 and one on March 20
        result = T.filter_by_date_range(df, date(2024, 3, 10), date(2024, 3, 10))
        assert len(result) == 1

    def test_filter_by_date_range_empty_result(self):
        df = self._make_date_df()
        result = T.filter_by_date_range(df, date(2020, 1, 1), date(2020, 12, 31))
        assert result.empty


# ── Location ─────────────────────────────────────────────────────────────────

class TestFilterByLocation:
    def _make_loc_df(self):
        rows = [
            _make_row(datetime(2024, 3, 15, 8, 0), CAT_SIREN, location="City Alpha"),
            _make_row(datetime(2024, 3, 15, 9, 0), CAT_SIREN, location="City Beta"),
        ]
        return pd.DataFrame(rows)

    def test_filter_by_location_matches(self):
        df = self._make_loc_df()
        result = T.filter_by_location(df, "City Alpha")
        assert len(result) == 1
        assert result["location_en"].iloc[0] == "City Alpha"

    def test_filter_by_location_no_match(self):
        df = self._make_loc_df()
        result = T.filter_by_location(df, "City Gamma")
        assert result.empty


# ── KPI summary ───────────────────────────────────────────────────────────────

class TestKpiSummary:
    def _make_kpi_df(self):
        rows = [
            _make_row(datetime(2024, 3, 15, 8, 0), CAT_SIREN),
            _make_row(datetime(2024, 3, 15, 9, 0), CAT_SIREN),
            _make_row(datetime(2024, 3, 15, 10, 0), CAT_PRE_ALERT),
            _make_row(datetime(2024, 3, 15, 11, 0), CAT_ALL_CLEAR),
            _make_row(datetime(2024, 3, 15, 12, 0), CAT_DRILL),
        ]
        return pd.DataFrame(rows)

    def test_kpi_summary_counts(self):
        df = self._make_kpi_df()
        summary = T.kpi_summary(df)
        assert summary["total"] == 5
        assert summary["sirens"] == 2
        assert summary["pre_alerts"] == 1
        assert summary["all_clear"] == 1
        assert summary["drills"] == 1


# ── Aggregations ──────────────────────────────────────────────────────────────

class TestMonthlyCounts:
    def test_monthly_counts_returns_rows(self, paired_df):
        result = T.monthly_counts(paired_df)
        assert not result.empty

    def test_monthly_counts_only_sirens(self, paired_df):
        # Pre-alerts (cat 14) must not appear in monthly_counts output
        # monthly_counts only counts SIREN_CATEGORIES
        result = T.monthly_counts(paired_df)
        # The total siren count should equal number of siren rows
        n_sirens = int(paired_df["category"].isin(T.SIREN_CATEGORIES).sum())
        total_counted = result["count"].sum()
        assert total_counted == n_sirens


class TestCategoryTotals:
    def test_category_totals_sorted_descending(self):
        rows = (
            [_make_row(datetime(2024, 3, 15, 8, i), CAT_SIREN) for i in range(5)] +
            [_make_row(datetime(2024, 3, 15, 9, i), CAT_PRE_ALERT) for i in range(2)]
        )
        df = pd.DataFrame(rows)
        result = T.category_totals(df)
        counts = result["count"].tolist()
        assert counts == sorted(counts, reverse=True)


class TestDailyPreAlertSirenCounts:
    def test_daily_pre_alert_siren_counts_types(self, paired_df):
        result = T.daily_pre_alert_siren_counts(paired_df)
        event_types = set(result["type"].unique())
        assert "Pre-Alert" in event_types
        assert "Siren" in event_types


# ── Area timings ──────────────────────────────────────────────────────────────

class TestAreaTimings:
    def test_area_timings_basic(self, paired_df):
        stats = T.area_timings(paired_df, AREA)
        assert stats["n_pre_alerts"] == 10
        assert stats["avg_pre_to_siren_min"] == pytest.approx(5.0, abs=0.01)
        assert stats["convergence_rate"] == pytest.approx(1.0, abs=0.001)

    def test_area_timings_no_match(self, no_match_df):
        stats = T.area_timings(no_match_df, AREA)
        assert stats["convergence_rate"] == pytest.approx(0.0, abs=0.001)
        assert stats["avg_pre_to_siren_min"] is None

    def test_area_timings_empty_area(self):
        df = pd.DataFrame(columns=[
            "date", "alertDate", "category", "data", "category_desc",
            "parsed_date", "parsed_alertDate", "hour", "day_of_week",
            "category_en", "location_en",
        ])
        stats = T.area_timings(df, AREA)
        assert stats["n_pre_alerts"] == 0


# ── Convergence rate over time ────────────────────────────────────────────────

class TestConvergenceRateOverTime:
    def test_convergence_rate_over_time_shape(self, paired_df):
        result = T.convergence_rate_over_time(paired_df, AREA)
        assert set(result.columns) >= {"date", "convergence_rate", "n_pre_alerts"}

    def test_convergence_rate_all_converge(self, paired_df):
        result = T.convergence_rate_over_time(paired_df, AREA)
        assert not result.empty
        assert result["convergence_rate"].mean() == pytest.approx(1.0, abs=0.001)


# ── High-risk windows ─────────────────────────────────────────────────────────

class TestHighRiskWindowsData:
    def _make_siren_df(self):
        rows = [
            _make_row(datetime(2024, 3, 15, 8, 0), CAT_SIREN),
            _make_row(datetime(2024, 3, 15, 8, 1), CAT_SIREN),
            _make_row(datetime(2024, 3, 15, 10, 0), CAT_SIREN),
        ]
        return pd.DataFrame(rows)

    def test_high_risk_windows_data_top_n(self):
        df = self._make_siren_df()
        result = T.high_risk_windows_data(df, top_n=2)
        assert len(result) <= 2

    def test_high_risk_windows_data_empty(self):
        df = pd.DataFrame(columns=[
            "date", "alertDate", "category", "data", "category_desc",
            "parsed_date", "parsed_alertDate", "hour", "day_of_week",
            "category_en", "location_en",
        ])
        result = T.high_risk_windows_data(df, top_n=5)
        assert result.empty
        assert set(result.columns) == {"label", "day_of_week", "hour", "count"}


# ── ML — regression model ─────────────────────────────────────────────────────

class TestTrainAreaModel:
    REQUIRED_KEYS = {
        "model", "location", "n_samples",
        "historical_avg_min", "historical_std_min", "historical_median_min",
        "historical_min_min", "historical_max_min", "y_values", "trained_at",
    }

    def test_train_area_model_success(self, paired_df):
        model_state, err = T.train_area_model(paired_df, AREA)
        assert err is None
        assert model_state is not None
        assert model_state["n_samples"] == 10
        assert self.REQUIRED_KEYS.issubset(set(model_state.keys()))

    def test_train_area_model_insufficient_data(self, no_match_df):
        model_state, err = T.train_area_model(no_match_df, AREA)
        assert model_state is None
        assert isinstance(err, str)
        assert len(err) > 0

    def test_predict_time_to_siren_now_range(self, paired_df):
        model_state, err = T.train_area_model(paired_df, AREA)
        assert err is None
        result = T.predict_time_to_siren_now(model_state)
        window = model_state["window_minutes"]
        assert 0.3 <= result <= window

    def test_train_area_model_progress_cb(self, paired_df):
        calls = []

        def _cb(fraction, text):
            calls.append((fraction, text))

        model_state, err = T.train_area_model(paired_df, AREA, progress_cb=_cb)
        assert err is None
        assert len(calls) > 0
        # Last callback must signal completion with fraction == 1.0
        last_fraction = calls[-1][0]
        assert last_fraction == pytest.approx(1.0, abs=0.001)


# ── ML — classifier ───────────────────────────────────────────────────────────

class TestTrainSirenClassifier:
    REQUIRED_KEYS = {
        "model", "location", "n_samples", "n_with_siren", "base_rate", "trained_at",
    }

    def test_train_siren_classifier_success(self, mixed_df):
        clf_state, err = T.train_siren_classifier(mixed_df, AREA)
        assert err is None
        assert clf_state is not None
        assert self.REQUIRED_KEYS.issubset(set(clf_state.keys()))
        assert 0.0 < clf_state["base_rate"] < 1.0

    def test_train_siren_classifier_insufficient_data(self, paired_df):
        # paired_df has all pre-alerts followed by a siren — no negative class
        clf_state, err = T.train_siren_classifier(paired_df, AREA)
        assert clf_state is None
        assert isinstance(err, str)
        assert len(err) > 0

    def test_predict_siren_probability_range(self, mixed_df):
        clf_state, err = T.train_siren_classifier(mixed_df, AREA)
        assert err is None
        prob = T.predict_siren_probability(clf_state)
        assert 0.0 <= prob <= 1.0


# ── area_timings (vectorised) ─────────────────────────────────────────────────

class TestAreaTimings:
    """
    Verify the merge_asof-based area_timings produces correct values.
    These also guard against regressions if the vectorisation is changed.
    """

    def test_all_pairs_match(self, paired_df):
        """10 pre-alert/siren pairs 5 min apart → 100 % convergence, avg 5 min."""
        t = T.area_timings(paired_df, AREA)
        assert t["n_pre_alerts"] == 10
        assert t["n_sirens"] == 10
        assert t["n_converged"] == 10
        assert t["convergence_rate"] == pytest.approx(1.0)
        assert t["avg_pre_to_siren_min"] == pytest.approx(5.0)

    def test_no_match_outside_window(self, no_match_df):
        """Sirens 20 min after pre-alerts → 0 matches (window = 15 min)."""
        t = T.area_timings(no_match_df, AREA)
        assert t["n_converged"] == 0
        assert t["convergence_rate"] == pytest.approx(0.0)
        assert t["avg_pre_to_siren_min"] is None

    def test_mixed_convergence_rate(self, mixed_df):
        """8 matched + 4 unmatched → 8/12 convergence."""
        t = T.area_timings(mixed_df, AREA)
        assert t["n_pre_alerts"] == 12
        assert t["n_converged"] == 8
        assert t["convergence_rate"] == pytest.approx(8 / 12)

    def test_unknown_area_returns_zeros(self, paired_df):
        """Area not in data → all zeros / Nones, no error."""
        t = T.area_timings(paired_df, "Nowhere")
        assert t["n_pre_alerts"] == 0
        assert t["n_sirens"] == 0
        assert t["convergence_rate"] is None
        assert t["avg_pre_to_siren_min"] is None

    def test_no_sirens_in_area(self, base_time):
        """Only pre-alerts, zero sirens → n_converged=0, avg_siren=None."""
        rows = [_make_row(base_time + pd.Timedelta(hours=i), CAT_PRE_ALERT)
                for i in range(5)]
        df = pd.DataFrame(rows)
        t = T.area_timings(df, AREA)
        assert t["n_converged"] == 0
        assert t["convergence_rate"] == pytest.approx(0.0)
        assert t["avg_pre_to_siren_min"] is None

    def test_returns_required_keys(self, paired_df):
        keys = {"n_pre_alerts", "n_sirens", "avg_pre_to_siren_min",
                "avg_pre_to_clear_min", "convergence_rate", "n_converged"}
        assert keys == set(T.area_timings(paired_df, AREA).keys())


# ── convergence_rate_over_time (vectorised) ───────────────────────────────────

class TestConvergenceRateOverTime:
    """Guard the vectorised convergence timeline against regressions."""

    def test_all_pairs_full_convergence(self, paired_df):
        """
        10 pairs spaced 3 h apart starting at 10:00 → events span two calendar
        days (day 1: pairs 0-4, day 2: pairs 5-9).  Every pre-alert is followed
        by a siren within 5 min, so convergence_rate = 1.0 on every day.
        """
        result = T.convergence_rate_over_time(paired_df, AREA)
        assert not result.empty
        assert (result["convergence_rate"] == 1.0).all()
        assert result["n_pre_alerts"].sum() == 10

    def test_no_sirens_zero_convergence(self, base_time):
        """Pre-alerts only → convergence_rate 0 on each day."""
        rows = [_make_row(base_time + pd.Timedelta(hours=i), CAT_PRE_ALERT)
                for i in range(3)]
        df = pd.DataFrame(rows)
        result = T.convergence_rate_over_time(df, AREA)
        assert not result.empty
        assert (result["convergence_rate"] == 0).all()

    def test_empty_area_returns_empty_df(self, paired_df):
        """Unknown area → empty DataFrame with correct columns."""
        result = T.convergence_rate_over_time(paired_df, "Nowhere")
        assert result.empty
        assert list(result.columns) == ["date", "convergence_rate", "n_pre_alerts"]

    def test_no_match_outside_window(self, no_match_df):
        """Sirens outside window → convergence_rate 0 on all days."""
        result = T.convergence_rate_over_time(no_match_df, AREA)
        assert not result.empty
        assert (result["convergence_rate"] == 0).all()

    def test_mixed_df_partial_convergence(self, mixed_df):
        """8 matched + 4 unmatched → mean convergence_rate < 1."""
        result = T.convergence_rate_over_time(mixed_df, AREA)
        overall = result["convergence_rate"].mean()
        assert 0 < overall < 1

    def test_output_columns(self, paired_df):
        result = T.convergence_rate_over_time(paired_df, AREA)
        assert set(result.columns) == {"date", "convergence_rate", "n_pre_alerts"}


# ── all_areas_risk_summary ────────────────────────────────────────────────────

class TestAllAreasRiskSummary:
    REQUIRED_COLS = {
        "area", "n_pre_alerts", "n_sirens", "n_converged",
        "convergence_rate", "n_days", "freq_score",
    }

    def test_returns_required_columns(self, paired_df):
        result = T.all_areas_risk_summary(paired_df)
        assert self.REQUIRED_COLS == set(result.columns)

    def test_single_area_full_convergence(self, paired_df):
        """paired_df: 10 pre-alerts + 10 sirens 5 min apart → rate = 1.0."""
        result = T.all_areas_risk_summary(paired_df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["area"] == AREA
        assert row["n_pre_alerts"] == 10
        assert row["convergence_rate"] == pytest.approx(1.0)

    def test_single_area_zero_convergence(self, no_match_df):
        """no_match_df: sirens 20 min after pre-alerts → outside 15-min window."""
        result = T.all_areas_risk_summary(no_match_df)
        assert len(result) == 1
        assert result.iloc[0]["convergence_rate"] == pytest.approx(0.0)
        assert result.iloc[0]["n_converged"] == 0

    def test_mixed_df_partial_convergence(self, mixed_df):
        """mixed_df: 12 pre-alerts, 8 converged → rate ≈ 0.667."""
        result = T.all_areas_risk_summary(mixed_df)
        row = result.iloc[0]
        assert row["n_pre_alerts"] == 12
        assert row["n_converged"] == 8
        assert row["convergence_rate"] == pytest.approx(8 / 12)

    def test_freq_score_capped_at_one(self, base_time):
        """An area with more than 1 pre-alert per day should have freq_score = 1.0."""
        # 5 pre-alerts all on the same day (n_days will be 1)
        rows = [_make_row(base_time + pd.Timedelta(hours=i), CAT_PRE_ALERT) for i in range(5)]
        df = pd.DataFrame(rows)
        result = T.all_areas_risk_summary(df)
        assert result.iloc[0]["freq_score"] == pytest.approx(1.0)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=[
            "date", "alertDate", "category", "data", "category_desc",
            "parsed_date", "parsed_alertDate", "hour", "day_of_week",
            "category_en", "location_en",
        ])
        result = T.all_areas_risk_summary(df)
        assert result.empty
        assert self.REQUIRED_COLS == set(result.columns)

    def test_no_pre_alerts_returns_empty(self, base_time):
        """A df with only sirens (no pre-alerts) should return an empty summary."""
        rows = [_make_row(base_time, CAT_SIREN)]
        df = pd.DataFrame(rows)
        result = T.all_areas_risk_summary(df)
        assert result.empty

    def test_multiple_areas(self, base_time):
        """Two areas should each appear as a separate row."""
        rows = (
            [_make_row(base_time + pd.Timedelta(hours=i), CAT_PRE_ALERT, location="City Alpha")
             for i in range(3)]
            + [_make_row(base_time + pd.Timedelta(hours=i), CAT_PRE_ALERT, location="City Beta")
               for i in range(2)]
        )
        df = pd.DataFrame(rows)
        result = T.all_areas_risk_summary(df)
        assert set(result["area"]) == {"City Alpha", "City Beta"}
        assert result[result["area"] == "City Alpha"]["n_pre_alerts"].iloc[0] == 3
        assert result[result["area"] == "City Beta"]["n_pre_alerts"].iloc[0] == 2
