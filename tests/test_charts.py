"""
Smoke tests for charts.py — verify each factory returns a go.Figure and does
not raise.  No visual/pixel assertions; just structural sanity checks.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import charts
from tests.helpers import AREA, CAT_PRE_ALERT, CAT_SIREN, _make_row


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_figure(fig):
    """Assert that *fig* is a Plotly Figure instance."""
    assert isinstance(fig, go.Figure)


def _minimal_model_state(n: int = 10) -> dict:
    """Build a minimal model_state dict for prediction_distribution_chart."""
    y = np.linspace(2.0, 10.0, n)
    return {
        "model":                None,   # not needed for chart
        "location":             AREA,
        "window_minutes":       15,
        "n_samples":            n,
        "historical_avg_min":   float(np.mean(y)),
        "historical_std_min":   float(np.std(y)),
        "historical_median_min": float(np.median(y)),
        "historical_min_min":   float(y.min()),
        "historical_max_min":   float(y.max()),
        "y_values":             y.tolist(),
        "trained_at":           "10:00:00",
    }


# ── prediction_distribution_chart ────────────────────────────────────────────

class TestPredictionDistributionChart:
    def test_prediction_distribution_chart(self):
        ms = _minimal_model_state()
        fig = charts.prediction_distribution_chart(ms, predicted_min=5.0)
        _assert_figure(fig)

    def test_prediction_distribution_chart_single_sample(self):
        # Edge case: only one historical observation
        ms = _minimal_model_state(n=1)
        fig = charts.prediction_distribution_chart(ms, predicted_min=3.0)
        _assert_figure(fig)


# ── convergence_rate_chart ────────────────────────────────────────────────────

class TestConvergenceRateChart:
    def test_convergence_rate_chart_empty(self):
        empty = pd.DataFrame(columns=["date", "convergence_rate", "n_pre_alerts"])
        fig = charts.convergence_rate_chart(empty, AREA)
        _assert_figure(fig)

    def test_convergence_rate_chart_data(self):
        conv_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-15", "2024-03-16", "2024-03-17"]),
            "convergence_rate": [1.0, 0.5, 0.75],
            "n_pre_alerts": [4, 2, 4],
        })
        fig = charts.convergence_rate_chart(conv_df, AREA)
        _assert_figure(fig)

    def test_convergence_rate_chart_has_two_traces_for_data(self):
        conv_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-15", "2024-03-16"]),
            "convergence_rate": [1.0, 0.5],
            "n_pre_alerts": [3, 2],
        })
        fig = charts.convergence_rate_chart(conv_df, AREA)
        # Expects raw-data scatter + smoothed line = 2 traces
        assert len(fig.data) == 2


# ── high_risk_windows_chart ───────────────────────────────────────────────────

class TestHighRiskWindowsChart:
    def test_high_risk_windows_chart_empty(self):
        empty = pd.DataFrame(columns=["label", "day_of_week", "hour", "count"])
        fig = charts.high_risk_windows_chart(empty, AREA)
        _assert_figure(fig)

    def test_high_risk_windows_chart_data(self):
        risk_df = pd.DataFrame({
            "label": ["Monday 08:00–09:00", "Tuesday 10:00–11:00"],
            "day_of_week": [0, 1],
            "hour": [8, 10],
            "count": [25, 18],
        })
        fig = charts.high_risk_windows_chart(risk_df, AREA)
        _assert_figure(fig)


# ── daily_pre_alert_siren_chart ───────────────────────────────────────────────

class TestDailyPreAlertSirenChart:
    def test_daily_pre_alert_siren_chart_empty(self):
        empty = pd.DataFrame(columns=["parsed_date", "type", "count"])
        fig = charts.daily_pre_alert_siren_chart(empty)
        _assert_figure(fig)

    def test_daily_pre_alert_siren_chart_data(self, paired_df):
        import transforms as T
        daily = T.daily_pre_alert_siren_counts(paired_df)
        fig = charts.daily_pre_alert_siren_chart(daily)
        _assert_figure(fig)


# ── timeline_chart ────────────────────────────────────────────────────────────

class TestTimelineChart:
    def _make_weekly_df(self):
        return pd.DataFrame({
            "week_start": pd.to_datetime(["2024-03-11", "2024-03-18", "2024-03-11"]),
            "category_en": [
                "Rocket / Missile Fire",
                "Rocket / Missile Fire",
                "Pre-Alert",
            ],
            "count": [5, 3, 2],
        })

    def test_timeline_chart(self):
        weekly = self._make_weekly_df()
        fig = charts.timeline_chart(weekly)
        _assert_figure(fig)

    def test_timeline_chart_empty(self):
        empty = pd.DataFrame(columns=["week_start", "category_en", "count"])
        fig = charts.timeline_chart(empty)
        _assert_figure(fig)


# ── interactive_risk_windows_chart ────────────────────────────────────────────

class TestInteractiveRiskWindowsChart:
    def _make_siren_df(self):
        rows = []
        base = datetime(2024, 3, 15, 8, 0)
        for i in range(20):
            t = base + pd.Timedelta(hours=i * 3)
            rows.append(_make_row(t, CAT_SIREN))
        # Mix in a pre-alert (should be ignored by the chart internals)
        rows.append(_make_row(base, CAT_PRE_ALERT))
        return pd.DataFrame(rows)

    def test_interactive_risk_windows_chart(self):
        df = self._make_siren_df()
        fig = charts.interactive_risk_windows_chart(df)
        _assert_figure(fig)

    def test_interactive_risk_windows_chart_21_traces(self):
        # 3 window sizes × 7 days = 21 Bar traces
        df = self._make_siren_df()
        fig = charts.interactive_risk_windows_chart(df)
        assert len(fig.data) == 21

    def test_interactive_risk_windows_chart_empty(self):
        # Even with an empty DataFrame (no siren rows) the function should
        # return a Figure without raising.
        df = pd.DataFrame(columns=[
            "date", "alertDate", "category", "data", "category_desc",
            "parsed_date", "parsed_alertDate", "hour", "day_of_week",
            "category_en", "location_en",
        ])
        # Cast dtypes so .isin() and .dropna() work
        df["category"] = pd.array([], dtype=int)
        df["hour"] = pd.array([], dtype=float)
        df["day_of_week"] = pd.array([], dtype=float)
        fig = charts.interactive_risk_windows_chart(df)
        _assert_figure(fig)
        assert len(fig.data) == 21


class TestComparisonCharts:
    def _make_conv_df(self, n=10):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "date": dates,
            "convergence_rate": [0.5] * n,
            "n_pre_alerts": [2] * n,
        })

    def test_comparison_convergence_chart_two_areas(self):
        conv_data = {
            "Area A": self._make_conv_df(),
            "Area B": self._make_conv_df(),
        }
        fig = charts.comparison_convergence_chart(conv_data)
        _assert_figure(fig)
        # 2 areas × 2 traces each (raw dots + rolling line)
        assert len(fig.data) == 4

    def test_comparison_convergence_chart_five_areas(self):
        conv_data = {f"Area {c}": self._make_conv_df() for c in "ABCDE"}
        fig = charts.comparison_convergence_chart(conv_data)
        _assert_figure(fig)
        assert len(fig.data) == 10

    def test_comparison_convergence_chart_empty_data(self):
        empty = pd.DataFrame(columns=["date", "convergence_rate", "n_pre_alerts"])
        fig = charts.comparison_convergence_chart({"Area A": empty, "Area B": empty})
        _assert_figure(fig)  # returns _empty_figure, still a Figure

    def test_comparison_summary_chart_basic(self):
        summary_df = pd.DataFrame([
            {"area": "Area A", "metric": "Convergence %", "value": 60},
            {"area": "Area A", "metric": "Avg Pre→Siren (min)", "value": 4.5},
            {"area": "Area B", "metric": "Convergence %", "value": 40},
            {"area": "Area B", "metric": "Avg Pre→Siren (min)", "value": 7.2},
        ])
        fig = charts.comparison_summary_chart(summary_df)
        _assert_figure(fig)
        assert len(fig.data) == 2  # one bar group per area

    def test_comparison_summary_chart_empty(self):
        fig = charts.comparison_summary_chart(pd.DataFrame())
        _assert_figure(fig)
