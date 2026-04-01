"""
Shared pytest fixtures for the Israel Alerts Dashboard test suite.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Ensure the worktree root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.helpers import (
    AREA,
    CAT_ALL_CLEAR,
    CAT_DRILL,
    CAT_PRE_ALERT,
    CAT_SIREN,
    _make_row,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_time() -> datetime:
    """A fixed base time used to generate synthetic events."""
    return datetime(2024, 3, 15, 10, 0, 0)


@pytest.fixture
def paired_df(base_time: datetime) -> pd.DataFrame:
    """
    10 pre-alert (cat 14) + 10 siren (cat 1) pairs.
    Each siren arrives exactly 5 minutes after its pre-alert.
    Pairs are spaced 3 hours apart.
    Used to test train_area_model success (all pairs match).
    """
    rows = []
    for i in range(10):
        t_pre = base_time + pd.Timedelta(hours=3 * i)
        t_sir = t_pre + pd.Timedelta(minutes=5)
        rows.append(_make_row(t_pre, CAT_PRE_ALERT))
        rows.append(_make_row(t_sir, CAT_SIREN))
    return pd.DataFrame(rows)


@pytest.fixture
def mixed_df(base_time: datetime) -> pd.DataFrame:
    """
    8 paired (pre-alert -> siren 5 min later, 3 h apart)
    + 4 standalone pre-alerts with no following siren.

    The 4 standalone pre-alerts are placed so that no siren falls
    within 15 minutes of them (they are placed after all paired events,
    spaced 3 h apart, with no matching siren row).

    This gives the classifier both positive (1) and negative (0) labels.
    """
    rows = []
    # 8 pairs
    for i in range(8):
        t_pre = base_time + pd.Timedelta(hours=3 * i)
        t_sir = t_pre + pd.Timedelta(minutes=5)
        rows.append(_make_row(t_pre, CAT_PRE_ALERT))
        rows.append(_make_row(t_sir, CAT_SIREN))
    # 4 standalone pre-alerts (no siren within window)
    offset_start = pd.Timedelta(hours=3 * 8)
    for j in range(4):
        t_pre = base_time + offset_start + pd.Timedelta(hours=3 * j)
        rows.append(_make_row(t_pre, CAT_PRE_ALERT))
    return pd.DataFrame(rows)


@pytest.fixture
def no_match_df(base_time: datetime) -> pd.DataFrame:
    """
    5 pre-alerts each followed by a siren 20 minutes later (outside the
    default 15-minute matching window).  Result: zero matched pairs.
    """
    rows = []
    for i in range(5):
        t_pre = base_time + pd.Timedelta(hours=3 * i)
        t_sir = t_pre + pd.Timedelta(minutes=20)   # outside 15-min window
        rows.append(_make_row(t_pre, CAT_PRE_ALERT))
        rows.append(_make_row(t_sir, CAT_SIREN))
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv_bytes() -> bytes:
    """
    Minimal CSV bytes with 3 rows:
      - one siren    (cat 1)
      - one pre-alert (cat 14)
      - one all-clear (cat 13)
    Date format: DD.MM.YYYY in the *date* column; ISO in *alertDate*.
    """
    lines = [
        "date,alertDate,category,data,category_desc",
        "15.03.2024,2024-03-15T10:00:00,1,Test Area,Rocket / Missile Fire",
        "15.03.2024,2024-03-15T10:05:00,14,Test Area,Pre-Alert",
        "15.03.2024,2024-03-15T10:20:00,13,Test Area,All Clear",
    ]
    return "\n".join(lines).encode("utf-8")
