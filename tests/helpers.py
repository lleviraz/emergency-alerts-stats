"""
Shared test helpers (constants and row-builder) used by conftest and test files.
Kept separate from conftest.py so that test modules can import from here
without triggering pytest's special conftest import resolution.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

AREA = "Test Area"

# Category constants (mirroring transforms.CATEGORY_MAP)
CAT_SIREN = 1       # Rocket / Missile Fire
CAT_PRE_ALERT = 14  # Pre-Alert
CAT_ALL_CLEAR = 13  # All Clear
CAT_DRILL = 15      # Drill — Rockets

CATEGORY_DESC = {
    1:  "Rocket / Missile Fire",
    14: "Pre-Alert",
    13: "All Clear",
    15: "Drill — Rockets",
}

CATEGORY_EN = {
    1:  "Rocket / Missile Fire",
    14: "Pre-Alert",
    13: "All Clear",
    15: "Drill — Rockets",
}


def _make_row(t: datetime, category: int, location: str = AREA) -> dict:
    """
    Build a dict with all dashboard columns correctly filled for timestamp *t*.

    Columns produced:
      date            str  DD.MM.YYYY
      alertDate       str  ISO 8601
      category        int
      data            str  (location)
      category_desc   str
      parsed_date     pd.Timestamp  (date-only, midnight)
      parsed_alertDate pd.Timestamp (with time)
      hour            int
      day_of_week     int
      category_en     str
      location_en     str
    """
    ts = pd.Timestamp(t)
    return {
        "date":              ts.strftime("%d.%m.%Y"),
        "alertDate":         ts.isoformat(),
        "category":          category,
        "data":              location,
        "category_desc":     CATEGORY_DESC.get(category, f"Unknown ({category})"),
        "parsed_date":       pd.Timestamp(ts.date()),
        "parsed_alertDate":  ts,
        "hour":              ts.hour,
        "day_of_week":       ts.dayofweek,
        "category_en":       CATEGORY_EN.get(category, f"Unknown ({category})"),
        "location_en":       location,
    }
