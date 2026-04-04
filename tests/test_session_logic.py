"""
Tests for the refresh-gate and session-isolation logic that lives in dashboard.py.

dashboard.py is a Streamlit app and cannot be cleanly imported, so we replicate
the pure decision functions here as exact copies.  If you change the logic in
dashboard.py you MUST update the replica below — the tests will then catch the
regression immediately.
"""

import pytest

# ── Pure replicas of dashboard.py decision logic ──────────────────────────────

COOLDOWN_MINUTES = 60   # must match dashboard.COOLDOWN_MINUTES
AUTO_REFRESH_HOURS = 4  # must match dashboard.AUTO_REFRESH_HOURS


def can_train(no_model: bool, is_local: bool, age_seconds: float | None) -> bool:
    """
    Replica of the _can_train expression in dashboard.py sidebar.

    Always True when:
      - session has no model for the current area (avoids the stuck-state bug)
      - running in local dev mode
      - area was never trained in this server run
      - cooldown period has elapsed
    """
    return (
        no_model
        or is_local
        or age_seconds is None
        or age_seconds >= COOLDOWN_MINUTES * 60
    )


def can_refresh(no_data: bool, is_local: bool, age_seconds: float | None) -> bool:
    """
    Replica of the _can_refresh expression in dashboard.py sidebar.

    Always True when:
      - session has no data (user would be permanently stuck otherwise)
      - running in local dev mode
      - data has never been downloaded in this server run
      - cooldown period has elapsed
    """
    return (
        no_data
        or is_local
        or age_seconds is None
        or age_seconds >= COOLDOWN_MINUTES * 60
    )


def should_auto_refresh(age_hours: float | None,
                        threshold: int = AUTO_REFRESH_HOURS) -> bool:
    """Replica of the auto-refresh gate in dashboard.py."""
    return age_hours is not None and age_hours >= threshold


# ── can_refresh ───────────────────────────────────────────────────────────────

class TestCanRefresh:
    """
    The refresh button gate must NEVER trap a user with no data loaded.
    Regression guard for the F5 / new-tab stuck-state bug.
    """

    # ── the critical case: empty session ────────────────────────────────────

    def test_no_data_always_enabled_fresh_session(self):
        """Empty session with cooldown active → button must be enabled."""
        age = 5 * 60  # 5 min — well within cooldown
        assert can_refresh(no_data=True, is_local=False, age_seconds=age) is True

    def test_no_data_enabled_one_second_after_download(self):
        """Even 1 s after a shared download, an empty session can reload."""
        assert can_refresh(no_data=True, is_local=False, age_seconds=1) is True

    def test_no_data_enabled_with_zero_age(self):
        assert can_refresh(no_data=True, is_local=False, age_seconds=0) is True

    def test_no_data_enabled_regardless_of_local_flag(self):
        """no_data trumps everything — local mode irrelevant here."""
        assert can_refresh(no_data=True, is_local=True, age_seconds=10) is True
        assert can_refresh(no_data=True, is_local=False, age_seconds=10) is True

    # ── local dev mode ───────────────────────────────────────────────────────

    def test_local_mode_always_enabled_with_data(self):
        """Local dev bypasses cooldown even when session has data."""
        age = 10 * 60  # 10 min — inside cooldown
        assert can_refresh(no_data=False, is_local=True, age_seconds=age) is True

    def test_local_mode_enabled_at_zero_age(self):
        assert can_refresh(no_data=False, is_local=True, age_seconds=0) is True

    # ── cloud mode with data present ─────────────────────────────────────────

    def test_cloud_blocked_during_cooldown(self):
        """Cloud + data + recent download → blocked."""
        age = 10 * 60  # 10 minutes
        assert can_refresh(no_data=False, is_local=False, age_seconds=age) is False

    def test_cloud_blocked_one_second_before_cooldown_ends(self):
        age = COOLDOWN_MINUTES * 60 - 1
        assert can_refresh(no_data=False, is_local=False, age_seconds=age) is False

    def test_cloud_enabled_exactly_at_cooldown(self):
        age = COOLDOWN_MINUTES * 60
        assert can_refresh(no_data=False, is_local=False, age_seconds=age) is True

    def test_cloud_enabled_after_cooldown(self):
        age = COOLDOWN_MINUTES * 60 + 300  # 5 min past threshold
        assert can_refresh(no_data=False, is_local=False, age_seconds=age) is True

    def test_never_downloaded_always_enabled(self):
        """age_seconds=None means first ever run — must be enabled."""
        assert can_refresh(no_data=False, is_local=False, age_seconds=None) is True


# ── can_train ─────────────────────────────────────────────────────────────────

class TestCanTrain:
    """
    The train button must never trap a session that has no model.
    Regression guard for the F5 / new-tab stuck-state bug on model training.
    """

    def test_no_model_always_enabled(self):
        """Session with no model must be able to train regardless of cooldown."""
        age = 5 * 60  # 5 min — well within cooldown
        assert can_train(no_model=True, is_local=False, age_seconds=age) is True

    def test_no_model_enabled_one_second_after_train(self):
        assert can_train(no_model=True, is_local=False, age_seconds=1) is True

    def test_no_model_enabled_at_zero_age(self):
        assert can_train(no_model=True, is_local=False, age_seconds=0) is True

    def test_local_mode_always_enabled_with_model(self):
        age = 10 * 60
        assert can_train(no_model=False, is_local=True, age_seconds=age) is True

    def test_cloud_blocked_during_cooldown_when_model_exists(self):
        age = 10 * 60
        assert can_train(no_model=False, is_local=False, age_seconds=age) is False

    def test_cloud_blocked_one_second_before_cooldown_ends(self):
        age = COOLDOWN_MINUTES * 60 - 1
        assert can_train(no_model=False, is_local=False, age_seconds=age) is False

    def test_cloud_enabled_exactly_at_cooldown(self):
        age = COOLDOWN_MINUTES * 60
        assert can_train(no_model=False, is_local=False, age_seconds=age) is True

    def test_never_trained_always_enabled(self):
        assert can_train(no_model=False, is_local=False, age_seconds=None) is True


# ── should_auto_refresh ───────────────────────────────────────────────────────

class TestAutoRefresh:
    """Auto-refresh should fire only when data is stale enough."""

    def test_fires_exactly_at_threshold(self):
        assert should_auto_refresh(age_hours=float(AUTO_REFRESH_HOURS)) is True

    def test_fires_above_threshold(self):
        assert should_auto_refresh(age_hours=AUTO_REFRESH_HOURS + 1.5) is True

    def test_does_not_fire_below_threshold(self):
        assert should_auto_refresh(age_hours=AUTO_REFRESH_HOURS - 0.1) is False

    def test_does_not_fire_when_age_unknown(self):
        """None means we don't know when data was last loaded — don't refresh."""
        assert should_auto_refresh(age_hours=None) is False

    def test_custom_threshold_fires(self):
        assert should_auto_refresh(age_hours=2.0, threshold=2) is True

    def test_custom_threshold_blocked(self):
        assert should_auto_refresh(age_hours=1.99, threshold=2) is False

    def test_zero_age_never_fires(self):
        assert should_auto_refresh(age_hours=0.0) is False
