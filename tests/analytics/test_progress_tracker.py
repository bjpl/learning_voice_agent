"""
Test Suite for Progress Tracker
================================

Comprehensive tests for progress tracking functionality.
Target: 30+ tests covering all progress tracking features.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import statistics

from app.analytics.progress_tracker import ProgressTracker
from app.analytics.progress_models import (
    ProgressMetrics,
    LearningStreak,
    TopicMastery,
    SessionProgress,
    DailyProgress,
    WeeklyProgress,
    MonthlyProgress,
    ProgressSnapshot,
    TrendDirection,
    ProgressLevel,
)


class TestProgressTrackerInitialization:
    """Tests for ProgressTracker initialization."""

    @pytest.mark.asyncio
    async def test_initializes_successfully(self, progress_tracker):
        """Test that progress tracker initializes without errors."""
        await progress_tracker.initialize()
        assert progress_tracker._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, progress_tracker):
        """Test that multiple initialization calls are safe."""
        await progress_tracker.initialize()
        await progress_tracker.initialize()
        assert progress_tracker._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_calls_store_initialize(self, progress_tracker, mock_progress_store):
        """Test that initialization initializes the stores."""
        await progress_tracker.initialize()
        mock_progress_store.initialize.assert_called_once()


class TestSessionProgressRecording:
    """Tests for recording session progress."""

    @pytest.mark.asyncio
    async def test_record_session_progress_stores_session(
        self, progress_tracker, mock_progress_store, sample_session_progress
    ):
        """Test that session progress is stored correctly."""
        session = sample_session_progress()
        await progress_tracker.initialize()

        result = await progress_tracker.record_session_progress(session)

        assert result.session_id == session.session_id
        mock_progress_store.store_session_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_session_finalizes_if_no_end_time(
        self, progress_tracker, sample_session_progress
    ):
        """Test that session is finalized if end_time is not set."""
        session = sample_session_progress()
        session.end_time = None
        await progress_tracker.initialize()

        result = await progress_tracker.record_session_progress(session)

        assert result.end_time is not None

    @pytest.mark.asyncio
    async def test_record_session_updates_streak(
        self, progress_tracker, mock_progress_store, sample_session_progress
    ):
        """Test that recording a session updates the streak."""
        session = sample_session_progress()
        await progress_tracker.initialize()

        await progress_tracker.record_session_progress(session)

        mock_progress_store.store_streak.assert_called()

    @pytest.mark.asyncio
    async def test_record_session_updates_topic_mastery(
        self, progress_tracker, mock_progress_store, sample_session_progress
    ):
        """Test that recording a session updates topic mastery."""
        session = sample_session_progress(topics=["machine learning", "python"])
        await progress_tracker.initialize()

        await progress_tracker.record_session_progress(session)

        # Should be called once for each topic
        assert mock_progress_store.store_topic_mastery.call_count >= 2

    @pytest.mark.asyncio
    async def test_record_session_invalidates_metrics_cache(
        self, progress_tracker, sample_session_progress
    ):
        """Test that recording session invalidates metrics cache."""
        session = sample_session_progress(user_id="user123")
        await progress_tracker.initialize()

        # Prime the cache
        progress_tracker._metrics_cache["user123"] = (datetime.utcnow(), ProgressMetrics())

        await progress_tracker.record_session_progress(session)

        assert "user123" not in progress_tracker._metrics_cache


class TestOverallProgress:
    """Tests for getting overall progress metrics."""

    @pytest.mark.asyncio
    async def test_get_overall_progress_aggregates_sessions(
        self, progress_tracker, mock_progress_store, sample_session_list
    ):
        """Test that overall progress aggregates session data correctly."""
        sessions = sample_session_list(n=5, avg_quality=0.8)
        mock_progress_store.get_sessions = AsyncMock(return_value=sessions)
        await progress_tracker.initialize()

        metrics = await progress_tracker.get_overall_progress()

        assert metrics.sessions_count == 5
        assert metrics.total_exchanges == sum(s.total_exchanges for s in sessions)

    @pytest.mark.asyncio
    async def test_get_overall_progress_calculates_averages(
        self, progress_tracker, mock_progress_store, sample_session_list
    ):
        """Test that quality averages are calculated correctly."""
        sessions = sample_session_list(n=10, avg_quality=0.75)
        mock_progress_store.get_sessions = AsyncMock(return_value=sessions)
        await progress_tracker.initialize()

        metrics = await progress_tracker.get_overall_progress()

        expected_avg = statistics.mean(s.avg_quality_score for s in sessions)
        assert abs(metrics.avg_quality_score - expected_avg) < 0.01

    @pytest.mark.asyncio
    async def test_get_overall_progress_uses_cache(
        self, progress_tracker, mock_progress_store
    ):
        """Test that cached metrics are returned if valid."""
        cached_metrics = ProgressMetrics(sessions_count=100)
        progress_tracker._metrics_cache["global"] = (datetime.utcnow(), cached_metrics)
        await progress_tracker.initialize()

        metrics = await progress_tracker.get_overall_progress()

        assert metrics.sessions_count == 100
        mock_progress_store.get_sessions.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_overall_progress_ignores_stale_cache(
        self, progress_tracker, mock_progress_store, analytics_config_fixture
    ):
        """Test that stale cache is ignored."""
        stale_time = datetime.utcnow() - timedelta(seconds=analytics_config_fixture.progress.cache_ttl_seconds + 1)
        cached_metrics = ProgressMetrics(sessions_count=999)
        progress_tracker._metrics_cache["global"] = (stale_time, cached_metrics)
        mock_progress_store.get_sessions = AsyncMock(return_value=[])
        await progress_tracker.initialize()

        metrics = await progress_tracker.get_overall_progress()

        assert metrics.sessions_count == 0  # Fresh calculation

    @pytest.mark.asyncio
    async def test_get_overall_progress_returns_empty_on_error(
        self, progress_tracker, mock_progress_store
    ):
        """Test that empty metrics are returned on error."""
        mock_progress_store.get_sessions = AsyncMock(side_effect=Exception("DB error"))
        await progress_tracker.initialize()

        metrics = await progress_tracker.get_overall_progress()

        assert metrics.sessions_count == 0


class TestLearningStreak:
    """Tests for learning streak functionality."""

    @pytest.mark.asyncio
    async def test_get_learning_streak_returns_cached(
        self, progress_tracker, sample_learning_streak
    ):
        """Test that cached streak is returned."""
        cached_streak = sample_learning_streak(current_streak=10)
        progress_tracker._streak_cache["global"] = cached_streak
        await progress_tracker.initialize()

        streak = await progress_tracker.get_learning_streak()

        assert streak.current_streak == 10

    @pytest.mark.asyncio
    async def test_get_learning_streak_loads_from_store(
        self, progress_tracker, mock_progress_store, sample_learning_streak
    ):
        """Test that streak is loaded from store if not cached."""
        stored_streak = sample_learning_streak(current_streak=15)
        mock_progress_store.get_streak = AsyncMock(return_value=stored_streak)
        await progress_tracker.initialize()

        streak = await progress_tracker.get_learning_streak()

        assert streak.current_streak == 15

    @pytest.mark.asyncio
    async def test_get_learning_streak_creates_new_if_not_exists(
        self, progress_tracker, mock_progress_store
    ):
        """Test that a new streak is created if none exists."""
        mock_progress_store.get_streak = AsyncMock(return_value=None)
        await progress_tracker.initialize()

        streak = await progress_tracker.get_learning_streak()

        assert streak.current_streak == 0

    @pytest.mark.asyncio
    async def test_streak_updates_on_consecutive_days(self, sample_learning_streak):
        """Test that streak increments on consecutive days."""
        streak = sample_learning_streak(
            current_streak=5,
            last_active_date=date.today() - timedelta(days=1)
        )

        streak.update(date.today())

        assert streak.current_streak == 6

    @pytest.mark.asyncio
    async def test_streak_resets_on_gap(self, sample_learning_streak):
        """Test that streak resets when there's a gap."""
        streak = sample_learning_streak(
            current_streak=10,
            longest_streak=10,
            last_active_date=date.today() - timedelta(days=3)
        )

        result = streak.update(date.today())

        assert streak.current_streak == 1
        assert result is False  # Indicates streak was broken

    @pytest.mark.asyncio
    async def test_streak_preserves_longest(self, sample_learning_streak):
        """Test that longest streak is preserved after reset."""
        streak = sample_learning_streak(
            current_streak=15,
            longest_streak=15,
            last_active_date=date.today() - timedelta(days=5)
        )

        streak.update(date.today())

        assert streak.longest_streak >= 15


class TestTopicMastery:
    """Tests for topic mastery tracking."""

    @pytest.mark.asyncio
    async def test_get_topic_mastery_returns_cached(
        self, progress_tracker, sample_topic_mastery
    ):
        """Test that cached mastery is returned."""
        cached = sample_topic_mastery(topic="python", mastery_score=0.9)
        progress_tracker._mastery_cache["global"] = {"python": cached}
        await progress_tracker.initialize()

        mastery = await progress_tracker.get_topic_mastery(None, "python")

        assert mastery.mastery_score == 0.9

    @pytest.mark.asyncio
    async def test_get_topic_mastery_loads_from_store(
        self, progress_tracker, mock_progress_store, sample_topic_mastery
    ):
        """Test that mastery is loaded from store."""
        stored = sample_topic_mastery(topic="ai", mastery_score=0.75)
        mock_progress_store.get_topic_mastery = AsyncMock(return_value=stored)
        await progress_tracker.initialize()

        mastery = await progress_tracker.get_topic_mastery(None, "ai")

        assert mastery.mastery_score == 0.75

    @pytest.mark.asyncio
    async def test_get_all_topic_mastery(
        self, progress_tracker, mock_progress_store, sample_topic_mastery
    ):
        """Test getting all topic mastery data."""
        masteries = [
            sample_topic_mastery(topic="python"),
            sample_topic_mastery(topic="javascript"),
        ]
        mock_progress_store.get_all_topic_mastery = AsyncMock(return_value=masteries)
        await progress_tracker.initialize()

        result = await progress_tracker.get_all_topic_mastery()

        assert len(result) == 2
        assert "python" in result
        assert "javascript" in result

    def test_topic_mastery_update_increases_interactions(self, sample_topic_mastery):
        """Test that mastery update increases interaction count."""
        mastery = sample_topic_mastery(total_interactions=5)

        mastery.update_mastery(0.8)

        assert mastery.total_interactions == 6

    def test_topic_mastery_update_changes_level(self, sample_topic_mastery):
        """Test that mastery level updates based on score and interactions."""
        mastery = sample_topic_mastery(
            mastery_score=0.5,
            total_interactions=25,
            level=ProgressLevel.INTERMEDIATE
        )

        # Simulate multiple high-quality interactions
        for _ in range(25):
            mastery.update_mastery(0.95, success=True)

        assert mastery.level in [ProgressLevel.ADVANCED, ProgressLevel.EXPERT]


class TestDailyProgress:
    """Tests for daily progress aggregation."""

    @pytest.mark.asyncio
    async def test_get_daily_progress_aggregates_sessions(
        self, progress_tracker, mock_progress_store, sample_session_list
    ):
        """Test that daily progress aggregates correctly."""
        today = date.today()
        sessions = sample_session_list(n=3)
        for s in sessions:
            s.start_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=10)
        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=sessions)
        await progress_tracker.initialize()

        progress = await progress_tracker.get_daily_progress(today)

        assert progress.total_sessions == 3
        assert progress.total_exchanges == sum(s.total_exchanges for s in sessions)

    @pytest.mark.asyncio
    async def test_get_daily_progress_calculates_quality_average(
        self, progress_tracker, mock_progress_store, sample_session_list
    ):
        """Test that daily quality average is calculated."""
        sessions = sample_session_list(n=5, avg_quality=0.8)
        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=sessions)
        await progress_tracker.initialize()

        progress = await progress_tracker.get_daily_progress(date.today())

        expected = statistics.mean(s.avg_quality_score for s in sessions)
        assert abs(progress.avg_quality_score - expected) < 0.1

    @pytest.mark.asyncio
    async def test_get_daily_progress_empty_day(
        self, progress_tracker, mock_progress_store
    ):
        """Test daily progress for a day with no sessions."""
        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=[])
        await progress_tracker.initialize()

        progress = await progress_tracker.get_daily_progress(date.today())

        assert progress.total_sessions == 0
        assert progress.total_exchanges == 0


class TestWeeklyProgress:
    """Tests for weekly progress aggregation."""

    @pytest.mark.asyncio
    async def test_get_weekly_progress_spans_seven_days(
        self, progress_tracker, mock_progress_store
    ):
        """Test that weekly progress covers 7 days."""
        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=[])
        await progress_tracker.initialize()

        monday = date.today() - timedelta(days=date.today().weekday())
        progress = await progress_tracker.get_weekly_progress(monday)

        assert progress.week_start == monday
        assert progress.week_end == monday + timedelta(days=6)

    @pytest.mark.asyncio
    async def test_get_weekly_progress_counts_active_days(
        self, progress_tracker, mock_progress_store, sample_session_progress
    ):
        """Test that active days are counted correctly."""
        # Return sessions only for specific days
        async def mock_sessions(target_date, user_id):
            if target_date.weekday() in [0, 2, 4]:  # Mon, Wed, Fri
                return [sample_session_progress()]
            return []

        mock_progress_store.get_sessions_for_date = AsyncMock(side_effect=mock_sessions)
        await progress_tracker.initialize()

        monday = date.today() - timedelta(days=date.today().weekday())
        progress = await progress_tracker.get_weekly_progress(monday)

        assert progress.active_days == 3


class TestMonthlyProgress:
    """Tests for monthly progress aggregation."""

    @pytest.mark.asyncio
    async def test_get_monthly_progress_correct_month(
        self, progress_tracker, mock_progress_store
    ):
        """Test that monthly progress is for correct month."""
        mock_progress_store.get_sessions_for_date = AsyncMock(return_value=[])
        await progress_tracker.initialize()

        progress = await progress_tracker.get_monthly_progress(2024, 6)

        assert progress.year == 2024
        assert progress.month == 6


class TestProgressSnapshot:
    """Tests for progress snapshots."""

    @pytest.mark.asyncio
    async def test_create_snapshot_captures_current_state(
        self, progress_tracker, mock_progress_store
    ):
        """Test that snapshot captures current progress state."""
        mock_progress_store.get_sessions = AsyncMock(return_value=[])
        mock_progress_store.get_all_topic_mastery = AsyncMock(return_value=[])
        await progress_tracker.initialize()

        snapshot = await progress_tracker.create_progress_snapshot()

        assert snapshot is not None
        assert snapshot.metrics is not None
        mock_progress_store.store_snapshot.assert_called_once()


class TestCacheManagement:
    """Tests for cache management."""

    @pytest.mark.asyncio
    async def test_clear_cache_removes_all_caches(self, progress_tracker):
        """Test that clear_cache removes all cached data."""
        progress_tracker._streak_cache["test"] = LearningStreak()
        progress_tracker._mastery_cache["test"] = {}
        progress_tracker._metrics_cache["test"] = (datetime.utcnow(), ProgressMetrics())
        await progress_tracker.initialize()

        progress_tracker.clear_cache()

        assert len(progress_tracker._streak_cache) == 0
        assert len(progress_tracker._mastery_cache) == 0
        assert len(progress_tracker._metrics_cache) == 0
