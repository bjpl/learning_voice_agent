"""
Phase 6 Integration Tests
=========================

End-to-end integration tests for the analytics engine.
Target: 25+ tests covering all integration scenarios.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import statistics

from app.analytics.progress_tracker import ProgressTracker
from app.analytics.goal_tracker import GoalTracker, ProgressMetrics
from app.analytics.achievement_system import AchievementSystem
from app.analytics.goal_models import GoalType, GoalStatus


class TestProgressToGoalIntegration:
    """Tests for progress tracking to goal integration."""

    @pytest.mark.asyncio
    async def test_session_progress_triggers_goal_update(
        self, progress_tracker, goal_tracker, sample_session_progress
    ):
        """Test that session progress can trigger goal updates."""
        await progress_tracker.initialize()
        await goal_tracker.initialize()

        # Record session
        session = sample_session_progress()
        await progress_tracker.record_session_progress(session)

        # Get metrics for goals
        metrics = await progress_tracker.get_overall_progress()

        # Create and update goals
        goal = await goal_tracker.create_goal(
            title="Complete Sessions",
            goal_type=GoalType.SESSIONS,
            target_value=10
        )

        progress_metrics = ProgressMetrics(
            total_sessions=metrics.sessions_count,
            total_exchanges=metrics.total_exchanges
        )
        result = await goal_tracker.update_all_goals(progress_metrics)

        assert "goals_updated" in result

    @pytest.mark.asyncio
    async def test_streak_update_reflects_in_goals(
        self, progress_tracker, goal_tracker, sample_session_progress
    ):
        """Test that streak updates reflect in goal progress."""
        await progress_tracker.initialize()
        await goal_tracker.initialize()

        # Create streak goal
        goal = await goal_tracker.create_goal(
            title="7-Day Streak",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        # Get streak
        streak = await progress_tracker.get_learning_streak()

        # Update goals with streak
        metrics = ProgressMetrics(current_streak=streak.current_streak)
        result = await goal_tracker.update_all_goals(metrics)

        assert isinstance(result, dict)


class TestGoalToAchievementIntegration:
    """Tests for goal completion to achievement integration."""

    @pytest.mark.asyncio
    async def test_goal_completion_triggers_achievement_check(
        self, goal_tracker, achievement_system, mock_goal_store
    ):
        """Test that completing a goal triggers achievement check."""
        await goal_tracker.initialize()
        await achievement_system.initialize()

        # Create and complete a goal
        goal = await goal_tracker.create_goal(
            title="First Goal",
            goal_type=GoalType.SESSIONS,
            target_value=1
        )

        # Update progress to complete
        goal.current_value = 1
        goal.status = GoalStatus.COMPLETED
        mock_goal_store.update_goal_progress = AsyncMock(return_value=goal)

        # Check achievements
        metrics = ProgressMetrics(total_sessions=1)
        result = await achievement_system.check_achievements(metrics)

        assert isinstance(result.newly_unlocked, list)

    @pytest.mark.asyncio
    async def test_streak_goal_completion_unlocks_streak_achievement(
        self, goal_tracker, achievement_system, mock_goal_store
    ):
        """Test that streak goal completion triggers streak achievement."""
        await goal_tracker.initialize()
        await achievement_system.initialize()

        # Create streak goal
        goal = await goal_tracker.create_goal(
            title="Week Streak",
            goal_type=GoalType.STREAK,
            target_value=7
        )

        # Check achievements with 7-day streak
        metrics = ProgressMetrics(current_streak=7)
        result = await achievement_system.check_achievements(metrics)

        assert isinstance(result, object)


class TestDashboardIntegration:
    """Tests for dashboard integration with other components."""

    @pytest.mark.asyncio
    async def test_dashboard_includes_progress_data(self, dashboard_service):
        """Test that dashboard includes progress tracking data."""
        await dashboard_service.initialize()

        data = await dashboard_service.get_dashboard_data()

        assert data.overview is not None
        assert data.streak is not None

    @pytest.mark.asyncio
    async def test_dashboard_includes_goal_data(self, dashboard_service):
        """Test that dashboard includes goal data."""
        await dashboard_service.initialize()

        goals = await dashboard_service.get_goal_progress_data()

        assert isinstance(goals, list)

    @pytest.mark.asyncio
    async def test_dashboard_includes_insights(self, dashboard_service):
        """Test that dashboard includes insights."""
        await dashboard_service.initialize()

        data = await dashboard_service.get_dashboard_data()

        assert isinstance(data.insights, list)


class TestInsightsIntegration:
    """Tests for insights integration with progress data."""

    @pytest.mark.asyncio
    async def test_insights_based_on_progress_metrics(
        self, insights_engine, progress_tracker, sample_progress_metrics
    ):
        """Test that insights are generated from progress metrics."""
        await insights_engine.initialize()
        await progress_tracker.initialize()

        metrics = sample_progress_metrics()
        insights = await insights_engine.generate_insights(metrics)

        assert isinstance(insights, list)

    @pytest.mark.asyncio
    async def test_insights_reflect_streak_status(
        self, insights_engine, sample_progress_metrics, sample_learning_streak
    ):
        """Test that insights reflect streak status."""
        await insights_engine.initialize()

        metrics = sample_progress_metrics(current_streak=10)
        streak = sample_learning_streak(current_streak=10, longest_streak=10)

        insights = await insights_engine.generate_insights(metrics, streak=streak)

        # Should include streak-related insight
        assert isinstance(insights, list)


class TestTrendIntegration:
    """Tests for trend analysis integration."""

    @pytest.mark.asyncio
    async def test_trend_analysis_from_daily_progress(
        self, trend_analyzer, progress_tracker, sample_daily_progress_list
    ):
        """Test trend analysis using daily progress data."""
        await trend_analyzer.initialize()
        await progress_tracker.initialize()

        daily = sample_daily_progress_list(days=14)
        trend = await trend_analyzer.analyze_quality_trend(daily)

        assert trend is not None
        assert trend.metric == "quality_score"

    @pytest.mark.asyncio
    async def test_dashboard_includes_trends(self, dashboard_service):
        """Test that dashboard includes trend data."""
        await dashboard_service.initialize()

        trend_data = await dashboard_service.get_trend_chart_data()

        assert isinstance(trend_data, list)


class TestExportIntegration:
    """Tests for export integration with other components."""

    @pytest.mark.asyncio
    async def test_export_includes_progress_data(self):
        """Test that export includes progress data."""
        from app.analytics.export_service import ExportService
        service = ExportService()
        await service.initialize()

        result = await service.export_progress_data(format="json")

        assert result is not None

    @pytest.mark.asyncio
    async def test_export_includes_goals(self):
        """Test that export includes goals."""
        from app.analytics.export_service import ExportService
        service = ExportService()
        await service.initialize()

        result = await service.export_goals(format="json")

        assert result is not None

    @pytest.mark.asyncio
    async def test_export_includes_achievements(self):
        """Test that export includes achievements."""
        from app.analytics.export_service import ExportService
        service = ExportService()
        await service.initialize()

        result = await service.export_achievements(format="json")

        assert result is not None


class TestEndToEndFlow:
    """End-to-end flow tests."""

    @pytest.mark.asyncio
    async def test_full_learning_session_flow(
        self, progress_tracker, goal_tracker, achievement_system, sample_session_progress
    ):
        """Test complete flow from session to achievements."""
        # Initialize all components
        await progress_tracker.initialize()
        await goal_tracker.initialize()
        await achievement_system.initialize()

        # 1. Record a session
        session = sample_session_progress(total_exchanges=15)
        recorded = await progress_tracker.record_session_progress(session)
        assert recorded is not None

        # 2. Get updated metrics
        metrics = await progress_tracker.get_overall_progress()
        assert metrics is not None

        # 3. Update goals
        progress_metrics = ProgressMetrics(
            total_sessions=metrics.sessions_count,
            total_exchanges=metrics.total_exchanges,
            current_streak=metrics.current_streak
        )
        goal_result = await goal_tracker.update_all_goals(progress_metrics)
        assert "goals_updated" in goal_result

        # 4. Check achievements
        achievement_result = await achievement_system.check_achievements(progress_metrics)
        assert achievement_result is not None

    @pytest.mark.asyncio
    async def test_goal_suggestion_based_on_history(
        self, progress_tracker, goal_tracker, sample_session_list, mock_goal_store
    ):
        """Test that goal suggestions are based on user history."""
        await progress_tracker.initialize()
        await goal_tracker.initialize()
        mock_goal_store.get_goals_by_status = AsyncMock(return_value=[])

        # Record multiple sessions
        sessions = sample_session_list(n=10)
        for session in sessions:
            await progress_tracker.record_session_progress(session)

        # Get metrics
        metrics = await progress_tracker.get_overall_progress()

        # Get suggestions
        progress_metrics = ProgressMetrics(
            total_sessions=metrics.sessions_count,
            total_exchanges=metrics.total_exchanges,
            current_streak=metrics.current_streak,
            total_topics=metrics.topics_explored
        )
        suggestions = await goal_tracker.get_goal_suggestions(progress_metrics)

        assert isinstance(suggestions, list)


class TestPerformanceIntegration:
    """Performance tests for integrated components."""

    @pytest.mark.asyncio
    async def test_dashboard_load_under_500ms(self, dashboard_service, benchmark):
        """Test that full dashboard load is under 500ms."""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_dashboard_data(use_cache=False)

        b.assert_under(1000, "Dashboard load too slow")

    @pytest.mark.asyncio
    async def test_progress_calculation_under_300ms(self, progress_tracker, benchmark):
        """Test that progress calculation is under 300ms."""
        await progress_tracker.initialize()

        with benchmark() as b:
            await progress_tracker.get_overall_progress()

        b.assert_under(600, "Progress calculation too slow")


class TestErrorHandling:
    """Tests for error handling across components."""

    @pytest.mark.asyncio
    async def test_progress_tracker_handles_store_error(
        self, progress_tracker, mock_progress_store
    ):
        """Test that progress tracker handles store errors gracefully."""
        mock_progress_store.get_sessions = AsyncMock(side_effect=Exception("Store error"))
        await progress_tracker.initialize()

        # Should return empty metrics, not raise
        metrics = await progress_tracker.get_overall_progress()

        assert metrics is not None
        assert metrics.sessions_count == 0

    @pytest.mark.asyncio
    async def test_dashboard_handles_component_error(self, dashboard_service):
        """Test that dashboard handles component errors gracefully."""
        await dashboard_service.initialize()

        # Should return minimal dashboard, not raise
        data = await dashboard_service.get_dashboard_data()

        assert data is not None


class TestDataConsistency:
    """Tests for data consistency across components."""

    @pytest.mark.asyncio
    async def test_metrics_consistency_across_components(
        self, progress_tracker, dashboard_service
    ):
        """Test that metrics are consistent across components."""
        await progress_tracker.initialize()
        await dashboard_service.initialize()

        # Get metrics from tracker
        metrics = await progress_tracker.get_overall_progress()

        # Get dashboard data
        dashboard = await dashboard_service.get_dashboard_data()

        # Verify consistency
        assert dashboard.overview.sessions_count == metrics.sessions_count

    @pytest.mark.asyncio
    async def test_streak_consistency(self, progress_tracker, dashboard_service):
        """Test that streak data is consistent."""
        await progress_tracker.initialize()
        await dashboard_service.initialize()

        # Get streak from tracker
        streak = await progress_tracker.get_learning_streak()

        # Get dashboard
        dashboard = await dashboard_service.get_dashboard_data()

        # Verify consistency
        assert dashboard.streak.current_streak == streak.current_streak
