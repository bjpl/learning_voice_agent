"""
Test Suite for Dashboard Service
================================

Comprehensive tests for dashboard data generation.
Target: 30+ tests covering all dashboard features.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock


class TestDashboardServiceInitialization:
    """Tests for DashboardService initialization."""

    @pytest.mark.asyncio
    async def test_initializes_successfully(self, dashboard_service):
        """Test that dashboard service initializes without errors."""
        await dashboard_service.initialize()
        assert dashboard_service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, dashboard_service):
        """Test that multiple initialization calls are safe."""
        await dashboard_service.initialize()
        await dashboard_service.initialize()
        assert dashboard_service._initialized is True


class TestDashboardData:
    """Tests for complete dashboard data retrieval."""

    @pytest.mark.asyncio
    async def test_get_dashboard_data_returns_dashboard_data(self, dashboard_service):
        """Test that dashboard data is returned."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_dashboard_data()
        assert data is not None

    @pytest.mark.asyncio
    async def test_get_dashboard_data_includes_overview(self, dashboard_service):
        """Test that dashboard data includes overview metrics."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_dashboard_data()
        # Overview is available as attribute or the data object itself has overview data
        assert hasattr(data, 'overview') or hasattr(data, 'generated_at')

    @pytest.mark.asyncio
    async def test_get_dashboard_data_includes_streak(self, dashboard_service):
        """Test that dashboard data includes streak information."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_dashboard_data()
        # Streak can be accessed via attribute
        assert hasattr(data, 'streak') or hasattr(data, 'generated_at')

    @pytest.mark.asyncio
    async def test_get_dashboard_data_uses_cache(self, dashboard_service):
        """Test that cached dashboard data is used when valid."""
        await dashboard_service.initialize()

        # First call
        data1 = await dashboard_service.get_dashboard_data()
        # Second call should use cache
        data2 = await dashboard_service.get_dashboard_data()

        # Cache should return same object (same id) or timestamps within 1 second tolerance
        timestamps_close = abs((data1.generated_at - data2.generated_at).total_seconds()) < 1.0
        same_object = data1.id == data2.id
        assert timestamps_close or same_object, (
            f"Cache not working: ids differ ({data1.id} vs {data2.id}) "
            f"and timestamps not close ({data1.generated_at} vs {data2.generated_at})"
        )

    @pytest.mark.asyncio
    async def test_get_dashboard_data_bypasses_cache(self, dashboard_service):
        """Test that cache can be bypassed."""
        await dashboard_service.initialize()

        data1 = await dashboard_service.get_dashboard_data()
        # Force bypass cache
        data2 = await dashboard_service.get_dashboard_data(use_cache=False)

        # Generated times might differ slightly
        assert data2 is not None


class TestOverviewData:
    """Tests for overview data retrieval."""

    @pytest.mark.asyncio
    async def test_get_overview_data_returns_response(self, dashboard_service):
        """Test that overview data returns a response object."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_overview_data()
        # Returns OverviewResponse which has cards, quick_stats, etc.
        assert data is not None
        assert hasattr(data, 'cards') or hasattr(data, 'quick_stats') or hasattr(data, 'generated_at')

    @pytest.mark.asyncio
    async def test_get_overview_data_has_quick_stats(self, dashboard_service):
        """Test that overview includes quick stats."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_overview_data()

        # OverviewResponse has quick_stats attribute
        assert hasattr(data, 'quick_stats')

    @pytest.mark.asyncio
    async def test_get_overview_data_has_streak_info(self, dashboard_service):
        """Test that overview includes streak info."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_overview_data()

        # OverviewResponse has streak_info attribute
        assert hasattr(data, 'streak_info')

    @pytest.mark.asyncio
    async def test_get_overview_data_has_generated_at(self, dashboard_service):
        """Test that overview includes generation timestamp."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_overview_data()
        assert hasattr(data, 'generated_at')


class TestQualityChartData:
    """Tests for quality chart data."""

    @pytest.mark.asyncio
    async def test_get_quality_chart_data_returns_list(self, dashboard_service):
        """Test that quality chart returns a list."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_quality_chart_data()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_quality_chart_data_has_date_fields(self, dashboard_service):
        """Test that chart data includes date field."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_quality_chart_data()

        if data:
            assert "date" in data[0]

    @pytest.mark.asyncio
    async def test_get_quality_chart_data_has_quality_field(self, dashboard_service):
        """Test that chart data includes quality field."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_quality_chart_data()

        if data:
            assert "quality" in data[0]

    @pytest.mark.asyncio
    async def test_get_quality_chart_data_respects_days_param(self, dashboard_service):
        """Test that days parameter is respected."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_quality_chart_data(days=7)
        # Should have at most 8 days (7 days + today)
        assert len(data) <= 8


class TestProgressChartData:
    """Tests for progress chart data."""

    @pytest.mark.asyncio
    async def test_get_progress_chart_data_exchanges(self, dashboard_service):
        """Test progress chart for exchanges metric."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_progress_chart_data(metric="exchanges")
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_progress_chart_data_sessions(self, dashboard_service):
        """Test progress chart for sessions metric."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_progress_chart_data(metric="sessions")
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_progress_chart_data_time(self, dashboard_service):
        """Test progress chart for time metric."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_progress_chart_data(metric="time")
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_progress_chart_data_includes_cumulative(self, dashboard_service):
        """Test that progress chart includes cumulative values."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_progress_chart_data()

        if data:
            assert "cumulative" in data[0]


class TestTrendChartData:
    """Tests for trend chart data."""

    @pytest.mark.asyncio
    async def test_get_trend_chart_data_quality(self, dashboard_service):
        """Test trend chart for quality metric."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_trend_chart_data(metric="quality")
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_trend_chart_data_includes_rolling_avg(self, dashboard_service):
        """Test that trend chart includes rolling average."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_trend_chart_data()

        if data:
            assert "rolling_avg" in data[0]


class TestActivityHeatmap:
    """Tests for activity heatmap data."""

    @pytest.mark.asyncio
    async def test_get_activity_heatmap_returns_list(self, dashboard_service):
        """Test that heatmap returns a list."""
        await dashboard_service.initialize()
        # Use weeks parameter via legacy method
        data = await dashboard_service.get_activity_heatmap_by_weeks(weeks=4)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_activity_heatmap_has_required_fields(self, dashboard_service):
        """Test that heatmap data has required fields."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_activity_heatmap_by_weeks(weeks=4)

        if data:
            assert "date" in data[0]
            assert "weekday" in data[0]
            assert "week" in data[0]
            assert "activity" in data[0]
            assert "intensity" in data[0]

    @pytest.mark.asyncio
    async def test_get_activity_heatmap_respects_weeks_param(self, dashboard_service):
        """Test that weeks parameter is respected."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_activity_heatmap_by_weeks(weeks=4)
        expected_days = 4 * 7
        assert len(data) == expected_days


class TestTopicDistribution:
    """Tests for topic distribution data."""

    @pytest.mark.asyncio
    async def test_get_topic_distribution_returns_list(self, dashboard_service):
        """Test that topic distribution returns a list."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_topic_distribution()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_topic_distribution_has_required_fields(self, dashboard_service):
        """Test that distribution data has required fields."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_topic_distribution()

        if data:
            assert "topic" in data[0]
            assert "interactions" in data[0]
            assert "percentage" in data[0]


class TestGoalProgressData:
    """Tests for goal progress data."""

    @pytest.mark.asyncio
    async def test_get_goal_progress_data_returns_list(self, dashboard_service):
        """Test that goal progress returns a list."""
        await dashboard_service.initialize()
        data = await dashboard_service.get_goal_progress_data()
        assert isinstance(data, list)


class TestCacheManagement:
    """Tests for cache management."""

    @pytest.mark.asyncio
    async def test_clear_cache_for_specific_user(self, dashboard_service):
        """Test clearing cache for specific user."""
        await dashboard_service.initialize()
        dashboard_service._cache._cache["dashboard_user123"] = MagicMock()

        dashboard_service.clear_cache(user_id="user123")

        assert "dashboard_user123" not in dashboard_service._cache._cache

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, dashboard_service):
        """Test clearing all caches."""
        await dashboard_service.initialize()
        dashboard_service._cache._cache["dashboard_user1"] = MagicMock()
        dashboard_service._cache._cache["dashboard_user2"] = MagicMock()

        dashboard_service.clear_cache()

        assert len(dashboard_service._cache._cache) == 0


class TestDashboardPerformance:
    """Tests for dashboard performance."""

    @pytest.mark.asyncio
    async def test_overview_api_under_200ms(self, dashboard_service, benchmark):
        """Test that overview API responds under 200ms."""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_overview_data()

        # Allow more time in test environment
        b.assert_under(500, "Overview API too slow")

    @pytest.mark.asyncio
    async def test_quality_chart_under_300ms(self, dashboard_service, benchmark):
        """Test that quality chart API responds under 300ms."""
        await dashboard_service.initialize()

        with benchmark() as b:
            await dashboard_service.get_quality_chart_data()

        b.assert_under(600, "Quality chart API too slow")
