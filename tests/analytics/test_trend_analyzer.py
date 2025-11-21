"""
Test Suite for Trend Analyzer
=============================

Comprehensive tests for trend analysis functionality.
Target: 20+ tests covering all trend analysis features.
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock

from app.analytics.progress_models import (
    DailyProgress,
    TrendDirection,
)


class TestTrendAnalyzerInitialization:
    """Tests for TrendAnalyzer initialization."""

    @pytest.mark.asyncio
    async def test_initializes_successfully(self, trend_analyzer):
        """Test that trend analyzer initializes without errors."""
        await trend_analyzer.initialize()
        assert trend_analyzer._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, trend_analyzer):
        """Test that multiple initialization calls are safe."""
        await trend_analyzer.initialize()
        await trend_analyzer.initialize()
        assert trend_analyzer._initialized is True


class TestQualityTrendAnalysis:
    """Tests for quality trend analysis."""

    @pytest.mark.asyncio
    async def test_analyze_quality_trend_with_data(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test quality trend analysis with sufficient data."""
        daily = sample_daily_progress_list(days=14, base_quality=0.75)
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_quality_trend(daily)

        assert trend is not None
        assert trend.metric == "quality_score"
        assert trend.direction in [TrendDirection.IMPROVING, TrendDirection.STABLE, TrendDirection.DECLINING]

    @pytest.mark.asyncio
    async def test_analyze_quality_trend_insufficient_data(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test quality trend with insufficient data."""
        daily = sample_daily_progress_list(days=2)
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_quality_trend(daily)

        assert trend is not None
        # Should return empty trend with low confidence
        assert trend.confidence <= 0.1 or len(trend.data_points) == 0

    @pytest.mark.asyncio
    async def test_analyze_quality_trend_improving(
        self, trend_analyzer, sample_daily_progress
    ):
        """Test detection of improving quality trend."""
        daily = []
        for i in range(14):
            quality = 0.5 + (i * 0.03)  # Steadily improving
            daily.append(sample_daily_progress(
                target_date=date.today() - timedelta(days=13 - i),
                avg_quality_score=quality
            ))
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_quality_trend(daily)

        assert trend.direction == TrendDirection.IMPROVING

    @pytest.mark.asyncio
    async def test_analyze_quality_trend_declining(
        self, trend_analyzer, sample_daily_progress
    ):
        """Test detection of declining quality trend."""
        daily = []
        for i in range(14):
            quality = 0.9 - (i * 0.04)  # Steadily declining
            daily.append(sample_daily_progress(
                target_date=date.today() - timedelta(days=13 - i),
                avg_quality_score=max(0.1, quality)
            ))
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_quality_trend(daily)

        assert trend.direction == TrendDirection.DECLINING


class TestActivityTrendAnalysis:
    """Tests for activity trend analysis."""

    @pytest.mark.asyncio
    async def test_analyze_activity_trend(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test activity trend analysis."""
        daily = sample_daily_progress_list(days=14)
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_activity_trend(daily)

        assert trend is not None
        assert trend.metric == "activity"


class TestEngagementTrendAnalysis:
    """Tests for engagement trend analysis."""

    @pytest.mark.asyncio
    async def test_analyze_engagement_trend(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test engagement trend analysis."""
        daily = sample_daily_progress_list(days=14)
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_engagement_trend(daily)

        assert trend is not None
        assert trend.metric == "engagement"


class TestRollingAverage:
    """Tests for rolling average calculation."""

    @pytest.mark.asyncio
    async def test_rolling_average_basic(self, trend_analyzer):
        """Test basic rolling average calculation."""
        data = [1, 2, 3, 4, 5, 6, 7]
        await trend_analyzer.initialize()

        result = await trend_analyzer.calculate_rolling_average(data, window=3)

        assert len(result) == len(data)
        # Last value should be average of last 3
        assert abs(result[-1] - 6.0) < 0.01

    @pytest.mark.asyncio
    async def test_rolling_average_with_short_data(self, trend_analyzer):
        """Test rolling average with data shorter than window."""
        data = [1, 2]
        await trend_analyzer.initialize()

        result = await trend_analyzer.calculate_rolling_average(data, window=7)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rolling_average_empty_data(self, trend_analyzer):
        """Test rolling average with empty data."""
        await trend_analyzer.initialize()

        result = await trend_analyzer.calculate_rolling_average([], window=7)

        assert result == []


class TestSeasonalityDetection:
    """Tests for seasonality detection."""

    @pytest.mark.asyncio
    async def test_detect_seasonality_with_pattern(
        self, trend_analyzer, sample_daily_progress
    ):
        """Test detection of weekly seasonality pattern."""
        daily = []
        for i in range(28):  # 4 weeks
            target_date = date.today() - timedelta(days=27 - i)
            # Weekend days have less activity
            exchanges = 10 if target_date.weekday() >= 5 else 30
            daily.append(sample_daily_progress(
                target_date=target_date,
                total_exchanges=exchanges
            ))
        await trend_analyzer.initialize()

        result = await trend_analyzer.detect_seasonality(daily)

        assert "detected" in result
        if result["detected"]:
            assert "best_day" in result
            assert "worst_day" in result

    @pytest.mark.asyncio
    async def test_detect_seasonality_insufficient_data(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test seasonality detection with insufficient data."""
        daily = sample_daily_progress_list(days=5)
        await trend_analyzer.initialize()

        result = await trend_analyzer.detect_seasonality(daily)

        assert result["detected"] is False
        assert "reason" in result


class TestForecasting:
    """Tests for forecasting functionality."""

    @pytest.mark.asyncio
    async def test_forecast_linear(self, trend_analyzer):
        """Test linear forecasting."""
        data = [10, 12, 14, 16, 18, 20]  # Linear trend
        await trend_analyzer.initialize()

        forecasts = await trend_analyzer.forecast(data, days=3, method="linear")

        assert len(forecasts) == 3
        # Linear trend should predict increasing values
        assert forecasts[0]["value"] > data[-1] * 0.8

    @pytest.mark.asyncio
    async def test_forecast_ema(self, trend_analyzer):
        """Test EMA forecasting."""
        data = [10, 12, 11, 13, 12, 14]
        await trend_analyzer.initialize()

        forecasts = await trend_analyzer.forecast(data, days=3, method="ema")

        assert len(forecasts) == 3
        # All forecasts should have confidence
        for f in forecasts:
            assert "confidence" in f
            assert 0 <= f["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_forecast_insufficient_data(self, trend_analyzer):
        """Test forecasting with insufficient data."""
        data = [10]
        await trend_analyzer.initialize()

        forecasts = await trend_analyzer.forecast(data, days=3)

        assert forecasts == []

    @pytest.mark.asyncio
    async def test_forecast_includes_confidence_intervals(self, trend_analyzer):
        """Test that forecasts include confidence intervals."""
        data = [10, 12, 14, 16, 18, 20]
        await trend_analyzer.initialize()

        forecasts = await trend_analyzer.forecast(data, days=3)

        for f in forecasts:
            assert "lower_bound" in f
            assert "upper_bound" in f
            assert f["lower_bound"] <= f["value"] <= f["upper_bound"]


class TestPeriodComparison:
    """Tests for period comparison."""

    @pytest.mark.asyncio
    async def test_compare_periods_basic(self, trend_analyzer):
        """Test basic period comparison."""
        current = [15, 16, 17, 18, 19]
        previous = [10, 11, 12, 13, 14]
        await trend_analyzer.initialize()

        result = await trend_analyzer.compare_periods(current, previous)

        assert "change" in result
        assert "change_percent" in result
        assert "direction" in result
        assert result["change"] > 0
        assert result["direction"] == TrendDirection.IMPROVING.value

    @pytest.mark.asyncio
    async def test_compare_periods_declining(self, trend_analyzer):
        """Test period comparison with decline."""
        current = [10, 11, 12, 13, 14]
        previous = [15, 16, 17, 18, 19]
        await trend_analyzer.initialize()

        result = await trend_analyzer.compare_periods(current, previous)

        assert result["change"] < 0
        assert result["direction"] == TrendDirection.DECLINING.value

    @pytest.mark.asyncio
    async def test_compare_periods_empty_data(self, trend_analyzer):
        """Test period comparison with empty data."""
        await trend_analyzer.initialize()

        result = await trend_analyzer.compare_periods([], [])

        assert "error" in result


class TestTrendSummary:
    """Tests for comprehensive trend summary."""

    @pytest.mark.asyncio
    async def test_get_trend_summary_includes_all_trends(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test that trend summary includes all trend types."""
        daily = sample_daily_progress_list(days=30)
        await trend_analyzer.initialize()

        summary = await trend_analyzer.get_trend_summary(daily)

        assert "quality" in summary
        assert "activity" in summary
        assert "engagement" in summary
        assert "seasonality" in summary

    @pytest.mark.asyncio
    async def test_get_trend_summary_includes_forecast(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test that trend summary includes forecasts."""
        daily = sample_daily_progress_list(days=30, include_empty_days=False)
        await trend_analyzer.initialize()

        summary = await trend_analyzer.get_trend_summary(daily)

        assert "quality_forecast" in summary

    @pytest.mark.asyncio
    async def test_get_trend_summary_includes_period_info(
        self, trend_analyzer, sample_daily_progress_list
    ):
        """Test that trend summary includes period information."""
        daily = sample_daily_progress_list(days=30)
        await trend_analyzer.initialize()

        summary = await trend_analyzer.get_trend_summary(daily)

        assert "period" in summary
        assert "start" in summary["period"]
        assert "end" in summary["period"]
        assert "days" in summary["period"]


class TestTrendDataSerialization:
    """Tests for TrendData serialization."""

    @pytest.mark.asyncio
    async def test_trend_data_to_dict(self, trend_analyzer, sample_daily_progress_list):
        """Test TrendData serialization to dictionary."""
        daily = sample_daily_progress_list(days=14)
        await trend_analyzer.initialize()

        trend = await trend_analyzer.analyze_quality_trend(daily)
        d = trend.to_dict()

        assert "metric" in d
        assert "direction" in d
        assert "change_percent" in d
        assert "confidence" in d
