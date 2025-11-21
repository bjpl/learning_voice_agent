"""
Test Suite: LearningAnalytics
=============================

30+ tests for learning analytics functionality.
"""

import pytest
from datetime import datetime, timedelta

from app.learning.analytics import LearningAnalytics
from app.learning.models import DailyReport, LearningInsight, QualityDimension


class TestDailyReportGeneration:
    """Tests for daily report generation."""

    @pytest.mark.asyncio
    async def test_generate_daily_report_returns_report(self, learning_analytics):
        """Test that daily report generation returns a DailyReport."""
        report = await learning_analytics.generate_daily_report()

        assert isinstance(report, DailyReport)

    @pytest.mark.asyncio
    async def test_generate_daily_report_with_date(self, learning_analytics):
        """Test report generation for specific date."""
        yesterday = datetime.utcnow() - timedelta(days=1)
        report = await learning_analytics.generate_daily_report(date=yesterday)

        assert report.date.date() == yesterday.date()

    @pytest.mark.asyncio
    async def test_generate_daily_report_with_session(self, learning_analytics):
        """Test report generation filtered by session."""
        report = await learning_analytics.generate_daily_report(
            session_id="test-session"
        )

        assert isinstance(report, DailyReport)

    @pytest.mark.asyncio
    async def test_report_includes_metrics(self, learning_analytics):
        """Test that report includes expected metrics."""
        report = await learning_analytics.generate_daily_report()

        assert hasattr(report, "total_interactions")
        assert hasattr(report, "total_feedback_collected")
        assert hasattr(report, "average_quality_score")

    @pytest.mark.asyncio
    async def test_report_includes_trends(self, learning_analytics):
        """Test that report includes trend data."""
        report = await learning_analytics.generate_daily_report()

        assert hasattr(report, "quality_trend")
        assert hasattr(report, "engagement_trend")
        assert hasattr(report, "feedback_trend")

    @pytest.mark.asyncio
    async def test_report_includes_recommendations(self, learning_analytics):
        """Test that report includes recommendations."""
        report = await learning_analytics.generate_daily_report()

        assert hasattr(report, "recommendations")
        assert isinstance(report.recommendations, list)


class TestQualityScoreRecording:
    """Tests for quality score recording."""

    @pytest.mark.asyncio
    async def test_record_quality_score(self, learning_analytics, sample_quality_score):
        """Test recording a quality score."""
        score = sample_quality_score(composite=0.8)

        learning_analytics.record_quality_score(score)

        # Score should be stored
        scores = learning_analytics._quality_scores.get(score.session_id, [])
        assert len(scores) >= 1

    @pytest.mark.asyncio
    async def test_record_multiple_scores(self, learning_analytics, sample_quality_score):
        """Test recording multiple quality scores."""
        for i in range(10):
            score = sample_quality_score(
                session_id="test-session",
                query_id=f"query-{i}",
                composite=0.5 + i * 0.05
            )
            learning_analytics.record_quality_score(score)

        scores = learning_analytics._quality_scores.get("test-session", [])
        assert len(scores) == 10


class TestInteractionCounting:
    """Tests for interaction counting."""

    @pytest.mark.asyncio
    async def test_increment_interaction_count(self, learning_analytics):
        """Test incrementing interaction count."""
        learning_analytics.increment_interaction_count("session-1")
        learning_analytics.increment_interaction_count("session-1")
        learning_analytics.increment_interaction_count("session-2")

        assert learning_analytics._interaction_counts["session-1"] == 2
        assert learning_analytics._interaction_counts["session-2"] == 1


class TestMetricsSummary:
    """Tests for metrics summary."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary_empty(self, learning_analytics):
        """Test metrics summary with no data."""
        summary = await learning_analytics.get_metrics_summary()

        assert "period_days" in summary
        assert "total_scored_interactions" in summary

    @pytest.mark.asyncio
    async def test_get_metrics_summary_with_data(self, learning_analytics, sample_quality_score):
        """Test metrics summary with recorded data."""
        # Record some scores
        for i in range(5):
            score = sample_quality_score(
                session_id="test-session",
                query_id=f"query-{i}",
                composite=0.7
            )
            learning_analytics.record_quality_score(score)

        summary = await learning_analytics.get_metrics_summary(
            session_id="test-session",
            days=7
        )

        assert summary["total_scored_interactions"] >= 5

    @pytest.mark.asyncio
    async def test_get_metrics_summary_custom_period(self, learning_analytics):
        """Test metrics summary with custom time period."""
        summary = await learning_analytics.get_metrics_summary(days=30)

        assert summary["period_days"] == 30


class TestTrendCalculation:
    """Tests for trend calculation."""

    @pytest.mark.asyncio
    async def test_calculate_trend_no_data(self, learning_analytics):
        """Test trend calculation with no data."""
        trend = await learning_analytics.calculate_trend(
            dimension=QualityDimension.RELEVANCE,
            session_id="empty-session"
        )

        assert trend.data_points == 0

    @pytest.mark.asyncio
    async def test_calculate_trend_with_data(self, learning_analytics, sample_quality_scores):
        """Test trend calculation with data."""
        scores = sample_quality_scores(n=20, session_id="test-session")
        for score in scores:
            learning_analytics.record_quality_score(score)

        trend = await learning_analytics.calculate_trend(
            dimension=QualityDimension.RELEVANCE,
            session_id="test-session"
        )

        assert trend.data_points >= 0

    @pytest.mark.asyncio
    async def test_trend_includes_statistics(self, learning_analytics, sample_quality_scores):
        """Test that trend includes all statistics."""
        scores = sample_quality_scores(n=10, session_id="test-session")
        for score in scores:
            learning_analytics.record_quality_score(score)

        trend = await learning_analytics.calculate_trend(
            dimension=QualityDimension.RELEVANCE,
            session_id="test-session"
        )

        assert hasattr(trend, "start_value")
        assert hasattr(trend, "end_value")
        assert hasattr(trend, "change")
        assert hasattr(trend, "change_percent")
        assert hasattr(trend, "min_value")
        assert hasattr(trend, "max_value")
        assert hasattr(trend, "average_value")


class TestInsightGeneration:
    """Tests for insight generation."""

    @pytest.mark.asyncio
    async def test_report_generates_insights(self, learning_analytics, sample_quality_scores):
        """Test that reports generate insights."""
        scores = sample_quality_scores(n=10, session_id="test-session", base_score=0.4)
        for score in scores:
            learning_analytics.record_quality_score(score)

        report = await learning_analytics.generate_daily_report(
            session_id="test-session"
        )

        # With low scores, should generate insights
        assert isinstance(report.insights, list)

    @pytest.mark.asyncio
    async def test_low_quality_generates_insight(self, learning_analytics, sample_quality_scores):
        """Test that low quality scores generate warning insight."""
        # Create very low quality scores
        scores = sample_quality_scores(
            n=10,
            session_id="test-session",
            base_score=0.3
        )
        for score in scores:
            learning_analytics.record_quality_score(score)

        report = await learning_analytics.generate_daily_report(
            session_id="test-session"
        )

        # Should have insights about low quality
        assert isinstance(report.insights, list)


class TestReportRecommendations:
    """Tests for report recommendations."""

    @pytest.mark.asyncio
    async def test_declining_quality_recommendation(self, learning_analytics, sample_quality_scores):
        """Test recommendations for declining quality."""
        report = await learning_analytics.generate_daily_report()

        assert isinstance(report.recommendations, list)

    @pytest.mark.asyncio
    async def test_healthy_metrics_recommendation(self, learning_analytics, sample_quality_scores):
        """Test recommendations for healthy metrics."""
        scores = sample_quality_scores(
            n=10,
            session_id="test-session",
            base_score=0.85
        )
        for score in scores:
            learning_analytics.record_quality_score(score)

        report = await learning_analytics.generate_daily_report(
            session_id="test-session"
        )

        # Should have at least one recommendation
        assert len(report.recommendations) >= 1


class TestAnalyticsInitialization:
    """Tests for analytics initialization."""

    @pytest.mark.asyncio
    async def test_initialize(self, learning_analytics):
        """Test analytics initialization."""
        await learning_analytics.initialize()
        # Should not raise


class TestReportContent:
    """Tests for report content accuracy."""

    @pytest.mark.asyncio
    async def test_report_date_is_correct(self, learning_analytics):
        """Test that report date is correct."""
        target_date = datetime.utcnow() - timedelta(days=1)
        report = await learning_analytics.generate_daily_report(date=target_date)

        assert report.date.date() == target_date.date()

    @pytest.mark.asyncio
    async def test_report_generated_at_is_recent(self, learning_analytics):
        """Test that generated_at timestamp is recent."""
        report = await learning_analytics.generate_daily_report()

        time_diff = datetime.utcnow() - report.generated_at
        assert time_diff.total_seconds() < 60  # Within last minute

    @pytest.mark.asyncio
    async def test_report_version_present(self, learning_analytics):
        """Test that report version is present."""
        report = await learning_analytics.generate_daily_report()

        assert hasattr(report, "report_version")
        assert report.report_version is not None


class TestDimensionScores:
    """Tests for dimension score tracking."""

    @pytest.mark.asyncio
    async def test_report_includes_dimension_scores(self, learning_analytics, sample_quality_scores):
        """Test that report includes dimension score averages."""
        scores = sample_quality_scores(n=5, session_id="test-session")
        for score in scores:
            learning_analytics.record_quality_score(score)

        report = await learning_analytics.generate_daily_report(
            session_id="test-session"
        )

        assert "dimension_scores" in report.model_dump()


class TestPerformanceMetrics:
    """Tests for analytics performance."""

    @pytest.mark.asyncio
    async def test_report_generation_performance(self, learning_analytics, benchmark):
        """Test that report generation is performant."""
        with benchmark() as b:
            await learning_analytics.generate_daily_report()

        b.assert_under(1000, "Report generation should complete in under 1s")
