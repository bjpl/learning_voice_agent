"""
Unit Tests for Learning Metrics - Phase 3 Learning Integration
================================================================

Test Coverage:
- Search quality recording
- MRR (Mean Reciprocal Rank) calculation
- Hit rate metrics (top-1, top-3, top-5)
- Trend analysis
- Baseline comparison
- Session tracking
"""

import pytest
from datetime import datetime, timedelta

from app.learning.learning_metrics import (
    LearningMetrics,
    SearchMetric,
    ImprovementStats,
    create_learning_metrics
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def metrics():
    """Create learning metrics tracker."""
    return LearningMetrics(window_size=100, baseline_window=10)


@pytest.fixture
def sample_results():
    """Create sample search results."""
    return [
        {"id": "r1", "similarity": 0.95},
        {"id": "r2", "similarity": 0.85},
        {"id": "r3", "similarity": 0.75},
        {"id": "r4", "similarity": 0.65},
        {"id": "r5", "similarity": 0.55},
    ]


# ============================================================================
# Test Suite: Basic Recording
# ============================================================================

class TestBasicRecording:
    """Test basic metric recording."""

    @pytest.mark.asyncio
    async def test_record_search_quality(self, metrics, sample_results):
        """Test recording a search with user selection."""
        # Act
        await metrics.record_search_quality(
            query="test query",
            results=sample_results,
            user_selected_index=1,
            session_id="session_1"
        )

        # Assert
        assert len(metrics._metrics) == 1
        metric = metrics._metrics[0]
        assert metric.query == "test query"
        assert metric.results_count == 5
        assert metric.user_selected_index == 1
        assert metric.selected_id == "r2"

    @pytest.mark.asyncio
    async def test_record_multiple_searches(self, metrics, sample_results):
        """Test recording multiple searches."""
        # Act
        for i in range(5):
            await metrics.record_search_quality(
                query=f"query_{i}",
                results=sample_results,
                user_selected_index=i % 3
            )

        # Assert
        assert len(metrics._metrics) == 5


# ============================================================================
# Test Suite: MRR Calculation
# ============================================================================

class TestMRRCalculation:
    """Test Mean Reciprocal Rank calculation."""

    def test_mrr_first_result(self):
        """Test MRR for first result selection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=0
        )
        assert metric.mrr_score == 1.0

    def test_mrr_second_result(self):
        """Test MRR for second result selection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=1
        )
        assert metric.mrr_score == 0.5

    def test_mrr_third_result(self):
        """Test MRR for third result selection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=2
        )
        assert abs(metric.mrr_score - 0.333) < 0.01

    def test_mrr_no_selection(self):
        """Test MRR when no result selected."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=None
        )
        assert metric.mrr_score == 0.0


# ============================================================================
# Test Suite: Hit Rate Metrics
# ============================================================================

class TestHitRates:
    """Test hit rate calculations."""

    def test_top_1_hit(self):
        """Test top-1 hit detection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=0
        )
        assert metric.is_top_1 is True
        assert metric.is_top_3 is True
        assert metric.is_top_5 is True

    def test_top_3_hit(self):
        """Test top-3 hit detection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=2
        )
        assert metric.is_top_1 is False
        assert metric.is_top_3 is True
        assert metric.is_top_5 is True

    def test_top_5_hit(self):
        """Test top-5 hit detection."""
        metric = SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=10,
            user_selected_index=4
        )
        assert metric.is_top_1 is False
        assert metric.is_top_3 is False
        assert metric.is_top_5 is True


# ============================================================================
# Test Suite: Improvement Statistics
# ============================================================================

class TestImprovementStats:
    """Test improvement statistics calculation."""

    @pytest.mark.asyncio
    async def test_get_improvement_stats(self, metrics, sample_results):
        """Test calculating improvement statistics."""
        # Arrange - record some searches
        for i in range(10):
            await metrics.record_search_quality(
                query=f"query_{i}",
                results=sample_results,
                user_selected_index=0 if i < 5 else 1  # Improving over time
            )

        # Act
        stats = await metrics.get_improvement_stats()

        # Assert
        assert stats.total_searches == 10
        assert stats.searches_with_selection == 10
        assert stats.avg_mrr > 0.5
        assert stats.top_1_hit_rate == 0.5  # 5 out of 10

    @pytest.mark.asyncio
    async def test_baseline_comparison(self, metrics, sample_results):
        """Test baseline comparison."""
        # Arrange - establish baseline (lower quality)
        for i in range(10):
            await metrics.record_search_quality(
                query=f"baseline_{i}",
                results=sample_results,
                user_selected_index=2  # Selecting 3rd result
            )

        # Baseline should now be calculated
        assert metrics._baseline_calculated is True

        # Add improved searches
        for i in range(10):
            await metrics.record_search_quality(
                query=f"improved_{i}",
                results=sample_results,
                user_selected_index=0  # Selecting 1st result
            )

        # Act
        stats = await metrics.get_improvement_stats()

        # Assert
        assert stats.baseline_mrr is not None
        assert stats.mrr_improvement is not None
        assert stats.mrr_improvement > 0  # Should show improvement


# ============================================================================
# Test Suite: Session Tracking
# ============================================================================

class TestSessionTracking:
    """Test per-session metric tracking."""

    @pytest.mark.asyncio
    async def test_session_metrics_tracking(self, metrics, sample_results):
        """Test tracking metrics per session."""
        # Arrange - record searches for different sessions
        await metrics.record_search_quality(
            query="query_1",
            results=sample_results,
            user_selected_index=0,
            session_id="session_1"
        )
        await metrics.record_search_quality(
            query="query_2",
            results=sample_results,
            user_selected_index=1,
            session_id="session_1"
        )
        await metrics.record_search_quality(
            query="query_3",
            results=sample_results,
            user_selected_index=0,
            session_id="session_2"
        )

        # Assert
        assert len(metrics._session_metrics["session_1"]) == 2
        assert len(metrics._session_metrics["session_2"]) == 1

    @pytest.mark.asyncio
    async def test_get_session_stats(self, metrics, sample_results):
        """Test getting statistics for a specific session."""
        # Arrange
        for i in range(5):
            await metrics.record_search_quality(
                query=f"query_{i}",
                results=sample_results,
                user_selected_index=0,
                session_id="session_test"
            )

        # Act
        stats = await metrics.get_session_stats("session_test")

        # Assert
        assert stats is not None
        assert stats.total_searches == 5
        assert stats.top_1_hit_rate == 1.0  # All selected first result


# ============================================================================
# Test Suite: Trend Analysis
# ============================================================================

class TestTrendAnalysis:
    """Test trend analysis over time."""

    @pytest.mark.asyncio
    async def test_trend_analysis(self, metrics, sample_results):
        """Test trend detection."""
        # Arrange - record searches with improving quality
        for i in range(20):
            # Quality improves over time (select earlier results as we go)
            selection = min(i // 4, 2)  # 0, 0, 0, 0, 1, 1, 1, 1, 2, 2...
            await metrics.record_search_quality(
                query=f"query_{i}",
                results=sample_results,
                user_selected_index=selection
            )

        # Act
        trend_data = await metrics.get_trend_analysis(
            bucket_size=timedelta(seconds=1)
        )

        # Assert
        assert "buckets" in trend_data
        assert "trend" in trend_data
        # Note: trend may vary based on timing

    @pytest.mark.asyncio
    async def test_insufficient_data_for_trend(self, metrics):
        """Test trend analysis with insufficient data."""
        # Act
        trend_data = await metrics.get_trend_analysis()

        # Assert
        assert trend_data["status"] == "insufficient_data"


# ============================================================================
# Test Suite: Summary and Utilities
# ============================================================================

class TestSummaryAndUtilities:
    """Test summary and utility functions."""

    @pytest.mark.asyncio
    async def test_get_summary(self, metrics, sample_results):
        """Test getting metrics summary."""
        # Arrange
        await metrics.record_search_quality(
            query="test",
            results=sample_results,
            user_selected_index=0,
            session_id="session_1"
        )

        # Act
        summary = metrics.get_summary()

        # Assert
        assert summary["total_searches"] == 1
        assert summary["sessions_tracked"] == 1

    def test_reset(self, metrics):
        """Test resetting metrics."""
        # Arrange - add some data
        metrics._metrics.append(SearchMetric(
            query="test",
            timestamp=datetime.utcnow(),
            results_count=5,
            user_selected_index=0
        ))

        # Act
        metrics.reset()

        # Assert
        assert len(metrics._metrics) == 0
        assert not metrics._baseline_calculated

    def test_factory_function(self):
        """Test factory function."""
        # Act
        metrics = create_learning_metrics(window_size=500, baseline_window=50)

        # Assert
        assert metrics.window_size == 500
        assert metrics.baseline_window == 50


# ============================================================================
# Test Suite: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_results(self, metrics):
        """Test recording search with no results."""
        # Act
        await metrics.record_search_quality(
            query="test",
            results=[],
            user_selected_index=None
        )

        # Assert
        assert len(metrics._metrics) == 1
        metric = metrics._metrics[0]
        assert metric.results_count == 0

    @pytest.mark.asyncio
    async def test_no_selection(self, metrics, sample_results):
        """Test recording search with no user selection."""
        # Act
        await metrics.record_search_quality(
            query="test",
            results=sample_results,
            user_selected_index=None
        )

        # Assert
        metric = metrics._metrics[0]
        assert metric.user_selected_index is None
        assert metric.mrr_score == 0.0

    @pytest.mark.asyncio
    async def test_window_size_limit(self, metrics, sample_results):
        """Test that window size limit is enforced."""
        # Arrange
        window_size = 100

        # Act - add more than window size
        for i in range(150):
            await metrics.record_search_quality(
                query=f"query_{i}",
                results=sample_results,
                user_selected_index=0
            )

        # Assert
        assert len(metrics._metrics) == window_size  # Should be capped


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
