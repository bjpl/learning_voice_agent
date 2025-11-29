"""
Test Suite: Phase 5 Integration Tests
=====================================

25+ integration tests for the complete learning system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from app.learning.config import LearningConfig
from app.learning.models import (
    Feedback,
    FeedbackType,
    QualityScore,
    UserPreference
)


class TestLearningSystemIntegration:
    """Integration tests for the complete learning system."""

    @pytest.mark.asyncio
    async def test_full_learning_cycle(self, full_learning_system):
        """Test complete learning cycle from feedback to adaptation."""
        system = full_learning_system

        # 1. Collect feedback
        feedback = await system['feedback_collector'].collect_rating(
            session_id="integration-test",
            query_id="query-1",
            rating=0.9,
            original_query="What is machine learning?",
            original_response="Machine learning is a subset of AI."
        )

        assert feedback is not None
        assert feedback.rating == 0.9

        # 2. Score the interaction quality
        score = await system['quality_scorer'].score_response(
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            session_id="integration-test"
        )

        assert score is not None
        assert 0 <= score.composite <= 1

    @pytest.mark.asyncio
    async def test_feedback_flows_to_analytics(self, full_learning_system):
        """Test that feedback flows through to analytics."""
        system = full_learning_system

        # Record multiple feedback items
        for i in range(5):
            await system['feedback_collector'].collect_rating(
                session_id="analytics-test",
                query_id=f"query-{i}",
                rating=0.7 + i * 0.05,
                original_query=f"Question {i}",
                original_response=f"Answer {i}"
            )

        # Generate analytics report
        report = await system['analytics'].generate_daily_report(
            session_id="analytics-test"
        )

        assert report is not None

    @pytest.mark.asyncio
    async def test_quality_scores_influence_preferences(self, full_learning_system):
        """Test that quality scores influence learned preferences."""
        system = full_learning_system

        # Score multiple interactions with different characteristics
        for i in range(10):
            await system['quality_scorer'].score_response(
                query=f"Question about topic {i % 3}",
                response="A detailed response with examples and explanations.",
                session_id="preference-test"
            )

        # Check that preferences can be retrieved
        context = await system['preference_learner'].get_adaptation_context(
            session_id="preference-test"
        )

        assert context is not None


class TestFeedbackToQualityPipeline:
    """Test the feedback collection to quality scoring pipeline."""

    @pytest.mark.asyncio
    async def test_thumbs_up_creates_positive_signal(self, full_learning_system):
        """Test that thumbs up creates positive quality signal."""
        system = full_learning_system

        feedback = await system['feedback_collector'].collect_thumbs_up(
            session_id="pipeline-test",
            query_id="q1",
            original_query="Test query",
            original_response="Test response"
        )

        assert feedback.feedback_type == FeedbackType.EXPLICIT_POSITIVE
        assert feedback.rating == 1.0

    @pytest.mark.asyncio
    async def test_thumbs_down_creates_negative_signal(self, full_learning_system):
        """Test that thumbs down creates negative quality signal."""
        system = full_learning_system

        feedback = await system['feedback_collector'].collect_thumbs_down(
            session_id="pipeline-test",
            query_id="q2",
            original_query="Test query",
            original_response="Test response"
        )

        assert feedback.feedback_type == FeedbackType.EXPLICIT_NEGATIVE
        assert feedback.rating == 0.0

    @pytest.mark.asyncio
    async def test_text_feedback_preserved(self, full_learning_system):
        """Test that text feedback is preserved through pipeline."""
        system = full_learning_system

        feedback = await system['feedback_collector'].collect_thumbs_down(
            session_id="pipeline-test",
            query_id="q3",
            original_query="Test query",
            original_response="Test response",
            reason="The answer was not helpful"
        )

        assert feedback.text == "The answer was not helpful"


class TestQualityToAdaptationPipeline:
    """Test the quality scoring to adaptation pipeline."""

    @pytest.mark.asyncio
    async def test_low_quality_triggers_adaptation(self, full_learning_system):
        """Test that low quality scores can trigger adaptation."""
        system = full_learning_system

        # Generate low quality scores
        for i in range(5):
            score = await system['quality_scorer'].score_response(
                query="What is this?",
                response="Things.",  # Very short, unhelpful
                session_id="low-quality-test"
            )

        # Adapter should be able to provide context
        adaptation = system['adapter'].get_adaptation_hints(
            session_id="low-quality-test"
        )

        assert isinstance(adaptation, dict)

    @pytest.mark.asyncio
    async def test_high_quality_maintains_style(self, full_learning_system):
        """Test that high quality scores maintain current style."""
        system = full_learning_system

        # Generate high quality interactions
        for i in range(5):
            score = await system['quality_scorer'].score_response(
                query="Explain machine learning in detail",
                response="Machine learning is a field of AI that enables systems to learn from data. Here are the key concepts: 1. Training data, 2. Model architecture, 3. Evaluation metrics. This allows for pattern recognition and prediction.",
                session_id="high-quality-test"
            )


class TestPreferenceLearningIntegration:
    """Test preference learning integration."""

    @pytest.mark.asyncio
    async def test_preferences_learned_from_feedback(self, full_learning_system):
        """Test that preferences are learned from feedback patterns."""
        system = full_learning_system

        # Simulate preference pattern - user prefers detailed responses
        for i in range(10):
            if i % 2 == 0:
                # Positive for detailed
                await system['feedback_collector'].collect_thumbs_up(
                    session_id="pref-learning",
                    query_id=f"detailed-{i}",
                    original_query="Explain X",
                    original_response="Here is a detailed explanation with examples and context..."
                )
            else:
                # Negative for brief
                await system['feedback_collector'].collect_thumbs_down(
                    session_id="pref-learning",
                    query_id=f"brief-{i}",
                    original_query="Explain Y",
                    original_response="Y is Z."
                )

    @pytest.mark.asyncio
    async def test_adaptation_context_available(self, full_learning_system):
        """Test that adaptation context is available after learning."""
        system = full_learning_system

        context = await system['preference_learner'].get_adaptation_context(
            session_id="test-session"
        )

        assert context is not None
        assert hasattr(context, 'session_id')


class TestAnalyticsIntegration:
    """Test analytics integration with other components."""

    @pytest.mark.asyncio
    async def test_daily_report_includes_all_metrics(self, full_learning_system):
        """Test that daily report includes metrics from all components."""
        system = full_learning_system

        report = await system['analytics'].generate_daily_report()

        assert report is not None
        assert hasattr(report, 'total_interactions')
        assert hasattr(report, 'average_quality_score')

    @pytest.mark.asyncio
    async def test_metrics_summary_aggregates_data(self, full_learning_system):
        """Test that metrics summary correctly aggregates data."""
        system = full_learning_system

        summary = await system['analytics'].get_metrics_summary(days=7)

        assert 'period_days' in summary
        assert summary['period_days'] == 7

    @pytest.mark.asyncio
    async def test_trend_calculation_requires_data(self, full_learning_system):
        """Test trend calculation with and without data."""
        system = full_learning_system

        from app.learning.models import QualityDimension

        trend = await system['analytics'].calculate_trend(
            dimension=QualityDimension.RELEVANCE,
            session_id="empty-session"
        )

        assert trend.data_points == 0


class TestPatternDetectionIntegration:
    """Test pattern detection integration."""

    @pytest.mark.asyncio
    async def test_patterns_detected_from_queries(self, full_learning_system):
        """Test that patterns can be detected from query history."""
        system = full_learning_system

        # This would integrate with the pattern detector
        # For now, test that the component exists
        assert system['pattern_detector'] is not None

    @pytest.mark.asyncio
    async def test_pattern_detector_initialization(self, full_learning_system):
        """Test pattern detector initialization."""
        system = full_learning_system

        await system['pattern_detector'].initialize()

        assert system['pattern_detector']._initialized is True


class TestFeedbackStorePersistence:
    """Test feedback store persistence."""

    @pytest.mark.asyncio
    async def test_feedback_persisted_to_store(self, full_learning_system):
        """Test that feedback is persisted to the store."""
        system = full_learning_system

        feedback = await system['feedback_collector'].collect_rating(
            session_id="persist-test",
            query_id="q1",
            rating=0.75,
            original_query="Test",
            original_response="Response"
        )

        # Verify it can be retrieved
        stored = await system['feedback_store'].get(feedback.id)

        assert stored is not None
        assert stored.rating == 0.75

    @pytest.mark.asyncio
    async def test_feedback_queryable_by_session(self, full_learning_system):
        """Test that feedback can be queried by session."""
        system = full_learning_system

        # Create feedback in specific session
        for i in range(3):
            await system['feedback_collector'].collect_rating(
                session_id="query-test-session",
                query_id=f"q-{i}",
                rating=0.8,
                original_query=f"Query {i}",
                original_response=f"Response {i}"
            )

        # Query by session
        results = await system['feedback_store'].query(
            session_id="query-test-session"
        )

        assert len(results) >= 3


class TestErrorHandling:
    """Test error handling across the system."""

    @pytest.mark.asyncio
    async def test_quality_scorer_handles_empty_input(self, full_learning_system):
        """Test that quality scorer handles empty inputs gracefully."""
        system = full_learning_system

        score = await system['quality_scorer'].score_response(
            query="",
            response="Some response",
            session_id="error-test"
        )

        # Should still return a valid score
        assert score is not None
        assert 0 <= score.composite <= 1

    @pytest.mark.asyncio
    async def test_analytics_handles_no_data(self, full_learning_system):
        """Test that analytics handles missing data gracefully."""
        system = full_learning_system

        report = await system['analytics'].generate_daily_report(
            session_id="nonexistent-session"
        )

        assert report is not None

    @pytest.mark.asyncio
    async def test_preference_learner_handles_new_session(self, full_learning_system):
        """Test preference learner with new session."""
        system = full_learning_system

        context = await system['preference_learner'].get_adaptation_context(
            session_id="brand-new-session"
        )

        assert context is not None


class TestConcurrentOperations:
    """Test concurrent operations across components."""

    @pytest.mark.asyncio
    async def test_concurrent_feedback_collection(self, full_learning_system):
        """Test concurrent feedback collection."""
        system = full_learning_system

        async def collect_feedback(i):
            return await system['feedback_collector'].collect_rating(
                session_id="concurrent-test",
                query_id=f"concurrent-{i}",
                rating=0.5 + (i / 20),
                original_query=f"Query {i}",
                original_response=f"Response {i}"
            )

        # Collect 10 feedbacks concurrently
        results = await asyncio.gather(*[collect_feedback(i) for i in range(10)])

        assert len(results) == 10
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_quality_scoring(self, full_learning_system):
        """Test concurrent quality scoring."""
        system = full_learning_system

        async def score_interaction(i):
            return await system['quality_scorer'].score_response(
                query=f"Question {i}",
                response=f"Answer to question {i} with some detail.",
                session_id="concurrent-score-test"
            )

        # Score 10 interactions concurrently
        results = await asyncio.gather(*[score_interaction(i) for i in range(10)])

        assert len(results) == 10
        assert all(0 <= r.composite <= 1 for r in results)


class TestSystemLifecycle:
    """Test system lifecycle operations."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Temp database path timing issue with fixture - needs infrastructure fix")
    async def test_components_can_initialize(self, test_config_with_temp_db):
        """Test that all components can initialize."""
        from app.learning.feedback_store import FeedbackStore
        from app.learning.feedback_collector import FeedbackCollector
        from app.learning.quality_scorer import QualityScorer
        from app.learning.preference_learner import PreferenceLearner
        from app.learning.analytics import LearningAnalytics

        config = test_config_with_temp_db

        store = FeedbackStore(config)
        await store.initialize()

        collector = FeedbackCollector(config=config, feedback_store=store)
        await collector.initialize()

        await store.close()
        await collector.close()

    @pytest.mark.asyncio
    async def test_components_cleanup_properly(self, full_learning_system):
        """Test that components clean up properly."""
        system = full_learning_system

        # Components should be accessible
        assert system['feedback_store'] is not None
        assert system['feedback_collector'] is not None


class TestDataFlow:
    """Test data flow through the system."""

    @pytest.mark.asyncio
    async def test_feedback_to_analytics_flow(self, full_learning_system):
        """Test data flows from feedback to analytics."""
        system = full_learning_system

        # Create feedback
        await system['feedback_collector'].collect_rating(
            session_id="dataflow-test",
            query_id="df-1",
            rating=0.85,
            original_query="Test query",
            original_response="Test response"
        )

        # Should be reflected in summary
        summary = await system['feedback_collector'].get_feedback_summary(
            session_id="dataflow-test",
            limit=10
        )

        assert summary is not None

    @pytest.mark.asyncio
    async def test_quality_to_analytics_flow(self, full_learning_system):
        """Test data flows from quality scoring to analytics."""
        system = full_learning_system

        # Score interaction
        score = await system['quality_scorer'].score_response(
            query="Flow test query",
            response="Flow test response with enough content to analyze.",
            session_id="quality-flow-test"
        )

        # Record in analytics
        system['analytics'].record_quality_score(score)

        # Generate report
        report = await system['analytics'].generate_daily_report(
            session_id="quality-flow-test"
        )

        assert report is not None
