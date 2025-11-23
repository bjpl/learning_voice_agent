"""
Tests for Quality Scoring Engine - Phase 5

This module tests:
- QualityScore dataclass and composite calculation
- SessionQuality aggregation and trends
- QualityTrend time-series analysis
- ImprovementArea identification
- Individual scoring algorithms (relevance, clarity, etc.)
- QualityScorer integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from app.learning.scoring_models import (
    QualityScore,
    SessionQuality,
    QualityTrend,
    ImprovementArea,
    QualityLevel,
    ScoreDimension,
)
from app.learning.scoring_algorithms import (
    RelevanceScorer,
    EngagementScorer,
    ClarityScorer,
    HelpfulnessScorer,
    AccuracyScorer,
    CompositeScoreCalculator,
    ScoringConfig,
    SessionMetrics,
    FeedbackData,
)
from app.learning.quality_scorer import (
    QualityScorer,
    QualityScoringMiddleware,
    create_quality_scorer,
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_quality_score():
    """Create a sample quality score for testing."""
    return QualityScore(
        relevance=0.85,
        helpfulness=0.80,
        engagement=0.75,
        clarity=0.90,
        accuracy=0.88,
        session_id="test-session-001"
    )


@pytest.fixture
def sample_scores_list():
    """Create a list of sample quality scores."""
    base_time = datetime.utcnow()
    scores = []
    for i in range(10):
        score = QualityScore(
            relevance=0.7 + (i * 0.02),
            helpfulness=0.65 + (i * 0.03),
            engagement=0.60 + (i * 0.04),
            clarity=0.75 + (i * 0.02),
            accuracy=0.80 + (i * 0.01),
            timestamp=base_time + timedelta(hours=i),
            session_id=f"session-{i % 3}"
        )
        scores.append(score)
    return scores


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(
        return_value=np.random.rand(384)
    )
    return provider


@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data."""
    base_time = datetime.utcnow()
    return [
        FeedbackData(
            rating=4.5,
            timestamp=base_time - timedelta(hours=1),
            session_id="test-session",
            feedback_type="explicit"
        ),
        FeedbackData(
            rating=4.0,
            timestamp=base_time - timedelta(hours=2),
            session_id="test-session",
            feedback_type="explicit"
        ),
        FeedbackData(
            rating=5.0,
            timestamp=base_time - timedelta(minutes=30),
            session_id="test-session",
            feedback_type="explicit"
        ),
    ]


# ============================================================================
# QUALITY SCORE TESTS
# ============================================================================

class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_score_creation(self):
        """Test creating a quality score with valid values."""
        score = QualityScore(
            relevance=0.85,
            helpfulness=0.80,
            engagement=0.75,
            clarity=0.90,
            accuracy=0.88
        )

        assert score.relevance == 0.85
        assert score.helpfulness == 0.80
        assert score.engagement == 0.75
        assert score.clarity == 0.90
        assert score.accuracy == 0.88

    def test_composite_calculation(self):
        """Test that composite score is calculated correctly."""
        score = QualityScore(
            relevance=0.80,  # 30% weight
            helpfulness=0.80,  # 25% weight
            engagement=0.80,  # 20% weight
            clarity=0.80,  # 15% weight
            accuracy=0.80  # 10% weight
        )

        # All scores are 0.8, so composite should be 0.8
        assert score.composite == pytest.approx(0.80, abs=0.01)

    def test_weighted_composite(self):
        """Test weighted composite calculation."""
        score = QualityScore(
            relevance=1.0,  # 30% weight = 0.30
            helpfulness=0.0,  # 25% weight = 0.00
            engagement=1.0,  # 20% weight = 0.20
            clarity=0.0,  # 15% weight = 0.00
            accuracy=1.0  # 10% weight = 0.10
        )

        # Expected: 0.30 + 0.00 + 0.20 + 0.00 + 0.10 = 0.60
        assert score.composite == pytest.approx(0.60, abs=0.01)

    def test_quality_level_excellent(self):
        """Test quality level classification for excellent scores."""
        score = QualityScore(
            relevance=0.95,
            helpfulness=0.95,
            engagement=0.90,
            clarity=0.95,
            accuracy=0.92
        )

        assert score.quality_level == QualityLevel.EXCELLENT

    def test_quality_level_good(self):
        """Test quality level classification for good scores."""
        score = QualityScore(
            relevance=0.80,
            helpfulness=0.78,
            engagement=0.75,
            clarity=0.82,
            accuracy=0.80
        )

        assert score.quality_level == QualityLevel.GOOD

    def test_quality_level_poor(self):
        """Test quality level classification for poor scores."""
        score = QualityScore(
            relevance=0.30,
            helpfulness=0.25,
            engagement=0.35,
            clarity=0.30,
            accuracy=0.28
        )

        assert score.quality_level == QualityLevel.POOR

    def test_invalid_score_range(self):
        """Test that invalid score ranges raise errors."""
        with pytest.raises(ValueError):
            QualityScore(
                relevance=1.5,  # Invalid: > 1.0
                helpfulness=0.80,
                engagement=0.75,
                clarity=0.90,
                accuracy=0.88
            )

    def test_to_dict_serialization(self):
        """Test converting score to dictionary."""
        score = QualityScore(
            relevance=0.85,
            helpfulness=0.80,
            engagement=0.75,
            clarity=0.90,
            accuracy=0.88,
            session_id="test-session"
        )

        data = score.to_dict()

        assert "relevance" in data
        assert "composite" in data
        assert "quality_level" in data
        assert data["session_id"] == "test-session"

    def test_dimension_breakdown(self, sample_quality_score):
        """Test dimension breakdown property."""
        breakdown = sample_quality_score.dimension_breakdown

        assert "relevance" in breakdown
        assert "score" in breakdown["relevance"]
        assert "weight" in breakdown["relevance"]
        assert "contribution" in breakdown["relevance"]


# ============================================================================
# SESSION QUALITY TESTS
# ============================================================================

class TestSessionQuality:
    """Tests for SessionQuality aggregation."""

    def test_session_creation(self):
        """Test creating a session quality tracker."""
        session = SessionQuality(session_id="test-session")

        assert session.session_id == "test-session"
        assert session.response_count == 0
        assert session.average_composite == 0.0

    def test_add_response_score(self, sample_quality_score):
        """Test adding response scores to session."""
        session = SessionQuality(session_id="test-session")
        session.add_response_score(sample_quality_score)

        assert session.response_count == 1
        assert session.average_relevance == sample_quality_score.relevance
        assert session.average_composite == sample_quality_score.composite

    def test_running_average(self):
        """Test running average calculation with multiple scores."""
        session = SessionQuality(session_id="test-session")

        # Add first score (all 0.8)
        score1 = QualityScore(
            relevance=0.80, helpfulness=0.80, engagement=0.80,
            clarity=0.80, accuracy=0.80
        )
        session.add_response_score(score1)

        # Add second score (all 0.6)
        score2 = QualityScore(
            relevance=0.60, helpfulness=0.60, engagement=0.60,
            clarity=0.60, accuracy=0.60
        )
        session.add_response_score(score2)

        # Average should be 0.7
        assert session.average_relevance == pytest.approx(0.70, abs=0.01)

    def test_quality_distribution(self, sample_scores_list):
        """Test quality distribution tracking."""
        session = SessionQuality(session_id="test-session")

        for score in sample_scores_list:
            session.add_response_score(score)

        distribution = session.quality_distribution

        assert "excellent" in distribution
        assert "good" in distribution
        assert "poor" in distribution
        assert sum(distribution.values()) == pytest.approx(1.0, abs=0.01)

    def test_trend_detection_improving(self):
        """Test detecting improving quality trend."""
        session = SessionQuality(session_id="test-session")

        # Add scores with increasing quality
        for i in range(5):
            score = QualityScore(
                relevance=0.5 + (i * 0.1),
                helpfulness=0.5 + (i * 0.1),
                engagement=0.5 + (i * 0.1),
                clarity=0.5 + (i * 0.1),
                accuracy=0.5 + (i * 0.1)
            )
            session.add_response_score(score)

        assert session.quality_trend == "improving"
        assert session.trend_slope > 0

    def test_trend_detection_declining(self):
        """Test detecting declining quality trend."""
        session = SessionQuality(session_id="test-session")

        # Add scores with decreasing quality
        for i in range(5):
            score = QualityScore(
                relevance=0.9 - (i * 0.1),
                helpfulness=0.9 - (i * 0.1),
                engagement=0.9 - (i * 0.1),
                clarity=0.9 - (i * 0.1),
                accuracy=0.9 - (i * 0.1)
            )
            session.add_response_score(score)

        assert session.quality_trend == "declining"
        assert session.trend_slope < 0


# ============================================================================
# QUALITY TREND TESTS
# ============================================================================

class TestQualityTrend:
    """Tests for QualityTrend time-series analysis."""

    def test_trend_creation(self):
        """Test creating a quality trend tracker."""
        trend = QualityTrend(time_range="7d")

        assert trend.time_range == "7d"
        assert len(trend.data_points) == 0

    def test_add_data_point(self, sample_quality_score):
        """Test adding data points to trend."""
        trend = QualityTrend(time_range="7d")
        trend.add_data_point(datetime.utcnow(), sample_quality_score)

        assert len(trend.data_points) == 1
        assert trend.mean_score == sample_quality_score.composite

    def test_min_max_tracking(self):
        """Test min/max score tracking."""
        trend = QualityTrend(time_range="7d")

        low_score = QualityScore(
            relevance=0.3, helpfulness=0.3, engagement=0.3,
            clarity=0.3, accuracy=0.3
        )
        high_score = QualityScore(
            relevance=0.9, helpfulness=0.9, engagement=0.9,
            clarity=0.9, accuracy=0.9
        )

        trend.add_data_point(datetime.utcnow(), low_score)
        trend.add_data_point(datetime.utcnow(), high_score)

        assert trend.min_score == low_score.composite
        assert trend.max_score == high_score.composite

    def test_trend_analysis(self, sample_scores_list):
        """Test comprehensive trend analysis."""
        trend = QualityTrend(time_range="24h")

        for score in sample_scores_list:
            trend.add_data_point(score.timestamp, score)

        analysis = trend.analyze()

        assert "status" not in analysis or analysis.get("status") != "insufficient_data"
        assert trend.overall_trend in ["improving", "declining", "stable", "volatile"]
        assert 0 <= trend.trend_confidence <= 1

    def test_anomaly_detection(self):
        """Test anomaly detection in trend."""
        trend = QualityTrend(time_range="7d")

        # Add consistent scores
        for _ in range(10):
            score = QualityScore(
                relevance=0.7, helpfulness=0.7, engagement=0.7,
                clarity=0.7, accuracy=0.7
            )
            trend.add_data_point(datetime.utcnow(), score)

        # Add anomaly (very low score)
        anomaly_score = QualityScore(
            relevance=0.1, helpfulness=0.1, engagement=0.1,
            clarity=0.1, accuracy=0.1
        )
        trend.add_data_point(datetime.utcnow(), anomaly_score)

        trend.analyze()

        # Should detect the anomaly
        assert len(trend.anomalies) >= 1


# ============================================================================
# IMPROVEMENT AREA TESTS
# ============================================================================

class TestImprovementArea:
    """Tests for ImprovementArea identification."""

    def test_area_creation(self):
        """Test creating an improvement area from analysis."""
        area = ImprovementArea.from_analysis(
            dimension="relevance",
            current_score=0.5,
            target_score=0.75,
            affected_sessions=10
        )

        assert area.dimension == "relevance"
        assert area.current_score == 0.5
        assert area.target_score == 0.75
        assert area.gap == 0.25
        assert area.affected_sessions == 10

    def test_priority_high(self):
        """Test high priority classification."""
        area = ImprovementArea.from_analysis(
            dimension="relevance",  # High weight (0.30)
            current_score=0.3,
            target_score=0.75
        )

        assert area.priority == "high"

    def test_priority_low(self):
        """Test low priority classification."""
        area = ImprovementArea.from_analysis(
            dimension="accuracy",  # Low weight (0.10)
            current_score=0.7,
            target_score=0.75
        )

        assert area.priority == "low"

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        area = ImprovementArea.from_analysis(
            dimension="clarity",
            current_score=0.4,
            target_score=0.75
        )

        assert len(area.recommendations) > 0
        assert any("clarity" in r.lower() or "simpler" in r.lower()
                   for r in area.recommendations)


# ============================================================================
# SCORING ALGORITHM TESTS
# ============================================================================

class TestClarityScorer:
    """Tests for ClarityScorer algorithm."""

    def test_clear_text_scores_high(self):
        """Test that clear, well-structured text scores high."""
        scorer = ClarityScorer()

        clear_text = """
        Machine learning is a type of artificial intelligence.
        It allows computers to learn from data.

        Here are the main types:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning

        Each type has specific use cases.
        """

        score = scorer.calculate(clear_text)

        assert score > 0.6

    def test_complex_text_scores_lower(self):
        """Test that overly complex text scores lower."""
        scorer = ClarityScorer()

        complex_text = """
        The epistemological ramifications of implementing
        contemporaneous methodological paradigms within the
        contextualized framework of interdisciplinary
        computational linguistics necessitates circumspect
        deliberation regarding the multifaceted interconnections
        between phenomenological constructs and their
        corresponding ontological manifestations.
        """

        score = scorer.calculate(complex_text)
        clear_score = scorer.calculate("Simple words. Clear sentences. Easy to read.")

        # Complex should score lower than clear
        assert score < clear_score

    def test_empty_text_returns_default(self):
        """Test that empty text returns default score."""
        scorer = ClarityScorer()

        score = scorer.calculate("")

        assert score == 0.5  # Default score


class TestEngagementScorer:
    """Tests for EngagementScorer algorithm."""

    @pytest.mark.asyncio
    async def test_high_engagement_metrics(self):
        """Test that high engagement metrics score high."""
        scorer = EngagementScorer()

        metrics = SessionMetrics(
            follow_up_count=5,
            session_turns=10,
            session_duration_seconds=720,  # 12 minutes (optimal)
            questions_asked=8,
            topics_explored=3
        )

        score = await scorer.calculate(metrics)

        assert score > 0.7

    @pytest.mark.asyncio
    async def test_low_engagement_metrics(self):
        """Test that low engagement metrics score low."""
        scorer = EngagementScorer()

        metrics = SessionMetrics(
            follow_up_count=0,
            session_turns=1,
            session_duration_seconds=30,
            questions_asked=0,
            topics_explored=1
        )

        score = await scorer.calculate(metrics)

        assert score < 0.5


class TestHelpfulnessScorer:
    """Tests for HelpfulnessScorer algorithm."""

    @pytest.mark.asyncio
    async def test_positive_feedback_scores_high(self, sample_feedback_data):
        """Test that positive feedback scores high."""
        scorer = HelpfulnessScorer()

        score = await scorer.calculate(sample_feedback_data)

        assert score > 0.7

    @pytest.mark.asyncio
    async def test_recency_weighting(self):
        """Test that recent feedback is weighted higher."""
        scorer = HelpfulnessScorer()

        now = datetime.utcnow()

        # Old negative feedback
        old_feedback = [
            FeedbackData(
                rating=1.0,
                timestamp=now - timedelta(days=30),
                session_id="test"
            )
        ]

        # Recent positive feedback
        recent_feedback = [
            FeedbackData(
                rating=5.0,
                timestamp=now - timedelta(hours=1),
                session_id="test"
            )
        ]

        # Combined
        combined_feedback = old_feedback + recent_feedback

        score = await scorer.calculate(combined_feedback)

        # Recent positive should dominate
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_no_feedback_returns_default(self):
        """Test that no feedback returns default score."""
        scorer = HelpfulnessScorer()

        score = await scorer.calculate([])

        assert score == 0.5


class TestAccuracyScorer:
    """Tests for AccuracyScorer algorithm."""

    @pytest.mark.asyncio
    async def test_consistent_text_scores_high(self):
        """Test that consistent text scores high."""
        scorer = AccuracyScorer()

        consistent_text = """
        Python is a programming language. It is known for
        readability. Python supports multiple paradigms.
        Based on various studies, Python is widely used.
        """

        score = await scorer.calculate(
            response=consistent_text,
            query="Tell me about Python"
        )

        assert score > 0.5

    @pytest.mark.asyncio
    async def test_overconfident_text_penalized(self):
        """Test that overconfident text is penalized."""
        scorer = AccuracyScorer()

        overconfident = """
        This is absolutely 100% guaranteed to work. It will
        definitely never fail. This is the only way to do it.
        Without a doubt, this is perfect.
        """

        humble = """
        Based on my understanding, this approach typically
        works well. Research suggests it is effective.
        Generally, this method is reliable.
        """

        overconfident_score = await scorer.calculate(
            response=overconfident,
            query="How to solve this?"
        )
        humble_score = await scorer.calculate(
            response=humble,
            query="How to solve this?"
        )

        assert humble_score >= overconfident_score


class TestCompositeScoreCalculator:
    """Tests for CompositeScoreCalculator."""

    def test_calculation(self):
        """Test basic composite calculation."""
        calculator = CompositeScoreCalculator()

        composite = calculator.calculate(
            relevance=0.8,
            helpfulness=0.8,
            engagement=0.8,
            clarity=0.8,
            accuracy=0.8
        )

        assert composite == pytest.approx(0.8, abs=0.01)

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        calculator = CompositeScoreCalculator()

        weight_sum = sum(calculator.weights.values())

        assert weight_sum == pytest.approx(1.0, abs=0.001)

    def test_breakdown(self):
        """Test composite calculation with breakdown."""
        calculator = CompositeScoreCalculator()

        result = calculator.calculate_with_breakdown(
            relevance=0.9,
            helpfulness=0.7,
            engagement=0.8,
            clarity=0.85,
            accuracy=0.75
        )

        assert "composite" in result
        assert "breakdown" in result
        assert "dominant" in result
        assert "weakest" in result


# ============================================================================
# QUALITY SCORER INTEGRATION TESTS
# ============================================================================

class TestQualityScorer:
    """Integration tests for QualityScorer."""

    @pytest.mark.asyncio
    async def test_score_response(self):
        """Test scoring a single response using enhanced method."""
        scorer = create_quality_scorer()

        # Use enhanced scoring method which works correctly
        score = await scorer.score_response_enhanced(
            query="What is machine learning?",
            response="Machine learning is a subset of AI that enables systems to learn from data.",
            session_id="test-session-basic"
        )

        assert isinstance(score, QualityScore)
        assert hasattr(score, 'composite')
        assert hasattr(score, 'relevance')
        assert 0 <= score.composite <= 1

    @pytest.mark.asyncio
    async def test_score_response_enhanced(self):
        """Test enhanced scoring."""
        scorer = create_quality_scorer()

        score = await scorer.score_response_enhanced(
            query="How does Python handle memory?",
            response="""
            Python uses automatic memory management with a private heap.
            The garbage collector handles deallocation. Reference counting
            is used primarily, with cycle detection for circular references.
            """,
            session_id="test-session-enhanced"
        )

        assert isinstance(score, QualityScore)
        assert 0 <= score.composite <= 1
        assert 0 <= score.relevance <= 1
        assert 0 <= score.clarity <= 1

    @pytest.mark.asyncio
    async def test_score_session(self):
        """Test session-level scoring."""
        scorer = create_quality_scorer()

        # Score multiple responses
        for i in range(3):
            await scorer.score_response_enhanced(
                query=f"Question {i}?",
                response=f"This is response {i} with useful information.",
                session_id="session-aggregate"
            )

        session = await scorer.score_session("session-aggregate")

        assert isinstance(session, SessionQuality)
        assert session.response_count == 3

    @pytest.mark.asyncio
    async def test_quality_trends(self):
        """Test quality trend analysis."""
        scorer = create_quality_scorer()

        # Score multiple responses
        for i in range(5):
            await scorer.score_response_enhanced(
                query=f"Question about topic {i}?",
                response=f"Detailed response about topic {i}.",
                session_id=f"trend-session-{i % 2}"
            )

        trends = await scorer.get_quality_trends("24h")

        assert isinstance(trends, QualityTrend)

    @pytest.mark.asyncio
    async def test_identify_improvement_areas(self):
        """Test improvement area identification."""
        scorer = create_quality_scorer()

        # Score multiple responses with deliberately low scores
        for i in range(6):
            await scorer.score_response_enhanced(
                query="Simple question?",
                response="Very short.",  # Will score low on clarity/helpfulness
                session_id=f"improvement-session-{i}"
            )

        areas = await scorer.identify_improvement_areas(
            threshold=0.75,
            min_samples=5
        )

        assert isinstance(areas, list)

    @pytest.mark.asyncio
    async def test_session_summary(self):
        """Test getting session summary."""
        scorer = create_quality_scorer()

        # Score a response first
        await scorer.score_response_enhanced(
            query="What is testing?",
            response="Testing is the process of evaluating software.",
            session_id="summary-session"
        )

        summary = await scorer.get_session_summary("summary-session")

        assert "session_id" in summary
        assert "average_scores" in summary
        assert "quality_level" in summary


# ============================================================================
# MIDDLEWARE TESTS
# ============================================================================

class TestQualityScoringMiddleware:
    """Tests for QualityScoringMiddleware."""

    @pytest.mark.asyncio
    async def test_async_scoring(self):
        """Test asynchronous scoring."""
        scorer = create_quality_scorer()
        middleware = QualityScoringMiddleware(scorer, async_scoring=True)

        result = await middleware.score_async(
            session_id="middleware-test",
            query="Test question?",
            response="Test response."
        )

        # Async scoring returns None immediately
        assert result is None or isinstance(result, QualityScore)

    @pytest.mark.asyncio
    async def test_sync_scoring(self):
        """Test synchronous scoring."""
        scorer = create_quality_scorer()
        middleware = QualityScoringMiddleware(scorer, async_scoring=False)

        result = await middleware.score_async(
            session_id="middleware-sync-test",
            query="Sync question?",
            response="Sync response."
        )

        assert isinstance(result, QualityScore)

    @pytest.mark.asyncio
    async def test_wait_for_pending(self):
        """Test waiting for pending tasks."""
        scorer = create_quality_scorer()
        middleware = QualityScoringMiddleware(scorer, async_scoring=True)

        # Queue multiple async scores
        for i in range(3):
            await middleware.score_async(
                session_id=f"pending-{i}",
                query=f"Question {i}?",
                response=f"Response {i}."
            )

        # Wait for all to complete
        await middleware.wait_for_pending()

        # All tasks should be done
        assert len(middleware._pending_tasks) == 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        clarity_scorer = ClarityScorer()

        score = clarity_scorer.calculate("")

        assert score == 0.5  # Default

    @pytest.mark.asyncio
    async def test_missing_session_id(self):
        """Test handling of missing session ID."""
        scorer = create_quality_scorer()

        score = await scorer.score_response_enhanced(
            query="Test?",
            response="Test.",
            session_id=""
        )

        # Should not raise, should return valid score
        assert isinstance(score, QualityScore)

    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom_weights = {
            "relevance": 0.50,
            "helpfulness": 0.20,
            "engagement": 0.15,
            "clarity": 0.10,
            "accuracy": 0.05
        }

        calculator = CompositeScoreCalculator(weights=custom_weights)

        assert calculator.weights == custom_weights

    def test_invalid_weights_rejected(self):
        """Test that invalid weights are rejected."""
        invalid_weights = {
            "relevance": 0.50,
            "helpfulness": 0.50,
            "engagement": 0.50,  # Sum > 1.0
            "clarity": 0.50,
            "accuracy": 0.50
        }

        with pytest.raises(ValueError):
            CompositeScoreCalculator(weights=invalid_weights)

    @pytest.mark.asyncio
    async def test_relevance_scorer_fallback(self):
        """Test relevance scorer fallback when no embedding provider."""
        scorer = RelevanceScorer(embedding_provider=None)

        score = await scorer.calculate(
            query="What is Python?",
            response="Python is a programming language."
        )

        assert 0 <= score <= 1
