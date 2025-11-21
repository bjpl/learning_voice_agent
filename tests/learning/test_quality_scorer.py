"""
Test Suite: QualityScorer
=========================

30+ tests for quality scoring functionality.
"""

import pytest
from datetime import datetime

from app.learning.quality_scorer import QualityScorer
from app.learning.models import QualityScore, QualityDimension, Feedback, FeedbackType


class TestQualityScorerBasics:
    """Basic quality scoring tests."""

    @pytest.mark.asyncio
    async def test_score_response_returns_quality_score(self, quality_scorer):
        """Test that scoring returns a QualityScore object."""
        score = await quality_scorer.score_response(
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            session_id="test-session"
        )

        assert isinstance(score, QualityScore)

    @pytest.mark.asyncio
    async def test_score_response_all_dimensions_valid(self, quality_scorer):
        """Test that all dimension scores are in valid range."""
        score = await quality_scorer.score_response(
            query="What is AI?",
            response="AI is artificial intelligence.",
            session_id="test-session"
        )

        assert 0 <= score.relevance <= 1
        assert 0 <= score.helpfulness <= 1
        assert 0 <= score.engagement <= 1
        assert 0 <= score.clarity <= 1
        assert 0 <= score.accuracy <= 1
        assert 0 <= score.composite <= 1

    @pytest.mark.asyncio
    async def test_score_response_includes_metadata(self, quality_scorer):
        """Test that score includes query and response text."""
        query = "Test query"
        response = "Test response"

        score = await quality_scorer.score_response(
            query=query,
            response=response,
            session_id="test-session"
        )

        assert score.query_text == query
        assert score.response_text == response


class TestRelevanceScoring:
    """Tests for relevance dimension scoring."""

    @pytest.mark.asyncio
    async def test_high_relevance_for_matching_content(self, quality_scorer):
        """Test high relevance when query and response match."""
        score = await quality_scorer.score_response(
            query="Explain neural networks",
            response="Neural networks are computing systems inspired by biological neural networks in the brain.",
            session_id="test-session"
        )

        # Should have moderate to high relevance due to keyword overlap
        assert score.relevance > 0.4

    @pytest.mark.asyncio
    async def test_lower_relevance_for_mismatched_content(self, quality_scorer):
        """Test lower relevance for mismatched query-response."""
        score = await quality_scorer.score_response(
            query="What is quantum computing?",
            response="The weather today is sunny and warm.",
            session_id="test-session"
        )

        # Completely unrelated should have lower relevance
        assert score.relevance < 0.6

    @pytest.mark.asyncio
    async def test_relevance_with_keyword_overlap(self, quality_scorer):
        """Test relevance increases with keyword overlap."""
        score = await quality_scorer.score_response(
            query="Python programming language features",
            response="Python is a programming language known for its features like readability.",
            session_id="test-session"
        )

        assert score.relevance > 0.5


class TestHelpfulnessScoring:
    """Tests for helpfulness dimension scoring."""

    @pytest.mark.asyncio
    async def test_helpfulness_for_question_answer(self, quality_scorer):
        """Test helpfulness when response answers a question."""
        score = await quality_scorer.score_response(
            query="What is the capital of France?",
            response="Paris is the capital of France. It's located in the north of the country.",
            session_id="test-session"
        )

        assert score.helpfulness > 0.5

    @pytest.mark.asyncio
    async def test_low_helpfulness_for_just_question(self, quality_scorer):
        """Test lower helpfulness when response is just a question."""
        score = await quality_scorer.score_response(
            query="How do I learn Python?",
            response="What would you like to know about Python?",
            session_id="test-session"
        )

        # Answering a question with a question is less helpful
        assert score.helpfulness < 0.7

    @pytest.mark.asyncio
    async def test_helpfulness_for_actionable_content(self, quality_scorer):
        """Test helpfulness for actionable responses."""
        score = await quality_scorer.score_response(
            query="How to create a list in Python?",
            response="Here's how you can create a list: my_list = [1, 2, 3]. Try using square brackets.",
            session_id="test-session"
        )

        assert score.helpfulness > 0.5

    @pytest.mark.asyncio
    async def test_low_helpfulness_for_very_short(self, quality_scorer):
        """Test lower helpfulness for very short responses."""
        score = await quality_scorer.score_response(
            query="Explain machine learning in detail",
            response="It's AI.",
            session_id="test-session"
        )

        assert score.helpfulness < 0.5


class TestEngagementScoring:
    """Tests for engagement dimension scoring."""

    @pytest.mark.asyncio
    async def test_engagement_with_feedback_history(self, quality_scorer, sample_feedback_list):
        """Test engagement scoring with positive feedback history."""
        feedback_history = sample_feedback_list(n=5, positive_ratio=0.8)

        score = await quality_scorer.score_response(
            query="Test query",
            response="Test response",
            session_id="test-session",
            feedback_history=feedback_history
        )

        # With mostly positive feedback, engagement should be higher
        assert score.engagement > 0.5

    @pytest.mark.asyncio
    async def test_engagement_without_feedback(self, quality_scorer):
        """Test engagement scoring without feedback history."""
        score = await quality_scorer.score_response(
            query="Test query",
            response="This is an interesting topic to explore.",
            session_id="test-session"
        )

        # Should still produce a valid score
        assert 0 <= score.engagement <= 1

    @pytest.mark.asyncio
    async def test_engagement_for_interactive_response(self, quality_scorer):
        """Test higher engagement for interactive responses."""
        score = await quality_scorer.score_response(
            query="How does this work?",
            response="Great question! Let me explain. What do you think about trying it yourself?",
            session_id="test-session"
        )

        assert score.engagement > 0.4


class TestClarityScoring:
    """Tests for clarity dimension scoring."""

    @pytest.mark.asyncio
    async def test_clarity_for_well_structured_response(self, quality_scorer):
        """Test clarity for well-structured responses."""
        score = await quality_scorer.score_response(
            query="What are the benefits of exercise?",
            response="""Exercise has many benefits:

- Improves cardiovascular health
- Reduces stress
- Increases energy levels

Regular exercise is recommended for everyone.""",
            session_id="test-session"
        )

        assert score.clarity > 0.5

    @pytest.mark.asyncio
    async def test_low_clarity_for_complex_text(self, quality_scorer):
        """Test lower clarity for overly complex text."""
        complex_response = (
            "The epistemological ramifications of superintelligent artificial general "
            "intelligence necessitate unprecedented reconceptualization of anthropocentric "
            "paradigms vis-a-vis technological singularity hypotheses and their concomitant "
            "societal perturbations."
        )

        score = await quality_scorer.score_response(
            query="What is AI?",
            response=complex_response,
            session_id="test-session"
        )

        assert score.clarity < 0.7

    @pytest.mark.asyncio
    async def test_clarity_penalizes_long_sentences(self, quality_scorer):
        """Test that very long sentences reduce clarity."""
        long_sentence = " ".join(["word"] * 50) + "."

        score = await quality_scorer.score_response(
            query="Test",
            response=long_sentence,
            session_id="test-session"
        )

        assert score.clarity < 0.6


class TestAccuracyScoring:
    """Tests for accuracy dimension scoring."""

    @pytest.mark.asyncio
    async def test_accuracy_with_confident_response(self, quality_scorer):
        """Test accuracy for confident responses."""
        score = await quality_scorer.score_response(
            query="What is 2+2?",
            response="The answer is 4. This is a mathematical fact.",
            session_id="test-session"
        )

        assert score.accuracy > 0.5

    @pytest.mark.asyncio
    async def test_accuracy_with_hedging_language(self, quality_scorer):
        """Test lower accuracy with excessive hedging."""
        score = await quality_scorer.score_response(
            query="What is Python?",
            response="I'm not sure, but Python might be a programming language. Perhaps it could be used for coding, possibly.",
            session_id="test-session"
        )

        # Excessive hedging should reduce accuracy score
        assert score.accuracy < 0.7

    @pytest.mark.asyncio
    async def test_accuracy_with_source_citation(self, quality_scorer):
        """Test higher accuracy with source citations."""
        score = await quality_scorer.score_response(
            query="What is the speed of light?",
            response="According to physics, the speed of light is approximately 299,792,458 meters per second.",
            session_id="test-session"
        )

        assert score.accuracy > 0.5


class TestCompositeScoring:
    """Tests for composite score calculation."""

    @pytest.mark.asyncio
    async def test_composite_is_weighted_average(self, quality_scorer):
        """Test that composite score is weighted average of dimensions."""
        score = await quality_scorer.score_response(
            query="Test query",
            response="Test response for composite calculation.",
            session_id="test-session"
        )

        weights = quality_scorer.get_dimension_weights()
        expected_composite = (
            weights[QualityDimension.RELEVANCE] * score.relevance +
            weights[QualityDimension.HELPFULNESS] * score.helpfulness +
            weights[QualityDimension.ENGAGEMENT] * score.engagement +
            weights[QualityDimension.CLARITY] * score.clarity +
            weights[QualityDimension.ACCURACY] * score.accuracy
        ) / sum(weights.values())

        assert abs(score.composite - expected_composite) < 0.01

    @pytest.mark.asyncio
    async def test_set_custom_weights(self, quality_scorer):
        """Test setting custom dimension weights."""
        custom_weights = {
            QualityDimension.RELEVANCE: 0.40,
            QualityDimension.HELPFULNESS: 0.30,
            QualityDimension.ENGAGEMENT: 0.10,
            QualityDimension.CLARITY: 0.10,
            QualityDimension.ACCURACY: 0.10
        }

        quality_scorer.set_dimension_weights(custom_weights)
        weights = quality_scorer.get_dimension_weights()

        assert weights[QualityDimension.RELEVANCE] == 0.40


class TestQualityThresholds:
    """Tests for quality threshold methods."""

    @pytest.mark.asyncio
    async def test_is_high_quality(self, quality_scorer, sample_quality_score):
        """Test high quality detection."""
        high_score = sample_quality_score(composite=0.85)
        low_score = sample_quality_score(composite=0.50)

        assert high_score.is_high_quality() is True
        assert low_score.is_high_quality() is False

    @pytest.mark.asyncio
    async def test_is_low_quality(self, quality_scorer, sample_quality_score):
        """Test low quality detection."""
        high_score = sample_quality_score(composite=0.85)
        low_score = sample_quality_score(composite=0.30)

        assert high_score.is_low_quality() is False
        assert low_score.is_low_quality() is True

    @pytest.mark.asyncio
    async def test_get_weakest_dimension(self, sample_quality_score):
        """Test identifying weakest dimension."""
        score = sample_quality_score(
            relevance=0.8,
            helpfulness=0.7,
            engagement=0.3,  # Weakest
            clarity=0.9,
            accuracy=0.6
        )

        weakest = score.get_weakest_dimension()
        assert weakest == QualityDimension.ENGAGEMENT


class TestBatchScoring:
    """Tests for batch scoring functionality."""

    @pytest.mark.asyncio
    async def test_batch_score_multiple(self, quality_scorer):
        """Test batch scoring of multiple interactions."""
        interactions = [
            {"query": "What is AI?", "response": "AI is artificial intelligence."},
            {"query": "What is ML?", "response": "ML is machine learning."},
            {"query": "What is DL?", "response": "DL is deep learning."}
        ]

        scores = await quality_scorer.batch_score(interactions, "test-session")

        assert len(scores) == 3
        for score in scores:
            assert isinstance(score, QualityScore)

    @pytest.mark.asyncio
    async def test_batch_score_empty_list(self, quality_scorer):
        """Test batch scoring with empty list."""
        scores = await quality_scorer.batch_score([], "test-session")
        assert len(scores) == 0


class TestScoringPerformance:
    """Performance tests for quality scoring."""

    @pytest.mark.asyncio
    async def test_scoring_performance(self, quality_scorer, benchmark):
        """Test that scoring completes within performance target."""
        with benchmark() as b:
            await quality_scorer.score_response(
                query="What is machine learning?",
                response="Machine learning is a subset of artificial intelligence.",
                session_id="test-session"
            )

        b.assert_under(200, "Quality scoring should complete in under 200ms")
