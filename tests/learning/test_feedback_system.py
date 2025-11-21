"""
Tests for the Feedback Collection System (Phase 5)
PATTERN: Comprehensive unit and integration tests
WHY: Ensure reliability of feedback collection and storage
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import os

from app.learning.feedback_models import (
    ExplicitFeedback,
    ImplicitFeedback,
    CorrectionFeedback,
    SessionFeedback,
    FeedbackStats,
    CorrectionType,
    FeedbackSentiment,
    ExplicitFeedbackRequest,
    ImplicitFeedbackRequest,
    CorrectionFeedbackRequest,
)
from app.learning.feedback_store import FeedbackStore
from app.learning.config import FeedbackConfig, feedback_config


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
async def feedback_store(temp_db_path):
    """Create and initialize a feedback store for testing."""
    store = FeedbackStore(db_path=temp_db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def sample_explicit_feedback():
    """Create sample explicit feedback for testing."""
    return ExplicitFeedback(
        session_id="test_session_123",
        exchange_id="exchange_456",
        rating=4,
        helpful=True,
        comment="This was very helpful!"
    )


@pytest.fixture
def sample_implicit_feedback():
    """Create sample implicit feedback for testing."""
    return ImplicitFeedback(
        session_id="test_session_123",
        response_time_ms=1500,
        user_response_time_ms=3000,
        engagement_duration_seconds=120,
        follow_up_count=2,
        copy_action=True,
        share_action=False
    )


@pytest.fixture
def sample_correction_feedback():
    """Create sample correction feedback for testing."""
    return CorrectionFeedback(
        session_id="test_session_123",
        original_text="What is machien learning?",
        corrected_text="What is machine learning?",
        correction_type=CorrectionType.SPELLING,
        edit_distance=2,
        edit_distance_ratio=0.08
    )


# ============================================================================
# FEEDBACK MODELS TESTS
# ============================================================================

class TestExplicitFeedback:
    """Tests for ExplicitFeedback model."""

    def test_create_explicit_feedback(self):
        """Test creating explicit feedback."""
        feedback = ExplicitFeedback(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=5,
            helpful=True,
            comment="Great response!"
        )

        assert feedback.session_id == "sess_123"
        assert feedback.exchange_id == "ex_456"
        assert feedback.rating == 5
        assert feedback.helpful is True
        assert feedback.comment == "Great response!"

    def test_sentiment_positive(self):
        """Test positive sentiment detection."""
        feedback = ExplicitFeedback(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=5,
            helpful=True
        )
        assert feedback.sentiment == FeedbackSentiment.POSITIVE

    def test_sentiment_negative(self):
        """Test negative sentiment detection."""
        feedback = ExplicitFeedback(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=2,
            helpful=False
        )
        assert feedback.sentiment == FeedbackSentiment.NEGATIVE

    def test_sentiment_neutral(self):
        """Test neutral sentiment detection."""
        feedback = ExplicitFeedback(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=3,
            helpful=True
        )
        assert feedback.sentiment == FeedbackSentiment.NEUTRAL

    def test_rating_validation(self):
        """Test rating value validation."""
        with pytest.raises(ValueError):
            ExplicitFeedback(
                session_id="sess_123",
                exchange_id="ex_456",
                rating=6,  # Invalid - should be 1-5
                helpful=True
            )

    def test_comment_sanitization(self):
        """Test comment whitespace sanitization."""
        feedback = ExplicitFeedback(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=4,
            helpful=True,
            comment="  Multiple   spaces   here  "
        )
        assert feedback.comment == "Multiple spaces here"


class TestImplicitFeedback:
    """Tests for ImplicitFeedback model."""

    def test_create_implicit_feedback(self):
        """Test creating implicit feedback."""
        feedback = ImplicitFeedback(
            session_id="sess_123",
            response_time_ms=1500,
            follow_up_count=2
        )

        assert feedback.session_id == "sess_123"
        assert feedback.response_time_ms == 1500
        assert feedback.follow_up_count == 2

    def test_engagement_score_optimal_response_time(self):
        """Test engagement score with optimal response time."""
        feedback = ImplicitFeedback(
            session_id="sess_123",
            response_time_ms=1500  # Optimal range: 500-2000ms
        )
        # Should get a high score for response time
        assert feedback.engagement_score >= 0.8

    def test_engagement_score_with_actions(self):
        """Test engagement score with copy/share actions."""
        feedback = ImplicitFeedback(
            session_id="sess_123",
            response_time_ms=1500,
            copy_action=True,
            share_action=True
        )
        # Should get a higher score with actions
        assert feedback.engagement_score >= 0.8

    def test_engagement_score_with_follow_ups(self):
        """Test engagement score with follow-up questions."""
        feedback = ImplicitFeedback(
            session_id="sess_123",
            response_time_ms=1500,
            follow_up_count=5
        )
        # Should increase score with follow-ups
        assert feedback.engagement_score >= 0.6

    def test_engagement_score_default(self):
        """Test default engagement score."""
        feedback = ImplicitFeedback(
            session_id="sess_123",
            response_time_ms=0
        )
        # With no signals, should be neutral
        score = feedback.engagement_score
        assert 0.0 <= score <= 1.0


class TestCorrectionFeedback:
    """Tests for CorrectionFeedback model."""

    def test_create_correction_feedback(self):
        """Test creating correction feedback."""
        correction = CorrectionFeedback(
            session_id="sess_123",
            original_text="What is machien learning?",
            corrected_text="What is machine learning?",
            correction_type=CorrectionType.SPELLING,
            edit_distance=2,
            edit_distance_ratio=0.08
        )

        assert correction.session_id == "sess_123"
        assert correction.correction_type == CorrectionType.SPELLING
        assert correction.edit_distance == 2

    def test_severity_minor(self):
        """Test minor severity detection."""
        correction = CorrectionFeedback(
            session_id="sess_123",
            original_text="test",
            corrected_text="tests",
            correction_type=CorrectionType.SPELLING,
            edit_distance_ratio=0.1
        )
        assert correction.severity == "minor"

    def test_severity_moderate(self):
        """Test moderate severity detection."""
        correction = CorrectionFeedback(
            session_id="sess_123",
            original_text="test original",
            corrected_text="different text",
            correction_type=CorrectionType.REPHRASE,
            edit_distance_ratio=0.4
        )
        assert correction.severity == "moderate"

    def test_severity_major(self):
        """Test major severity detection."""
        correction = CorrectionFeedback(
            session_id="sess_123",
            original_text="completely different",
            corrected_text="totally changed text here",
            correction_type=CorrectionType.REPHRASE,
            edit_distance_ratio=0.6
        )
        assert correction.severity == "major"

    def test_texts_must_differ(self):
        """Test that corrected text must differ from original."""
        with pytest.raises(ValueError):
            CorrectionFeedback(
                session_id="sess_123",
                original_text="same text",
                corrected_text="same text",
                correction_type=CorrectionType.SPELLING
            )


# ============================================================================
# FEEDBACK STORE TESTS
# ============================================================================

class TestFeedbackStore:
    """Tests for FeedbackStore persistence layer."""

    @pytest.mark.asyncio
    async def test_store_initialization(self, temp_db_path):
        """Test store initialization creates tables."""
        store = FeedbackStore(db_path=temp_db_path)
        await store.initialize()

        assert store._initialized is True
        await store.close()

    @pytest.mark.asyncio
    async def test_save_explicit_feedback(self, feedback_store, sample_explicit_feedback):
        """Test saving explicit feedback."""
        feedback_id = await feedback_store.save_explicit(sample_explicit_feedback)

        assert feedback_id is not None
        assert len(feedback_id) > 0

    @pytest.mark.asyncio
    async def test_get_explicit_by_session(self, feedback_store, sample_explicit_feedback):
        """Test retrieving explicit feedback by session."""
        # Save feedback
        await feedback_store.save_explicit(sample_explicit_feedback)

        # Retrieve
        results = await feedback_store.get_explicit_by_session(
            sample_explicit_feedback.session_id
        )

        assert len(results) == 1
        assert results[0].rating == sample_explicit_feedback.rating
        assert results[0].helpful == sample_explicit_feedback.helpful

    @pytest.mark.asyncio
    async def test_save_implicit_feedback(self, feedback_store, sample_implicit_feedback):
        """Test saving implicit feedback."""
        feedback_id = await feedback_store.save_implicit(sample_implicit_feedback)

        assert feedback_id is not None

    @pytest.mark.asyncio
    async def test_save_implicit_batch(self, feedback_store):
        """Test saving multiple implicit feedback items."""
        feedback_list = [
            ImplicitFeedback(
                session_id="batch_session",
                response_time_ms=1000 + i * 100
            )
            for i in range(5)
        ]

        feedback_ids = await feedback_store.save_implicit_batch(feedback_list)

        assert len(feedback_ids) == 5

    @pytest.mark.asyncio
    async def test_save_correction(self, feedback_store, sample_correction_feedback):
        """Test saving correction feedback."""
        correction_id = await feedback_store.save_correction(sample_correction_feedback)

        assert correction_id is not None

    @pytest.mark.asyncio
    async def test_get_corrections_by_session(self, feedback_store, sample_correction_feedback):
        """Test retrieving corrections by session."""
        await feedback_store.save_correction(sample_correction_feedback)

        results = await feedback_store.get_corrections_by_session(
            sample_correction_feedback.session_id
        )

        assert len(results) == 1
        assert results[0].correction_type == sample_correction_feedback.correction_type

    @pytest.mark.asyncio
    async def test_get_session_feedback(self, feedback_store, sample_explicit_feedback):
        """Test getting aggregated session feedback."""
        # Save some feedback
        await feedback_store.save_explicit(sample_explicit_feedback)

        session_feedback = await feedback_store.get_session_feedback(
            sample_explicit_feedback.session_id
        )

        assert session_feedback.session_id == sample_explicit_feedback.session_id
        assert session_feedback.explicit_feedback_count == 1

    @pytest.mark.asyncio
    async def test_get_aggregate_stats(self, feedback_store, sample_explicit_feedback):
        """Test getting aggregate statistics."""
        # Save feedback
        await feedback_store.save_explicit(sample_explicit_feedback)

        stats = await feedback_store.get_aggregate_stats()

        assert stats.total_explicit_feedback >= 1

    @pytest.mark.asyncio
    async def test_get_feedback_count(self, feedback_store, sample_explicit_feedback):
        """Test getting feedback count."""
        await feedback_store.save_explicit(sample_explicit_feedback)

        count = await feedback_store.get_feedback_count(
            session_id=sample_explicit_feedback.session_id,
            feedback_type='explicit'
        )

        assert count == 1


# ============================================================================
# FEEDBACK CONFIG TESTS
# ============================================================================

class TestFeedbackConfig:
    """Tests for FeedbackConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeedbackConfig()

        assert config.min_rating == 1
        assert config.max_rating == 5
        assert config.quick_response_threshold_ms == 2000
        assert config.correction_edit_distance_threshold == 0.3

    def test_rating_range(self):
        """Test getting rating range."""
        config = FeedbackConfig()
        min_rating, max_rating = config.get_rating_range()

        assert min_rating == 1
        assert max_rating == 5

    def test_is_quick_response(self):
        """Test quick response detection."""
        config = FeedbackConfig()

        assert config.is_quick_response(1000) is True
        assert config.is_quick_response(3000) is False

    def test_is_slow_response(self):
        """Test slow response detection."""
        config = FeedbackConfig()

        assert config.is_slow_response(15000) is True
        assert config.is_slow_response(5000) is False

    def test_is_correction(self):
        """Test correction detection threshold."""
        config = FeedbackConfig()

        assert config.is_correction(0.5) is True
        assert config.is_correction(0.2) is False


# ============================================================================
# REQUEST MODEL TESTS
# ============================================================================

class TestRequestModels:
    """Tests for API request models."""

    def test_explicit_feedback_request(self):
        """Test ExplicitFeedbackRequest model."""
        request = ExplicitFeedbackRequest(
            session_id="sess_123",
            exchange_id="ex_456",
            rating=5,
            helpful=True,
            comment="Great!"
        )

        assert request.session_id == "sess_123"
        assert request.rating == 5

    def test_implicit_feedback_request(self):
        """Test ImplicitFeedbackRequest model."""
        request = ImplicitFeedbackRequest(
            session_id="sess_123",
            response_time_ms=1500,
            follow_up_count=2
        )

        assert request.session_id == "sess_123"
        assert request.response_time_ms == 1500

    def test_correction_feedback_request(self):
        """Test CorrectionFeedbackRequest model."""
        request = CorrectionFeedbackRequest(
            session_id="sess_123",
            original_text="original",
            corrected_text="corrected"
        )

        assert request.session_id == "sess_123"
        assert request.original_text == "original"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFeedbackIntegration:
    """Integration tests for the feedback system."""

    @pytest.mark.asyncio
    async def test_full_feedback_workflow(self, feedback_store):
        """Test complete feedback collection workflow."""
        session_id = "integration_test_session"

        # 1. Submit explicit feedback
        explicit = ExplicitFeedback(
            session_id=session_id,
            exchange_id="ex_1",
            rating=5,
            helpful=True,
            comment="Excellent response!"
        )
        await feedback_store.save_explicit(explicit)

        # 2. Track implicit engagement
        implicit = ImplicitFeedback(
            session_id=session_id,
            response_time_ms=1200,
            follow_up_count=3,
            copy_action=True
        )
        await feedback_store.save_implicit(implicit)

        # 3. Log a correction
        correction = CorrectionFeedback(
            session_id=session_id,
            original_text="What is AI?",
            corrected_text="What is artificial intelligence?",
            correction_type=CorrectionType.ELABORATION,
            edit_distance=20,
            edit_distance_ratio=0.55
        )
        await feedback_store.save_correction(correction)

        # 4. Get session feedback
        session_feedback = await feedback_store.get_session_feedback(session_id)

        assert session_feedback.explicit_feedback_count == 1
        assert session_feedback.implicit_feedback_count == 1
        assert session_feedback.correction_count == 1
        assert session_feedback.average_rating == 5.0

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, feedback_store):
        """Test feedback across multiple sessions."""
        sessions = ["session_a", "session_b", "session_c"]

        for session_id in sessions:
            explicit = ExplicitFeedback(
                session_id=session_id,
                exchange_id="ex_1",
                rating=4,
                helpful=True
            )
            await feedback_store.save_explicit(explicit)

        stats = await feedback_store.get_aggregate_stats()

        assert stats.total_sessions >= 3
        assert stats.total_explicit_feedback >= 3

    @pytest.mark.asyncio
    async def test_rating_distribution(self, feedback_store):
        """Test rating distribution in aggregate stats."""
        session_id = "distribution_test"

        # Submit feedback with different ratings
        ratings = [1, 2, 3, 4, 5, 5, 4, 4, 5, 5]
        for i, rating in enumerate(ratings):
            feedback = ExplicitFeedback(
                session_id=session_id,
                exchange_id=f"ex_{i}",
                rating=rating,
                helpful=rating >= 3
            )
            await feedback_store.save_explicit(feedback)

        stats = await feedback_store.get_aggregate_stats()

        assert stats.rating_distribution is not None
        assert sum(stats.rating_distribution.values()) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
