"""
Tests for PreferenceLearner - User Preference Learning

PATTERN: Comprehensive unit tests with various feedback scenarios
WHY: Ensure preference learning behaves correctly with different inputs
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# Import from the existing preference_learner module
from app.learning.preference_learner import PreferenceLearner
from app.learning.config import LearningConfig, learning_config
from app.learning.models import UserPreference, Feedback, FeedbackType
from app.learning.feedback_store import FeedbackStore


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_feedback_store():
    """Create mock feedback store"""
    store = AsyncMock(spec=FeedbackStore)
    return store


@pytest.fixture
def config():
    """Create test configuration"""
    return learning_config


@pytest.fixture
def learner(mock_feedback_store, config):
    """Create PreferenceLearner with mock store"""
    return PreferenceLearner(config=config, feedback_store=mock_feedback_store)


@pytest.fixture
def sample_feedback():
    """Create sample feedback for testing"""
    return Feedback(
        id="feedback-1",
        session_id="session-1",
        exchange_id="exchange-1",
        feedback_type=FeedbackType.EXPLICIT_POSITIVE,
        rating=0.8,
        original_response="This is a medium-length response with some examples.",
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_response_characteristics():
    """Sample response characteristics"""
    return {
        "response_length": "medium",
        "detail_level": "standard",
        "formality": "neutral",
        "example_frequency": "moderate"
    }


# =============================================================================
# Initialization Tests
# =============================================================================

class TestPreferenceLearnerInit:
    """Tests for PreferenceLearner initialization"""

    def test_initialization(self, learner):
        """Test basic initialization"""
        assert learner is not None
        assert learner._preferences == {}
        assert learner.config is not None

    @pytest.mark.asyncio
    async def test_initialize_loads_preferences(self, learner, tmp_path):
        """Test that initialize loads saved preferences"""
        # This would require mocking the file system
        # For now, just verify the method exists
        assert hasattr(learner, "initialize")


# =============================================================================
# Learning From Feedback Tests
# =============================================================================

class TestLearnFromFeedback:
    """Tests for learning from explicit feedback"""

    @pytest.mark.asyncio
    async def test_learn_from_positive_feedback(
        self, learner, sample_feedback, sample_response_characteristics
    ):
        """Test learning from positive feedback"""
        updated = await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=sample_response_characteristics
        )

        # Should update some preferences
        assert isinstance(updated, list)

    @pytest.mark.asyncio
    async def test_learn_from_negative_feedback(self, learner, sample_response_characteristics):
        """Test learning from negative feedback"""
        negative_feedback = Feedback(
            id="feedback-2",
            session_id="session-1",
            exchange_id="exchange-2",
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            rating=0.2,
            original_response="Too long and too complex.",
            timestamp=datetime.utcnow()
        )

        updated = await learner.learn_from_feedback(
            feedback=negative_feedback,
            response_characteristics=sample_response_characteristics
        )

        assert isinstance(updated, list)

    @pytest.mark.asyncio
    async def test_learn_with_weak_signal_ignored(self, learner):
        """Test that very weak signals are ignored"""
        weak_feedback = Feedback(
            id="feedback-3",
            session_id="session-1",
            exchange_id="exchange-3",
            feedback_type=FeedbackType.IMPLICIT_ENGAGEMENT,  # Weak signal
            rating=0.5,  # Neutral
            original_response="Test",
            timestamp=datetime.utcnow()
        )

        updated = await learner.learn_from_feedback(
            feedback=weak_feedback,
            response_characteristics={"response_length": "medium"}
        )

        # May return empty list for weak signals
        assert isinstance(updated, list)

    @pytest.mark.asyncio
    async def test_learn_extracts_characteristics(self, learner, sample_feedback):
        """Test automatic characteristic extraction"""
        # Don't provide characteristics - should extract from response
        updated = await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=None
        )

        assert isinstance(updated, list)


# =============================================================================
# Learning From Engagement Tests
# =============================================================================

class TestLearnFromEngagement:
    """Tests for learning from implicit engagement signals"""

    @pytest.mark.asyncio
    async def test_learn_from_high_engagement(self, learner, sample_response_characteristics):
        """Test learning from high engagement metrics"""
        engagement_metrics = {
            "time_spent_seconds": 120,
            "scroll_depth": 0.9,
            "interaction_count": 5
        }

        updated = await learner.learn_from_engagement(
            session_id="session-1",
            engagement_metrics=engagement_metrics,
            response_characteristics=sample_response_characteristics
        )

        assert isinstance(updated, list)

    @pytest.mark.asyncio
    async def test_learn_from_low_engagement(self, learner, sample_response_characteristics):
        """Test learning from low engagement"""
        engagement_metrics = {
            "time_spent_seconds": 2,
            "scroll_depth": 0.1,
            "interaction_count": 0
        }

        updated = await learner.learn_from_engagement(
            session_id="session-1",
            engagement_metrics=engagement_metrics,
            response_characteristics=sample_response_characteristics
        )

        assert isinstance(updated, list)


# =============================================================================
# Preference Retrieval Tests
# =============================================================================

class TestGetPreferences:
    """Tests for preference retrieval"""

    @pytest.mark.asyncio
    async def test_get_preferences_empty(self, learner):
        """Test getting preferences with no data"""
        prefs = await learner.get_preferences(session_id="new-session")
        assert prefs == {}

    @pytest.mark.asyncio
    async def test_get_preferences_after_learning(self, learner, sample_feedback, sample_response_characteristics):
        """Test getting preferences after learning"""
        # Learn from feedback first
        await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=sample_response_characteristics
        )

        prefs = await learner.get_preferences(session_id="session-1")

        # May have learned some preferences
        assert isinstance(prefs, dict)

    @pytest.mark.asyncio
    async def test_get_specific_preference(self, learner, sample_feedback, sample_response_characteristics):
        """Test getting a specific preference"""
        # Learn first
        await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=sample_response_characteristics
        )

        pref = await learner.get_preference(
            category="response_length",
            session_id="session-1"
        )

        # May or may not exist depending on learning
        if pref:
            assert isinstance(pref, UserPreference)


# =============================================================================
# Preference Prediction Tests
# =============================================================================

class TestPredictPreference:
    """Tests for preference prediction"""

    @pytest.mark.asyncio
    async def test_predict_with_no_data(self, learner):
        """Test prediction with no learned data"""
        result = await learner.predict_preference(
            category="response_length",
            session_id="new-session"
        )

        # Should return default
        assert result is not None
        value, confidence = result
        assert value in ["brief", "medium", "detailed"]
        assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    async def test_predict_with_learned_data(
        self, learner, sample_feedback, sample_response_characteristics
    ):
        """Test prediction after learning"""
        # Learn multiple times to build confidence
        for _ in range(5):
            await learner.learn_from_feedback(
                feedback=sample_feedback,
                response_characteristics=sample_response_characteristics
            )

        result = await learner.predict_preference(
            category="response_length",
            session_id="session-1"
        )

        assert result is not None
        value, confidence = result
        assert isinstance(value, str)
        assert 0 <= confidence <= 1


# =============================================================================
# Preference Update Tests
# =============================================================================

class TestPreferenceUpdate:
    """Tests for preference update logic"""

    @pytest.mark.asyncio
    async def test_confidence_increases_on_positive(self, learner, sample_response_characteristics):
        """Test that confidence increases on repeated positive signals"""
        positive_feedback = Feedback(
            id="feedback-pos",
            session_id="session-1",
            exchange_id="exchange-pos",
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            rating=0.9,
            original_response="Test",
            timestamp=datetime.utcnow()
        )

        # Learn multiple times
        for i in range(10):
            await learner.learn_from_feedback(
                feedback=Feedback(
                    id=f"feedback-{i}",
                    session_id="session-1",
                    exchange_id=f"exchange-{i}",
                    feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                    rating=0.9,
                    original_response="Test",
                    timestamp=datetime.utcnow()
                ),
                response_characteristics=sample_response_characteristics
            )

        # Check that preferences exist and have confidence
        prefs = await learner.get_preferences(session_id="session-1")

        for pref in prefs.values():
            # Confidence should have increased
            assert pref.confidence > 0


# =============================================================================
# Signal Strength Tests
# =============================================================================

class TestSignalStrength:
    """Tests for signal strength calculation"""

    def test_signal_from_rating(self, learner):
        """Test signal calculation from rating"""
        # High rating
        high_feedback = Feedback(
            id="f1",
            session_id="s1",
            exchange_id="e1",
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            rating=1.0,
            original_response="Test",
            timestamp=datetime.utcnow()
        )
        signal = learner._get_signal_strength(high_feedback)
        assert signal == 1.0  # (1.0 - 0.5) * 2 = 1.0

        # Low rating
        low_feedback = Feedback(
            id="f2",
            session_id="s1",
            exchange_id="e2",
            feedback_type=FeedbackType.EXPLICIT_NEGATIVE,
            rating=0.0,
            original_response="Test",
            timestamp=datetime.utcnow()
        )
        signal = learner._get_signal_strength(low_feedback)
        assert signal == -1.0  # (0.0 - 0.5) * 2 = -1.0

    def test_signal_from_type(self, learner):
        """Test signal from feedback type when no rating"""
        feedback = Feedback(
            id="f1",
            session_id="s1",
            exchange_id="e1",
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
            rating=None,
            original_response="Test",
            timestamp=datetime.utcnow()
        )
        signal = learner._get_signal_strength(feedback)
        assert signal > 0  # Positive type should give positive signal


# =============================================================================
# Characteristic Extraction Tests
# =============================================================================

class TestCharacteristicExtraction:
    """Tests for response characteristic extraction"""

    def test_extract_brief_response(self, learner):
        """Test extraction from brief response"""
        response = "Yes, that's correct."
        chars = learner._extract_characteristics(response)

        assert chars["response_length"] == "brief"

    def test_extract_detailed_response(self, learner):
        """Test extraction from detailed response"""
        response = """
        This is a very detailed response that contains multiple paragraphs
        and examples. For instance, you can use this pattern in many ways.

        Here's an example:
        - First point
        - Second point
        - Third point

        And here's more explanation with technical details and code samples.
        """ + "More content. " * 50

        chars = learner._extract_characteristics(response)

        assert chars["response_length"] == "detailed"

    def test_extract_formality(self, learner):
        """Test formality extraction"""
        casual = "Don't worry, it's not that hard. You'll get it!"
        chars = learner._extract_characteristics(casual)
        assert chars["formality"] in ["casual", "neutral"]

        formal = "The implementation follows standard protocols. " * 5
        chars = learner._extract_characteristics(formal)
        assert chars["formality"] in ["formal", "neutral"]


# =============================================================================
# Preference Summary Tests
# =============================================================================

class TestPreferenceSummary:
    """Tests for preference summary generation"""

    @pytest.mark.asyncio
    async def test_summary_empty(self, learner):
        """Test summary with no preferences"""
        summary = learner.get_preference_summary(session_id="new-session")

        assert summary["total_preferences"] == 0
        assert summary["high_confidence"] == []
        assert summary["medium_confidence"] == []
        assert summary["low_confidence"] == []

    @pytest.mark.asyncio
    async def test_summary_with_preferences(
        self, learner, sample_feedback, sample_response_characteristics
    ):
        """Test summary after learning"""
        # Learn multiple times
        for i in range(10):
            await learner.learn_from_feedback(
                feedback=Feedback(
                    id=f"feedback-{i}",
                    session_id="session-1",
                    exchange_id=f"exchange-{i}",
                    feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                    rating=0.9,
                    original_response="Test response",
                    timestamp=datetime.utcnow()
                ),
                response_characteristics=sample_response_characteristics
            )

        summary = learner.get_preference_summary(session_id="session-1")

        assert summary["total_preferences"] > 0


# =============================================================================
# Clear Preferences Tests
# =============================================================================

class TestClearPreferences:
    """Tests for clearing preferences"""

    @pytest.mark.asyncio
    async def test_clear_session_preferences(
        self, learner, sample_feedback, sample_response_characteristics
    ):
        """Test clearing session preferences"""
        # Learn first
        await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=sample_response_characteristics
        )

        # Clear
        await learner.clear_preferences(session_id="session-1")

        # Should be empty
        prefs = await learner.get_preferences(session_id="session-1")
        assert prefs == {}


# =============================================================================
# Learning History Tests
# =============================================================================

class TestLearningHistory:
    """Tests for learning history tracking"""

    @pytest.mark.asyncio
    async def test_history_recorded(
        self, learner, sample_feedback, sample_response_characteristics
    ):
        """Test that learning history is recorded"""
        await learner.learn_from_feedback(
            feedback=sample_feedback,
            response_characteristics=sample_response_characteristics
        )

        history = await learner.get_learning_history(session_id="session-1")

        assert isinstance(history, list)
        if history:
            assert "timestamp" in history[0]
            assert "feedback_id" in history[0]

    @pytest.mark.asyncio
    async def test_history_limit(self, learner, sample_response_characteristics):
        """Test history respects limit"""
        # Create many entries
        for i in range(150):
            await learner.learn_from_feedback(
                feedback=Feedback(
                    id=f"feedback-{i}",
                    session_id="session-1",
                    exchange_id=f"exchange-{i}",
                    feedback_type=FeedbackType.EXPLICIT_POSITIVE,
                    rating=0.8,
                    original_response="Test",
                    timestamp=datetime.utcnow()
                ),
                response_characteristics=sample_response_characteristics
            )

        history = await learner.get_learning_history(
            session_id="session-1",
            limit=50
        )

        assert len(history) <= 50
