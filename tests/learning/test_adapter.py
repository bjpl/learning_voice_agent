"""
Tests for ResponseAdapter - Adaptive Prompt and Response Customization

PATTERN: Comprehensive unit tests with mocking
WHY: Ensure adaptation logic works correctly in isolation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.learning.adapter import (
    ResponseAdapter,
    UserPreferences,
    AdaptationResult,
    ImprovementSuggestion,
    create_response_adapter,
)
from app.learning.config import AdaptationConfig
from app.learning.store import LearningStore, FeedbackRecord


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_store():
    """Create mock learning store"""
    store = AsyncMock(spec=LearningStore)
    store.get_preferences = AsyncMock(return_value=None)
    store.save_preferences = AsyncMock(return_value=True)
    store.get_feedback_history = AsyncMock(return_value=[])
    store.get_vocabulary_adjustments = AsyncMock(return_value={})
    return store


@pytest.fixture
def config():
    """Create test configuration"""
    return AdaptationConfig(
        max_response_adjustments=3,
        min_confidence_for_adaptation=0.6,
        enable_prompt_adaptation=True,
        enable_context_enhancement=True,
        enable_response_calibration=True,
        preference_cache_ttl=60
    )


@pytest.fixture
def adapter(mock_store, config):
    """Create ResponseAdapter with mock store"""
    return ResponseAdapter(store=mock_store, config=config)


@pytest.fixture
def sample_preferences():
    """Sample user preferences for testing"""
    return UserPreferences(
        response_length="short",
        technical_depth="beginner",
        communication_style="casual",
        topic_interests=["Python", "Machine Learning", "Data Science"],
        vocabulary_adjustments={"API": "application programming interface"},
        confidence_scores={
            "response_length_short": 0.8,
            "technical_depth_beginner": 0.7,
            "communication_style_casual": 0.75
        },
        interaction_count=50,
        last_updated=datetime.utcnow().isoformat()
    )


# =============================================================================
# UserPreferences Tests
# =============================================================================

class TestUserPreferences:
    """Tests for UserPreferences data class"""

    def test_default_values(self):
        """Test default preference values"""
        prefs = UserPreferences()

        assert prefs.response_length == "medium"
        assert prefs.technical_depth == "intermediate"
        assert prefs.communication_style == "balanced"
        assert prefs.topic_interests == []
        assert prefs.vocabulary_adjustments == {}
        assert prefs.interaction_count == 0

    def test_to_dict(self, sample_preferences):
        """Test conversion to dictionary"""
        data = sample_preferences.to_dict()

        assert data["response_length"] == "short"
        assert data["technical_depth"] == "beginner"
        assert data["communication_style"] == "casual"
        assert "Python" in data["topic_interests"]
        assert "API" in data["vocabulary_adjustments"]

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "response_length": "long",
            "technical_depth": "expert",
            "communication_style": "formal",
            "topic_interests": ["AI", "ML"],
            "vocabulary_adjustments": {"ML": "machine learning"},
            "confidence_scores": {"response_length_long": 0.9},
            "interaction_count": 100
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.response_length == "long"
        assert prefs.technical_depth == "expert"
        assert prefs.communication_style == "formal"
        assert "AI" in prefs.topic_interests
        assert prefs.interaction_count == 100

    def test_get_confidence(self, sample_preferences):
        """Test confidence retrieval"""
        conf = sample_preferences.get_confidence("response_length_short")
        assert conf == 0.8

        # Test default for unknown key
        unknown_conf = sample_preferences.get_confidence("unknown_key")
        assert unknown_conf == 0.5

    def test_is_confident(self, sample_preferences):
        """Test confidence threshold check"""
        assert sample_preferences.is_confident("response_length_short", threshold=0.7)
        assert not sample_preferences.is_confident("response_length_short", threshold=0.9)


# =============================================================================
# ResponseAdapter Tests
# =============================================================================

class TestResponseAdapter:
    """Tests for ResponseAdapter class"""

    @pytest.mark.asyncio
    async def test_adapt_prompt_with_preferences(self, adapter, mock_store, sample_preferences):
        """Test prompt adaptation with user preferences"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()

        base_prompt = "You are a helpful assistant."
        result = await adapter.adapt_prompt(
            base_prompt=base_prompt,
            session_id="test-session",
            query="Explain Python decorators"
        )

        assert result.success
        assert "User preferences:" in result.adapted_content
        # Either length instruction is applied OR topic interests are applied
        # (depends on confidence threshold being met)
        assert len(result.adaptations_applied) > 0
        # Check that at least some adaptation was applied
        assert any(
            "response_length" in a or "topic" in a or "technical" in a
            for a in result.adaptations_applied
        )

    @pytest.mark.asyncio
    async def test_adapt_prompt_disabled(self, mock_store):
        """Test that prompt adaptation respects disabled flag"""
        config = AdaptationConfig(enable_prompt_adaptation=False)
        adapter = ResponseAdapter(store=mock_store, config=config)

        base_prompt = "You are a helpful assistant."
        result = await adapter.adapt_prompt(
            base_prompt=base_prompt,
            session_id="test-session",
            query="Test query"
        )

        assert result.success
        assert result.adapted_content == base_prompt
        assert "none_disabled" in result.adaptations_applied

    @pytest.mark.asyncio
    async def test_adapt_prompt_no_preferences(self, adapter, mock_store):
        """Test prompt adaptation without stored preferences"""
        mock_store.get_preferences.return_value = None

        base_prompt = "You are a helpful assistant."
        result = await adapter.adapt_prompt(
            base_prompt=base_prompt,
            session_id="new-session",
            query="Test query"
        )

        assert result.success
        # Should return base prompt with no adaptations
        assert base_prompt in result.adapted_content

    @pytest.mark.asyncio
    async def test_enhance_context(self, adapter, mock_store, sample_preferences):
        """Test context enhancement"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()

        base_context = {"conversation_history": []}
        enhanced = await adapter.enhance_context(
            query="Tell me about Python",
            session_id="test-session",
            base_context=base_context
        )

        assert "user_preferences" in enhanced
        assert enhanced["user_preferences"]["response_length"] == "short"
        assert "relevant_topics" in enhanced
        assert "Python" in enhanced["relevant_topics"]

    @pytest.mark.asyncio
    async def test_enhance_context_disabled(self, mock_store):
        """Test context enhancement when disabled"""
        config = AdaptationConfig(enable_context_enhancement=False)
        adapter = ResponseAdapter(store=mock_store, config=config)

        base_context = {"key": "value"}
        enhanced = await adapter.enhance_context(
            query="Test",
            session_id="test-session",
            base_context=base_context
        )

        assert enhanced == base_context

    @pytest.mark.asyncio
    async def test_calibrate_response_vocabulary(self, adapter, mock_store, sample_preferences):
        """Test response calibration with vocabulary adjustments"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()

        response = "You can use the API to fetch data."
        result = await adapter.calibrate_response(
            response=response,
            session_id="test-session"
        )

        assert result.success
        assert "application programming interface" in result.adapted_content

    @pytest.mark.asyncio
    async def test_calibrate_response_no_changes(self, adapter, mock_store):
        """Test calibration with no vocabulary adjustments"""
        prefs = UserPreferences(vocabulary_adjustments={})
        mock_store.get_preferences.return_value = prefs.to_dict()

        response = "This is a test response."
        result = await adapter.calibrate_response(
            response=response,
            session_id="test-session"
        )

        assert result.success
        assert result.adapted_content == response
        assert "no_changes" in result.adaptations_applied

    @pytest.mark.asyncio
    async def test_get_improvement_suggestions(self, adapter, mock_store, sample_preferences):
        """Test improvement suggestion generation"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()
        mock_store.get_feedback_history.return_value = [
            FeedbackRecord(
                session_id="test-session",
                feedback_type="explicit",
                helpful=True,
                response_metadata={"word_count": 50}
            ),
            FeedbackRecord(
                session_id="test-session",
                feedback_type="explicit",
                helpful=True,
                response_metadata={"word_count": 40}
            ),
            FeedbackRecord(
                session_id="test-session",
                feedback_type="explicit",
                helpful=False,
                response_metadata={"word_count": 250}
            )
        ]

        suggestions = await adapter.get_improvement_suggestions(
            session_id="test-session",
            limit=5
        )

        assert isinstance(suggestions, list)
        # May or may not have suggestions based on patterns
        for suggestion in suggestions:
            assert isinstance(suggestion, ImprovementSuggestion)
            assert suggestion.dimension
            assert suggestion.confidence >= 0


# =============================================================================
# Instruction Generation Tests
# =============================================================================

class TestInstructionGeneration:
    """Tests for instruction generation helpers"""

    def test_get_length_instruction(self, adapter):
        """Test length instruction generation"""
        short = adapter._get_length_instruction("short")
        assert "concise" in short.lower()

        long = adapter._get_length_instruction("long")
        assert "detailed" in long.lower()

        unknown = adapter._get_length_instruction("unknown")
        assert unknown is None

    def test_get_depth_instruction(self, adapter):
        """Test depth instruction generation"""
        beginner = adapter._get_depth_instruction("beginner")
        assert "simply" in beginner.lower() or "jargon" in beginner.lower()

        expert = adapter._get_depth_instruction("expert")
        assert "technical" in expert.lower()

    def test_get_style_instruction(self, adapter):
        """Test style instruction generation"""
        formal = adapter._get_style_instruction("formal")
        assert "professional" in formal.lower()

        casual = adapter._get_style_instruction("casual")
        assert "friendly" in casual.lower() or "conversational" in casual.lower()

    def test_get_relevant_topics(self, adapter):
        """Test topic relevance matching"""
        query = "How do I use Python for data analysis?"
        topics = ["Python", "Java", "Data Science", "Machine Learning"]

        relevant = adapter._get_relevant_topics(query, topics)

        assert "Python" in relevant
        # Data Science should match due to "data"
        assert any("Data" in t for t in relevant)


# =============================================================================
# Caching Tests
# =============================================================================

class TestCaching:
    """Tests for preference caching"""

    @pytest.mark.asyncio
    async def test_preference_caching(self, adapter, mock_store, sample_preferences):
        """Test that preferences are cached"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()

        # First call - should hit store
        await adapter._get_preferences("test-session")
        assert mock_store.get_preferences.call_count == 1

        # Second call - should use cache
        await adapter._get_preferences("test-session")
        assert mock_store.get_preferences.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, adapter, mock_store, sample_preferences):
        """Test cache invalidation"""
        mock_store.get_preferences.return_value = sample_preferences.to_dict()

        # Populate cache
        await adapter._get_preferences("test-session")

        # Invalidate
        adapter.invalidate_cache("test-session")

        # Should hit store again
        await adapter._get_preferences("test-session")
        assert mock_store.get_preferences.call_count == 2


# =============================================================================
# Experiment Management Tests
# =============================================================================

class TestExperimentManagement:
    """Tests for experiment registration"""

    def test_register_experiment(self, adapter):
        """Test experiment registration"""
        adapter.register_experiment("session-1", "improvement-1")

        assert adapter.is_treatment_group("session-1")

    def test_unregister_experiment(self, adapter):
        """Test experiment unregistration"""
        adapter.register_experiment("session-1", "improvement-1")
        adapter.unregister_experiment("session-1")

        assert not adapter.is_treatment_group("session-1")


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactory:
    """Tests for factory function"""

    def test_create_response_adapter(self, mock_store, config):
        """Test factory function"""
        adapter = create_response_adapter(store=mock_store, config=config)

        assert isinstance(adapter, ResponseAdapter)
        assert adapter.store == mock_store
        assert adapter.config == config

    def test_create_response_adapter_defaults(self):
        """Test factory with defaults"""
        adapter = create_response_adapter()

        assert isinstance(adapter, ResponseAdapter)
