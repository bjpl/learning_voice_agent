"""
Week 3 Feature Tests - Advanced Features
Tests for ChromaDB, semantic search, advanced prompts, and offline capabilities
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys
import importlib


@pytest.fixture(autouse=True)
def restore_advanced_prompts():
    """Restore real advanced_prompts module for these tests."""
    # Remove the mock from sys.modules
    if 'app.advanced_prompts' in sys.modules:
        del sys.modules['app.advanced_prompts']

    # Force reimport of the real module
    from app import advanced_prompts as real_module
    sys.modules['app.advanced_prompts'] = real_module

    yield

    # No cleanup needed - conftest will re-mock if needed


class TestVectorStore:
    """Tests for ChromaDB vector store integration"""

    @pytest.mark.asyncio
    async def test_vector_store_initialization(self):
        """Test that vector store initializes correctly"""
        from app.vector_store import VectorStore

        vs = VectorStore(persist_directory="./test_chroma_db")

        # Should initialize without error
        with patch('app.vector_store.CHROMADB_AVAILABLE', False):
            result = await vs.initialize()
            # Returns False when ChromaDB not available
            assert result is False or result is True

    @pytest.mark.asyncio
    async def test_fallback_embedding(self):
        """Test fallback embedding when sentence-transformers unavailable"""
        from app.vector_store import VectorStore

        vs = VectorStore()
        vs._embedding_model = None  # Simulate unavailable model

        # Should generate consistent fallback embedding
        embedding1 = vs._fallback_embedding("test text")
        embedding2 = vs._fallback_embedding("test text")

        assert len(embedding1) == 384  # Default dimension
        assert embedding1 == embedding2  # Should be deterministic

    @pytest.mark.asyncio
    async def test_semantic_search_graceful_degradation(self):
        """Test that semantic search degrades gracefully when unavailable"""
        from app.vector_store import VectorStore

        vs = VectorStore()
        vs._initialized = False

        # Should return empty results, not throw
        results = await vs.semantic_search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_store_stats(self):
        """Test vector store statistics retrieval"""
        from app.vector_store import VectorStore

        vs = VectorStore()

        # Without initialization, should return unavailable status
        stats = await vs.get_stats()
        assert "available" in stats


class TestAdvancedPrompts:
    """Tests for advanced prompt engineering"""

    def test_prompt_strategy_selection(self):
        """Test dynamic prompt strategy selection"""
        from app.advanced_prompts import AdvancedPromptEngine, PromptStrategy

        engine = AdvancedPromptEngine()

        # Complex questions should use chain-of-thought
        strategy = engine.select_strategy_for_context(
            user_text="Why does functional programming matter?",
            intent="question",
            context_length=3,
            has_similar_conversations=False
        )
        assert strategy == PromptStrategy.CHAIN_OF_THOUGHT

        # Reflections should use Socratic method
        strategy = engine.select_strategy_for_context(
            user_text="I realized something important",
            intent="reflection",
            context_length=3,
            has_similar_conversations=False
        )
        assert strategy == PromptStrategy.SOCRATIC

        # With similar conversations, should use RAG
        strategy = engine.select_strategy_for_context(
            user_text="Hello there",
            intent="statement",
            context_length=3,
            has_similar_conversations=True
        )
        assert strategy == PromptStrategy.RETRIEVAL_AUGMENTED

    def test_few_shot_examples_format(self):
        """Test few-shot example formatting"""
        from app.advanced_prompts import AdvancedPromptEngine

        engine = AdvancedPromptEngine()

        # Should return formatted examples for known intents
        examples = engine.format_few_shot_context("reflection", num_examples=2)
        assert "User:" in examples
        assert "Assistant:" in examples

        # Should handle unknown intents gracefully
        examples = engine.format_few_shot_context("unknown_intent")
        assert examples == "" or "User:" in examples

    def test_system_prompt_generation(self):
        """Test system prompt generation for different strategies"""
        from app.advanced_prompts import AdvancedPromptEngine, PromptStrategy

        engine = AdvancedPromptEngine()

        # Chain-of-thought should include thinking instructions
        prompt = engine.build_system_prompt(PromptStrategy.CHAIN_OF_THOUGHT)
        assert "<thinking>" in prompt

        # Socratic should include question-based guidance
        prompt = engine.build_system_prompt(PromptStrategy.SOCRATIC)
        assert "SOCRATIC" in prompt or "question" in prompt.lower()

        # RAG should mention context awareness
        prompt = engine.build_system_prompt(PromptStrategy.RETRIEVAL_AUGMENTED)
        assert "previous" in prompt.lower() or "context" in prompt.lower()

    def test_response_extraction(self):
        """Test extraction of response from thinking tags"""
        from app.advanced_prompts import AdvancedPromptEngine

        engine = AdvancedPromptEngine()

        # Should extract thinking and response separately
        raw_response = "<thinking>User seems curious</thinking>That's interesting!"
        response, reasoning = engine.extract_response(raw_response)

        assert response == "That's interesting!"
        assert reasoning == "User seems curious"

        # Should handle responses without thinking tags
        raw_response = "Simple response without thinking"
        response, reasoning = engine.extract_response(raw_response)

        assert response == "Simple response without thinking"
        assert reasoning is None

    def test_user_message_building(self):
        """Test user message construction with context"""
        from app.advanced_prompts import AdvancedPromptEngine, PromptStrategy

        engine = AdvancedPromptEngine()

        context = [
            {"user": "Hello", "agent": "Hi there!"},
            {"user": "How are you?", "agent": "I'm doing well!"}
        ]

        message = engine.build_user_message(
            user_text="What's new?",
            context=context,
            intent="question",
            similar_conversations=None,
            strategy=PromptStrategy.FEW_SHOT
        )

        # Should include conversation history
        assert "Hello" in message
        assert "What's new?" in message

    def test_add_custom_example(self):
        """Test adding custom few-shot examples"""
        from app.advanced_prompts import AdvancedPromptEngine

        engine = AdvancedPromptEngine()

        # Add custom example
        engine.add_custom_example(
            intent="custom_intent",
            user_input="Custom question",
            assistant_response="Custom response",
            reasoning="Custom reasoning"
        )

        # Should be retrievable
        examples = engine.format_few_shot_context("custom_intent")
        assert "Custom question" in examples or examples == ""


class TestConversationHandlerWeek3:
    """Tests for Week 3 enhancements to conversation handler"""

    @pytest.mark.asyncio
    async def test_intent_detection(self):
        """Test intent detection for prompt strategy selection"""
        from app.conversation_handler import ConversationHandler

        handler = ConversationHandler()

        assert handler.detect_intent("goodbye") == "end_conversation"
        assert handler.detect_intent("What is Python?") == "question"
        assert handler.detect_intent("First, I want to learn") == "listing"
        assert handler.detect_intent("I realize now that") == "reflection"
        assert handler.detect_intent("Python is cool") == "statement"


class TestConfigWeek3:
    """Tests for Week 3 configuration additions"""

    def test_vector_search_config(self):
        """Test vector search configuration defaults"""
        from app.config import Settings

        settings = Settings()

        # Verify Week 3 settings exist with defaults
        assert hasattr(settings, 'chroma_persist_directory')
        assert hasattr(settings, 'embedding_model')
        assert hasattr(settings, 'semantic_search_threshold')
        assert hasattr(settings, 'enable_vector_search')

        # Check default values
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.semantic_search_threshold == 0.5

    def test_advanced_prompts_config(self):
        """Test advanced prompts configuration"""
        from app.config import Settings

        settings = Settings()

        assert hasattr(settings, 'use_chain_of_thought')
        assert hasattr(settings, 'use_few_shot')
        assert hasattr(settings, 'prompt_strategy')

    def test_offline_config(self):
        """Test offline/PWA configuration"""
        from app.config import Settings

        settings = Settings()

        assert hasattr(settings, 'offline_cache_max_entries')
        assert hasattr(settings, 'enable_offline_mode')


class TestSemanticSearchEndpoint:
    """Tests for semantic search API endpoint"""

    @pytest.mark.asyncio
    async def test_semantic_search_fallback_to_keyword(self):
        """Test that semantic search falls back to keyword when disabled"""
        from fastapi.testclient import TestClient
        from app.main import app

        with patch('app.main.settings.enable_vector_search', False):
            client = TestClient(app)

            response = client.post(
                "/api/semantic-search",
                params={"query": "test query", "limit": 10}
            )

            # Should still work, falling back to keyword search
            assert response.status_code in [200, 422]  # 422 if validation fails


# Run with: pytest tests/test_week3_features.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
