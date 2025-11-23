"""
Unit Tests for Conversation Handler Module
Tests Claude integration, intent detection, and response generation
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestConversationHandler:
    """Test suite for ConversationHandler class"""

    @pytest.mark.unit
    def test_system_prompt_creation(self):
        """Test that system prompt can be generated correctly"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            # Test the basic system prompt method
            basic_prompt = handler._create_basic_system_prompt()
            assert basic_prompt is not None
            assert "learning companion" in basic_prompt.lower()
            assert "clarifying question" in basic_prompt.lower()
            assert "3 sentences" in basic_prompt

    @pytest.mark.unit
    def test_format_context_empty(self):
        """Test context formatting with empty exchanges"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            result = handler._format_context([])

            assert "start of our conversation" in result

    @pytest.mark.unit
    def test_format_context_with_exchanges(self, sample_context):
        """Test context formatting with conversation history"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            result = handler._format_context(sample_context)

            assert "Previous conversation" in result
            assert "User:" in result
            assert "You:" in result
            assert "machine learning" in result

    @pytest.mark.unit
    def test_should_add_followup_short_input_no_question(self):
        """Test followup detection for short inputs without questions"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            # Short input, response without question
            result = handler._should_add_followup("okay", "Got it")
            assert result is True

    @pytest.mark.unit
    def test_should_add_followup_with_question(self):
        """Test followup detection when response has question"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            # Response already contains question
            result = handler._should_add_followup("okay", "What would you like to learn?")
            assert result is False

    @pytest.mark.unit
    def test_should_add_followup_ending_response(self):
        """Test followup detection for ending responses"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            # Ending responses should not get followup
            result = handler._should_add_followup("bye", "Goodbye!")
            assert result is False

    @pytest.mark.unit
    def test_should_add_followup_long_input(self):
        """Test followup detection for long inputs"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            # Long input should not trigger followup
            result = handler._should_add_followup(
                "I am learning about machine learning concepts",
                "That's great"
            )
            assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_anthropic_client):
        """Test successful response generation"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            handler.client = mock_anthropic_client

            response = await handler.generate_response(
                "I want to learn Python",
                [],
                None
            )

            assert response is not None
            assert len(response) > 0
            mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, mock_anthropic_client, sample_context):
        """Test response generation with conversation context"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            handler.client = mock_anthropic_client

            response = await handler.generate_response(
                "Tell me more about backpropagation",
                sample_context,
                None
            )

            assert response is not None
            # Verify the API was called
            mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_rate_limit_error(self):
        """Test handling of rate limit errors"""
        import anthropic

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=anthropic.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body={}
        ))

        with patch('app.conversation_handler.anthropic.AsyncAnthropic', return_value=mock_client):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            handler.client = mock_client

            response = await handler.generate_response("Hello", [], None)

            assert "moment" in response.lower()
            assert "repeat" in response.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_response_api_error(self):
        """Test handling of API errors"""
        import anthropic

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=anthropic.APIError(
            message="API Error",
            request=MagicMock(),
            body={}
        ))

        with patch('app.conversation_handler.anthropic.AsyncAnthropic', return_value=mock_client):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            handler.client = mock_client

            response = await handler.generate_response("Hello", [], None)

            assert "trouble connecting" in response.lower()


class TestIntentDetection:
    """Test suite for intent detection"""

    @pytest.mark.unit
    def test_detect_intent_end_conversation(self):
        """Test detection of end conversation intent"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            assert handler.detect_intent("goodbye") == "end_conversation"
            assert handler.detect_intent("bye") == "end_conversation"
            assert handler.detect_intent("see you later") == "end_conversation"
            assert handler.detect_intent("talk to you later") == "end_conversation"

    @pytest.mark.unit
    def test_detect_intent_question(self):
        """Test detection of question intent"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            assert handler.detect_intent("what is machine learning?") == "question"
            assert handler.detect_intent("why does this happen?") == "question"
            assert handler.detect_intent("how do neural networks work?") == "question"
            assert handler.detect_intent("when should I use this?") == "question"
            assert handler.detect_intent("where can I learn more?") == "question"
            assert handler.detect_intent("who invented this?") == "question"

    @pytest.mark.unit
    def test_detect_intent_listing(self):
        """Test detection of listing intent"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            assert handler.detect_intent("first, I need to understand basics") == "listing"
            assert handler.detect_intent("second point is important") == "listing"
            assert handler.detect_intent("next, we should consider") == "listing"
            assert handler.detect_intent("then I want to learn") == "listing"
            assert handler.detect_intent("also, I'm interested in") == "listing"

    @pytest.mark.unit
    def test_detect_intent_reflection(self):
        """Test detection of reflection intent"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            assert handler.detect_intent("I realize now that") == "reflection"
            assert handler.detect_intent("I think this means") == "reflection"
            assert handler.detect_intent("I feel like this is important") == "reflection"
            assert handler.detect_intent("I believe we should") == "reflection"
            assert handler.detect_intent("I understand the concept") == "reflection"

    @pytest.mark.unit
    def test_detect_intent_statement(self):
        """Test detection of default statement intent"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            assert handler.detect_intent("Machine learning is interesting") == "statement"
            assert handler.detect_intent("The model works well") == "statement"
            assert handler.detect_intent("Python is my favorite language") == "statement"


class TestSummaryCreation:
    """Test suite for summary creation"""

    @pytest.mark.unit
    def test_create_summary_empty(self):
        """Test summary creation with empty exchanges"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            summary = handler.create_summary([])

            assert "didn't get to explore" in summary

    @pytest.mark.unit
    def test_create_summary_with_topics(self, sample_context):
        """Test summary creation with conversation topics"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()
            summary = handler.create_summary(sample_context)

            assert "explored" in summary.lower() or "conversation" in summary.lower()

    @pytest.mark.unit
    def test_create_summary_extracts_topics(self):
        """Test that summary extracts key topics"""
        with patch('app.conversation_handler.anthropic.AsyncAnthropic'):
            from app.conversation_handler import ConversationHandler

            handler = ConversationHandler()

            exchanges = [
                {"user": "I'm learning about machine learning algorithms", "agent": "Great!"},
                {"user": "Neural networks are fascinating", "agent": "Indeed!"},
                {"user": "Machine learning has many applications", "agent": "Yes!"}
            ]

            summary = handler.create_summary(exchanges)

            # Should extract repeated/important words
            assert len(summary) > 0


class TestConversationHandlerSingleton:
    """Test singleton instance"""

    @pytest.mark.unit
    def test_conversation_handler_singleton(self):
        """Test that conversation_handler singleton is available"""
        from app.conversation_handler import conversation_handler

        assert conversation_handler is not None
        assert hasattr(conversation_handler, 'generate_response')
        assert hasattr(conversation_handler, 'detect_intent')
        assert hasattr(conversation_handler, 'create_summary')
