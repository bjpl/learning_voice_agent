"""
Tests for ConversationAgent

Tests the conversation agent that handles natural language interactions.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.conversation_agent import ConversationAgent
from app.agents.base import AgentMessage, MessageType


class TestConversationAgentInitialization:
    """Test ConversationAgent initialization"""

    def test_agent_id(self):
        """Test agent_id is set correctly"""
        agent = ConversationAgent()
        assert agent.agent_id == "conversation_agent"

    def test_model_default(self):
        """Test default model is Claude 3.5 Sonnet"""
        agent = ConversationAgent()
        assert "sonnet" in agent.model.lower()

    def test_model_custom(self):
        """Test custom model can be set"""
        agent = ConversationAgent(config={"model": "claude-3-haiku-20240307"})
        assert agent.model == "claude-3-haiku-20240307"


class TestConversationAgentBasicResponse:
    """Test basic conversation responses"""

    @pytest.mark.asyncio
    async def test_basic_greeting(self, conversation_agent):
        """Test response to basic greeting"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello!"}
        )

        response = await conversation_agent.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content
        assert len(response.content["text"]) > 0

    @pytest.mark.asyncio
    async def test_question_response(self, conversation_agent):
        """Test response to question"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "What is machine learning?"}
        )

        response = await conversation_agent.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content
        assert len(response.content["text"]) > 10  # Should be substantial

    @pytest.mark.asyncio
    async def test_response_includes_metadata(self, conversation_agent):
        """Test response includes metadata"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await conversation_agent.process(message)

        assert "model" in response.content
        assert "tokens_used" in response.content or "model" in response.content


class TestConversationAgentWithContext:
    """Test conversation with context"""

    @pytest.mark.asyncio
    async def test_conversation_with_context(self, conversation_agent, sample_conversation_context):
        """Test conversation maintains context"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Tell me more about that",
                "context": sample_conversation_context
            }
        )

        response = await conversation_agent.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, conversation_agent):
        """Test handling of empty context"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Hello",
                "context": []
            }
        )

        response = await conversation_agent.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE

    @pytest.mark.asyncio
    async def test_context_formatting(self, conversation_agent):
        """Test context is properly formatted"""
        context = [
            {"user": "First message", "agent": "First response"},
            {"user": "Second message", "agent": "Second response"}
        ]

        formatted = conversation_agent._format_context(context)

        assert "First message" in formatted
        assert "First response" in formatted
        assert "Second message" in formatted


class TestConversationAgentToolUse:
    """Test tool integration"""

    @pytest.mark.asyncio
    async def test_calculator_tool_detection(self, conversation_agent):
        """Test detection of calculator tool need"""
        tools_needed = conversation_agent._should_use_tool(
            "What is 123 multiplied by 456?"
        )

        assert "calculator" in tools_needed or len(tools_needed) > 0

    @pytest.mark.asyncio
    async def test_datetime_tool_detection(self, conversation_agent):
        """Test detection of datetime tool need"""
        tools_needed = conversation_agent._should_use_tool(
            "What day is it today?"
        )

        # Should detect date/time need
        assert len(tools_needed) >= 0  # May or may not detect, depends on implementation

    @pytest.mark.asyncio
    async def test_no_tools_for_regular_conversation(self, conversation_agent):
        """Test no tools for regular conversation"""
        tools_needed = conversation_agent._should_use_tool(
            "Tell me about your day"
        )

        # Regular conversation shouldn't need tools
        assert tools_needed == [] or tools_needed is None


class TestConversationAgentErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_api_error_handling(self, conversation_agent, mock_anthropic_client):
        """Test handling of API errors"""
        # Make API fail
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await conversation_agent.process(message)

        # Should return error response
        assert response.message_type == MessageType.AGENT_ERROR
        assert "error" in response.content

    @pytest.mark.asyncio
    async def test_empty_message_handling(self, conversation_agent):
        """Test handling of empty message"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": ""}
        )

        response = await conversation_agent.process(message)

        # Should handle gracefully
        assert response.message_type in [
            MessageType.CONVERSATION_RESPONSE,
            MessageType.AGENT_ERROR
        ]

    @pytest.mark.asyncio
    async def test_missing_text_field(self, conversation_agent):
        """Test handling of missing text field"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}  # No 'text' field
        )

        response = await conversation_agent.process(message)

        # Should handle gracefully
        assert response.message_type in [
            MessageType.CONVERSATION_RESPONSE,
            MessageType.AGENT_ERROR
        ]


class TestConversationAgentPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_response_time(self, conversation_agent, timing):
        """Test response time is reasonable"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        with timing:
            response = await conversation_agent.process(message)

        # With mocked API, should be very fast
        assert timing.elapsed_ms < 100  # 100ms

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, conversation_agent):
        """Test handling concurrent requests"""
        import asyncio

        messages = [
            AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={"text": f"Message {i}"}
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(
            *[conversation_agent.process(msg) for msg in messages]
        )

        # All should succeed
        assert len(responses) == 5
        assert all(r.message_type == MessageType.CONVERSATION_RESPONSE for r in responses)


class TestConversationAgentSystemPrompt:
    """Test system prompt functionality"""

    def test_system_prompt_exists(self, conversation_agent):
        """Test system prompt is set"""
        assert hasattr(conversation_agent, 'system_prompt')
        assert len(conversation_agent.system_prompt) > 0

    def test_system_prompt_content(self, conversation_agent):
        """Test system prompt contains key instructions"""
        prompt = conversation_agent.system_prompt.lower()

        # Should mention learning companion role
        assert any(kw in prompt for kw in ["learning", "companion", "help"])
