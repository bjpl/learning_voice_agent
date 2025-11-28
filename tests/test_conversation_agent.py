"""
Integration Tests for ConversationAgent

SPECIFICATION:
- Test Claude 3.5 Sonnet integration
- Test tool calling functionality
- Test streaming responses
- Test error handling and resilience
- Test context management

PATTERN: Async test fixtures with mocking
WHY: Reliable tests without external API calls
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import anthropic

from app.agents.conversation_agent import ConversationAgent
from app.agents.base import AgentMessage, MessageRole, MessageType
from app.agents.tools import tool_registry


class MockAnthropicMessage:
    """Mock Anthropic API response"""

    def __init__(
        self,
        text: str = "Test response",
        stop_reason: str = "end_turn",
        input_tokens: int = 10,
        output_tokens: int = 20,
        tool_use: bool = False,
    ):
        self.stop_reason = stop_reason

        if tool_use:
            self.content = [
                Mock(
                    type="text",
                    text="Let me calculate that for you.",
                ),
                Mock(
                    type="tool_use",
                    id="tool_123",
                    name="calculate",
                    input={"expression": "2 + 2"},
                ),
            ]
        else:
            self.content = [
                Mock(type="text", text=text)
            ]

        self.usage = Mock(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@pytest.fixture
def conversation_agent():
    """Create a ConversationAgent instance for testing"""
    return ConversationAgent(
        model="claude-3-5-sonnet-20241022",
        enable_tools=True,
        enable_streaming=False,
    )


@pytest.fixture
def sample_message():
    """Create a sample agent message"""
    return AgentMessage(
        role=MessageRole.USER,
        content="What is 2 + 2?",
        message_type=MessageType.REQUEST,
        context={
            "session_id": "test_session_123",
            "conversation_history": [
                {"user": "Hello", "agent": "Hi! How can I help you today?"}
            ],
        },
    )


class TestConversationAgentInitialization:
    """Test ConversationAgent initialization"""

    def test_initialization_default_values(self):
        """Test agent initializes with default values"""
        agent = ConversationAgent()

        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.agent_name == "conversation_agent_v2"
        assert agent.enable_tools is True
        assert agent.enable_streaming is False
        assert len(agent.tools) == 4  # 4 default tools

    def test_initialization_custom_values(self):
        """Test agent initializes with custom values"""
        agent = ConversationAgent(
            model="claude-3-5-sonnet-20241022",
            agent_name="custom_agent",
            enable_tools=False,
            enable_streaming=True,
        )

        assert agent.model == "claude-3-5-sonnet-20241022"
        assert agent.agent_name == "custom_agent"
        assert agent.enable_tools is False
        assert agent.enable_streaming is True
        assert len(agent.tools) == 0

    def test_tools_loaded_from_registry(self, conversation_agent):
        """Test tools are loaded from registry"""
        tool_names = [tool["name"] for tool in conversation_agent.tools]

        assert "search_knowledge" in tool_names
        assert "calculate" in tool_names
        assert "get_datetime" in tool_names
        assert "memory_store" in tool_names


class TestConversationProcessing:
    """Test conversation message processing"""

    @pytest.mark.asyncio
    async def test_process_simple_message(self, conversation_agent, sample_message):
        """Test processing a simple message without tools"""
        mock_response = MockAnthropicMessage(text="The answer is 4.")

        with patch.object(
            conversation_agent.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await conversation_agent.process(sample_message)

            assert response.role == MessageRole.ASSISTANT
            assert response.message_type == MessageType.RESPONSE
            assert "4" in response.content
            assert response.metadata is not None
            assert response.metadata.model == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_process_with_context(self, conversation_agent):
        """Test processing with conversation context"""
        message = AgentMessage(
            role=MessageRole.USER,
            content="What did I just ask?",
            context={
                "conversation_history": [
                    {"user": "What is 2 + 2?", "agent": "The answer is 4."}
                ],
            },
        )

        mock_response = MockAnthropicMessage(text="You asked what 2 + 2 is.")

        with patch.object(
            conversation_agent.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await conversation_agent.process(message)

            assert "asked" in response.content.lower()

    @pytest.mark.asyncio
    async def test_metrics_updated_after_processing(self, conversation_agent, sample_message):
        """Test that metrics are updated after processing"""
        mock_response = MockAnthropicMessage()

        with patch.object(
            conversation_agent.client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            metrics_before = conversation_agent.get_metrics()
            await conversation_agent.process(sample_message)
            metrics_after = conversation_agent.get_metrics()

            assert metrics_after["total_messages"] == metrics_before["total_messages"] + 1
            assert metrics_after["total_tokens"] > metrics_before["total_tokens"]


class TestToolCalling:
    """Test tool calling functionality"""

    @pytest.mark.asyncio
    async def test_calculator_tool_execution(self, conversation_agent):
        """Test calculator tool execution"""
        result = await conversation_agent._handle_tool_use(
            tool_name="calculate",
            tool_input={"expression": "2 + 2"},
        )

        assert result["success"] is True
        assert result["result"] == 4
        assert result["expression"] == "2 + 2"

    @pytest.mark.asyncio
    async def test_datetime_tool_execution(self, conversation_agent):
        """Test datetime tool execution"""
        result = await conversation_agent._handle_tool_use(
            tool_name="get_datetime",
            tool_input={"format": "date"},
        )

        assert result["success"] is True
        assert "datetime" in result
        assert result["format"] == "date"

    @pytest.mark.asyncio
    async def test_memory_store_tool_execution(self, conversation_agent):
        """Test memory storage and retrieval"""
        context = {"memory_store": {}}

        # Store a value
        store_result = await conversation_agent._handle_tool_use(
            tool_name="memory_store",
            tool_input={"action": "store", "key": "user_name", "value": "Alice"},
            context=context,
        )

        assert store_result["success"] is True
        assert store_result["key"] == "user_name"

        # Retrieve the value
        retrieve_result = await conversation_agent._handle_tool_use(
            tool_name="memory_store",
            tool_input={"action": "retrieve", "key": "user_name"},
            context=context,
        )

        assert retrieve_result["success"] is True
        assert retrieve_result["value"] == "Alice"

    @pytest.mark.asyncio
    async def test_search_tool_execution(self, conversation_agent):
        """Test search tool with conversation history"""
        context = {
            "conversation_history": [
                {"user": "I love Python programming", "agent": "That's great!"},
                {"user": "I also enjoy machine learning", "agent": "Interesting!"},
            ]
        }

        result = await conversation_agent._handle_tool_use(
            tool_name="search_knowledge",
            tool_input={"query": "Python", "limit": 5},
            context=context,
        )

        assert result["success"] is True
        assert len(result["results"]) > 0
        assert "Python" in result["results"][0]["user"]

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, conversation_agent):
        """Test error handling in tool execution"""
        result = await conversation_agent._handle_tool_use(
            tool_name="calculate",
            tool_input={"expression": "invalid expression !!!"},
        )

        assert result["success"] is False
        assert "error" in result


class TestErrorHandling:
    """Test error handling and resilience"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_handling(self, conversation_agent, sample_message):
        """Test circuit breaker error response"""
        from circuitbreaker import CircuitBreakerError

        with patch.object(
            conversation_agent,
            "_call_claude_api",
            side_effect=CircuitBreakerError("Circuit open"),
        ):
            response = await conversation_agent.process(sample_message)

            assert response.role == MessageRole.ASSISTANT
            assert "high load" in response.content.lower()
            assert response.metadata.error == "circuit_breaker_open"

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, conversation_agent, sample_message):
        """Test timeout error response"""
        with patch.object(
            conversation_agent,
            "_call_claude_api",
            side_effect=TimeoutError("Operation timed out"),
        ):
            response = await conversation_agent.process(sample_message)

            assert response.role == MessageRole.ASSISTANT
            assert "longer than expected" in response.content.lower()
            assert response.metadata.error == "timeout"

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, conversation_agent, sample_message):
        """Test rate limit error response"""
        with patch.object(
            conversation_agent,
            "_call_claude_api",
            side_effect=anthropic.RateLimitError("Rate limit exceeded"),
        ):
            response = await conversation_agent.process(sample_message)

            assert response.role == MessageRole.ASSISTANT
            assert "moment" in response.content.lower()
            assert response.metadata.error == "rate_limit"

    @pytest.mark.asyncio
    async def test_generic_error_handling(self, conversation_agent, sample_message):
        """Test generic error response"""
        with patch.object(
            conversation_agent,
            "_call_claude_api",
            side_effect=Exception("Unexpected error"),
        ):
            response = await conversation_agent.process(sample_message)

            assert response.role == MessageRole.ASSISTANT
            assert "wrong" in response.content.lower()
            assert response.metadata.error is not None


class TestIntelligenceFeatures:
    """Test intelligence features"""

    def test_intent_detection_question(self, conversation_agent):
        """Test intent detection for questions"""
        assert conversation_agent.detect_intent("What is machine learning?") == "question"
        assert conversation_agent.detect_intent("How does this work?") == "question"
        assert conversation_agent.detect_intent("Why is the sky blue?") == "question"

    def test_intent_detection_calculation(self, conversation_agent):
        """Test intent detection for calculations"""
        assert conversation_agent.detect_intent("Calculate 2 plus 2") == "calculation"
        assert conversation_agent.detect_intent("What is 5 times 3?") == "calculation"

    def test_intent_detection_datetime(self, conversation_agent):
        """Test intent detection for datetime queries"""
        assert conversation_agent.detect_intent("What time is it?") == "datetime_query"
        assert conversation_agent.detect_intent("What day is today?") == "datetime_query"

    def test_intent_detection_memory(self, conversation_agent):
        """Test intent detection for memory operations"""
        assert conversation_agent.detect_intent("Remember my name is Alice") == "store_memory"
        assert conversation_agent.detect_intent("I prefer Python") == "store_memory"

    def test_intent_detection_end_conversation(self, conversation_agent):
        """Test intent detection for ending conversation"""
        assert conversation_agent.detect_intent("Goodbye") == "end_conversation"
        assert conversation_agent.detect_intent("Thanks, bye!") == "end_conversation"

    def test_entity_extraction(self, conversation_agent):
        """Test entity extraction from text"""
        entities = conversation_agent.extract_entities(
            "I learned about Python and Machine Learning today. I completed 5 exercises."
        )

        assert "5" in entities["numbers"]
        assert "Python" in entities["topics"]
        assert "Machine" in entities["topics"]


class TestCapabilities:
    """Test agent capabilities reporting"""

    def test_get_capabilities(self, conversation_agent):
        """Test capabilities reporting"""
        capabilities = conversation_agent.get_capabilities()

        assert capabilities["agent_name"] == "conversation_agent_v2"
        assert capabilities["model"] == "claude-3-5-sonnet-20241022"
        assert capabilities["features"]["tool_calling"] is True
        assert capabilities["features"]["context_management"] is True
        assert len(capabilities["tools"]) == 4

    def test_capabilities_without_tools(self):
        """Test capabilities when tools are disabled"""
        agent = ConversationAgent(enable_tools=False)
        capabilities = agent.get_capabilities()

        assert capabilities["features"]["tool_calling"] is False
        assert len(capabilities["tools"]) == 0


class TestContextManagement:
    """Test context window management"""

    def test_format_context_for_claude(self, conversation_agent):
        """Test context formatting"""
        history = [
            {"user": "First message", "agent": "First response"},
            {"user": "Second message", "agent": "Second response"},
        ]

        messages = conversation_agent._format_context_for_claude(
            current_message="Third message",
            conversation_history=history,
        )

        assert len(messages) == 5  # 2 exchanges + current = 5 messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First message"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Third message"

    def test_context_truncation(self, conversation_agent):
        """Test context is truncated to max length"""
        # Create history longer than max_context_length
        long_history = [
            {"user": f"Message {i}", "agent": f"Response {i}"}
            for i in range(20)
        ]

        messages = conversation_agent._format_context_for_claude(
            current_message="Current message",
            conversation_history=long_history,
        )

        # Should only include last max_context_length exchanges + current
        expected_length = (conversation_agent.max_context_length * 2) + 1
        assert len(messages) == expected_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
