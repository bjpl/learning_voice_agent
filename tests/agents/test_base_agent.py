"""
Tests for BaseAgent

Tests the base agent class that all agents inherit from.
"""
import pytest
from unittest.mock import AsyncMock
from app.agents.base import BaseAgent, AgentMessage, MessageType
import uuid


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""

    def __init__(self, agent_id: str = None, config: dict = None):
        """Initialize with config support for backward compat"""
        super().__init__(agent_id=agent_id, agent_type="ConcreteAgent")
        self.config = config or {}

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Simple echo implementation"""
        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            content={"echo": message.content},
            correlation_id=message.message_id
        )


@pytest.fixture
def concrete_agent():
    """Instance of ConcreteAgent for testing"""
    return ConcreteAgent(agent_id="test_agent", config={"test_key": "test_value"})


class TestBaseAgentInitialization:
    """Test agent initialization"""

    def test_agent_id_set(self, concrete_agent):
        """Test agent_id is set correctly"""
        assert concrete_agent.agent_id == "test_agent"

    def test_config_set(self, concrete_agent):
        """Test config is set correctly"""
        assert concrete_agent.config == {"test_key": "test_value"}

    def test_config_defaults_to_empty_dict(self):
        """Test config defaults to empty dict if not provided"""
        agent = ConcreteAgent("test")
        assert agent.config == {}

    def test_state_initialized(self, concrete_agent):
        """Test state is initialized as empty dict"""
        assert concrete_agent.state == {}


class TestBaseAgentLifecycle:
    """Test agent lifecycle methods"""

    @pytest.mark.asyncio
    async def test_initialize(self, concrete_agent):
        """Test initialize method"""
        await concrete_agent.initialize()
        # BaseAgent.initialize() is a no-op, just verify it doesn't raise

    @pytest.mark.asyncio
    async def test_cleanup(self, concrete_agent):
        """Test cleanup method"""
        await concrete_agent.cleanup()
        # BaseAgent.cleanup() is a no-op, just verify it doesn't raise


class TestBaseAgentMessageProcessing:
    """Test agent message processing"""

    @pytest.mark.asyncio
    async def test_process_returns_agent_message(self, concrete_agent, sample_agent_message):
        """Test process returns AgentMessage"""
        response = await concrete_agent.process(sample_agent_message)

        assert isinstance(response, AgentMessage)

    @pytest.mark.asyncio
    async def test_process_basic_functionality(self, concrete_agent):
        """Test basic message processing"""
        message = AgentMessage(
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await concrete_agent.process(message)

        assert response.content["echo"] == {"text": "Hello"}


class TestCreateResponse:
    """Test _create_response helper method"""

    def test_create_response_sets_sender(self, concrete_agent):
        """Test sender is set to agent_id"""
        original = AgentMessage(
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}
        )

        response = concrete_agent._create_response(
            content={"result": "success"},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=original
        )

        assert response.sender == "test_agent"

    def test_create_response_sets_recipient(self, concrete_agent):
        """Test recipient is set to original sender"""
        original = AgentMessage(
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}
        )

        response = concrete_agent._create_response(
            content={"result": "success"},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=original
        )

        assert response.recipient == "user"

    def test_create_response_sets_correlation_id(self, concrete_agent):
        """Test correlation_id links to original message"""
        original = AgentMessage(
            message_id="original-123",
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}
        )

        response = concrete_agent._create_response(
            content={"result": "success"},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=original
        )

        assert response.correlation_id == "original-123"

    def test_create_response_sets_content(self, concrete_agent):
        """Test content is set correctly"""
        original = AgentMessage(
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}
        )

        content = {"result": "success", "data": [1, 2, 3]}

        response = concrete_agent._create_response(
            content=content,
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=original
        )

        assert response.content == content

    def test_create_response_sets_message_type(self, concrete_agent):
        """Test message_type is set correctly"""
        original = AgentMessage(
            sender="user",
            recipient="test_agent",
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}
        )

        response = concrete_agent._create_response(
            content={},
            message_type=MessageType.CONVERSATION_RESPONSE,
            original_message=original
        )

        assert response.message_type == MessageType.CONVERSATION_RESPONSE


class TestAgentState:
    """Test agent state management"""

    def test_state_can_be_modified(self, concrete_agent):
        """Test agent state can be modified"""
        concrete_agent.state["key"] = "value"
        assert concrete_agent.state["key"] == "value"

    def test_state_persists_across_calls(self, concrete_agent):
        """Test state persists"""
        concrete_agent.state["counter"] = 0

        concrete_agent.state["counter"] += 1
        assert concrete_agent.state["counter"] == 1

        concrete_agent.state["counter"] += 1
        assert concrete_agent.state["counter"] == 2


class TestAgentConfig:
    """Test agent configuration"""

    def test_config_can_be_accessed(self, concrete_agent):
        """Test config is accessible"""
        assert concrete_agent.config["test_key"] == "test_value"

    def test_config_is_readonly_pattern(self, concrete_agent):
        """Test config follows read-only pattern"""
        # Config can be modified but shouldn't be in practice
        # This is a convention test
        original_config = concrete_agent.config.copy()
        concrete_agent.config["new_key"] = "new_value"

        assert concrete_agent.config["new_key"] == "new_value"
        assert "new_key" not in original_config
