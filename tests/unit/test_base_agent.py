"""
Unit tests for BaseAgent
PATTERN: Comprehensive testing of agent foundation
WHY: Ensure reliable base for all agent implementations
"""

import pytest
import asyncio
from datetime import datetime

from app.agents.base import BaseAgent, AgentMessage, MessageType


class TestAgent(BaseAgent):
    """Simple test agent for testing BaseAgent functionality"""

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Echo the message content back"""
        return await self.send_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            content={"echo": message.content},
            correlation_id=message.message_id,
        )


@pytest.fixture
def test_agent():
    """Create a test agent instance"""
    return TestAgent(agent_id="test-agent-1")


@pytest.mark.asyncio
class TestAgentMessage:
    """Test AgentMessage dataclass"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = AgentMessage(
            sender="agent-1",
            recipient="agent-2",
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )

        assert msg.sender == "agent-1"
        assert msg.recipient == "agent-2"
        assert msg.message_type == MessageType.REQUEST
        assert msg.content == {"query": "test"}
        assert msg.message_id is not None

    def test_message_serialization(self):
        """Test message to_dict and from_dict"""
        original = AgentMessage(
            sender="agent-1",
            recipient="agent-2",
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["sender"] == "agent-1"
        assert data["message_type"] == "request"

        # Deserialize
        restored = AgentMessage.from_dict(data)
        assert restored.sender == original.sender
        assert restored.recipient == original.recipient
        assert restored.message_type == original.message_type
        assert restored.content == original.content

    def test_correlation_id(self):
        """Test correlation ID for message tracking"""
        msg1 = AgentMessage(sender="a", recipient="b", message_type=MessageType.REQUEST)
        msg2 = AgentMessage(
            sender="b",
            recipient="a",
            message_type=MessageType.RESPONSE,
            correlation_id=msg1.message_id,
        )

        assert msg2.correlation_id == msg1.message_id


@pytest.mark.asyncio
class TestBaseAgent:
    """Test BaseAgent functionality"""

    async def test_agent_initialization(self, test_agent):
        """Test agent initialization"""
        assert test_agent.agent_id == "test-agent-1"
        assert test_agent.agent_type == "TestAgent"
        assert not test_agent.is_running
        assert test_agent.inbox.qsize() == 0
        assert test_agent.outbox.qsize() == 0

    async def test_send_message(self, test_agent):
        """Test sending a message"""
        message = await test_agent.send_message(
            recipient="agent-2",
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )

        assert message.sender == test_agent.agent_id
        assert message.recipient == "agent-2"
        assert message.message_type == MessageType.REQUEST
        assert test_agent.outbox.qsize() == 1

    async def test_receive_message(self, test_agent):
        """Test receiving a message"""
        message = AgentMessage(
            sender="agent-2",
            recipient=test_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )

        await test_agent.receive_message(message)

        assert test_agent.inbox.qsize() == 1
        assert test_agent.metrics["messages_received"] == 1

    async def test_process_message(self, test_agent):
        """Test processing a message"""
        message = AgentMessage(
            sender="agent-2",
            recipient=test_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )

        response = await test_agent.process(message)

        assert response.recipient == "agent-2"
        assert response.message_type == MessageType.RESPONSE
        assert response.content["echo"] == {"query": "test"}

    async def test_agent_run_loop(self, test_agent):
        """Test agent run loop"""
        # Send a message to the agent
        message = AgentMessage(
            sender="agent-2",
            recipient=test_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"query": "test"},
        )
        await test_agent.receive_message(message)

        # Start agent in background
        run_task = asyncio.create_task(test_agent.run())

        # Wait a bit for processing
        await asyncio.sleep(0.1)

        # Stop agent
        await test_agent.stop()
        await run_task

        # Check that message was processed
        assert test_agent.outbox.qsize() == 1
        response = await test_agent.outbox.get()
        assert response.content["echo"] == {"query": "test"}

    async def test_metrics_tracking(self, test_agent):
        """Test metrics collection"""
        # Send and receive messages
        await test_agent.send_message(
            recipient="agent-2",
            message_type=MessageType.REQUEST,
            content={"test": 1},
        )

        message = AgentMessage(
            sender="agent-2",
            recipient=test_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"test": 2},
        )
        await test_agent.receive_message(message)
        await test_agent.process(message)

        metrics = test_agent.get_metrics()

        assert metrics["agent_id"] == "test-agent-1"
        assert metrics["messages_sent"] == 2  # One explicit + one from process
        assert metrics["messages_received"] == 1
        assert metrics["errors"] == 0

    async def test_error_handling(self, test_agent):
        """Test error handling in message processing"""

        class ErrorAgent(BaseAgent):
            async def process(self, message: AgentMessage) -> AgentMessage:
                raise ValueError("Test error")

        error_agent = ErrorAgent(agent_id="error-agent")

        message = AgentMessage(
            sender="test",
            recipient=error_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"test": 1},
        )

        await error_agent.receive_message(message)

        # Run agent
        run_task = asyncio.create_task(error_agent.run())
        await asyncio.sleep(0.1)
        await error_agent.stop()
        await run_task

        # Should have error response in outbox
        assert error_agent.outbox.qsize() == 1
        error_response = await error_agent.outbox.get()
        assert error_response.message_type == MessageType.ERROR
        assert "Test error" in error_response.content["error"]

    async def test_graceful_shutdown(self, test_agent):
        """Test graceful shutdown"""
        # Start agent
        run_task = asyncio.create_task(test_agent.run())
        await asyncio.sleep(0.05)

        assert test_agent.is_running

        # Stop agent
        await test_agent.stop()
        await run_task

        assert not test_agent.is_running
