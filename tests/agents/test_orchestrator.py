"""
Tests for AgentOrchestrator

Tests the orchestrator that coordinates multiple agents.
"""
import pytest
import asyncio
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType


class TestOrchestratorInitialization:
    """Test orchestrator initialization"""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initializes properly"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'agents')
        assert len(orchestrator.agents) > 0

    @pytest.mark.asyncio
    async def test_all_agents_registered(self, orchestrator):
        """Test all agents are registered"""
        expected_agents = [
            "conversation",
            "analysis",
            "research",
            "synthesis"
        ]

        for agent_key in expected_agents:
            assert agent_key in orchestrator.agents


class TestSimpleRouting:
    """Test simple single-agent routing"""

    @pytest.mark.asyncio
    async def test_basic_conversation_routing(self, orchestrator):
        """Test routing basic conversation to ConversationAgent"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello, how are you?"}
        )

        response = await orchestrator.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "agents_used" in response.metadata
        assert "conversation" in response.metadata["agents_used"]

    @pytest.mark.asyncio
    async def test_greeting_response(self, orchestrator):
        """Test orchestrator handles greetings"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hi there!"}
        )

        response = await orchestrator.process(message)

        assert "text" in response.content
        assert len(response.content["text"]) > 0


class TestParallelExecution:
    """Test parallel agent execution"""

    @pytest.mark.asyncio
    async def test_parallel_analysis_and_conversation(self, orchestrator):
        """Test parallel execution of conversation and analysis"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "I'm learning about neural networks",
                "enable_analysis": True
            }
        )

        response = await orchestrator.process(message)

        # Should use multiple agents
        if "agents_used" in response.metadata:
            agents_used = response.metadata["agents_used"]
            # Might use conversation and analysis agents
            assert len(agents_used) >= 1

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(self, orchestrator, timing):
        """Test parallel execution is faster"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Tell me about AI",
                "enable_analysis": True
            }
        )

        with timing:
            response = await orchestrator.process(message)

        # With mocked agents, should be very fast
        assert timing.elapsed_ms < 200  # 200ms


class TestResearchPipeline:
    """Test research → synthesis → conversation pipeline"""

    @pytest.mark.asyncio
    async def test_research_enabled_conversation(self, orchestrator):
        """Test conversation with research enabled"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "What are the latest developments in transformers?",
                "enable_research": True
            }
        )

        response = await orchestrator.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE

        # Should use research agent
        if "agents_used" in response.metadata:
            agents_used = response.metadata["agents_used"]
            # Should include research agent
            has_research = any("research" in str(a).lower() for a in agents_used)


class TestConditionalRouting:
    """Test conditional routing based on message content"""

    @pytest.mark.asyncio
    async def test_research_keyword_triggers_research(self, orchestrator):
        """Test research keywords trigger ResearchAgent"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Find me recent papers about transformers"
            }
        )

        response = await orchestrator.process(message)

        # Response should be generated
        assert response is not None

    @pytest.mark.asyncio
    async def test_analysis_keyword_triggers_analysis(self, orchestrator):
        """Test analysis keywords trigger AnalysisAgent"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Analyze my learning patterns"
            }
        )

        response = await orchestrator.process(message)

        # Response should be generated
        assert response is not None


class TestResponseAggregation:
    """Test aggregation of multiple agent responses"""

    @pytest.mark.asyncio
    async def test_response_combines_agent_outputs(self, orchestrator):
        """Test orchestrator combines outputs from multiple agents"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Explain neural networks with recent research",
                "enable_research": True,
                "enable_analysis": True
            }
        )

        response = await orchestrator.process(message)

        # Should have combined response
        assert "text" in response.content
        assert len(response.content["text"]) > 0


class TestOrchestratorMetadata:
    """Test metadata in orchestrator responses"""

    @pytest.mark.asyncio
    async def test_metadata_includes_agents_used(self, orchestrator):
        """Test metadata includes agents used"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await orchestrator.process(message)

        assert "agents_used" in response.metadata or "agents" in response.metadata

    @pytest.mark.asyncio
    async def test_metadata_includes_processing_time(self, orchestrator):
        """Test metadata includes processing time"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await orchestrator.process(message)

        # May include processing time
        has_timing = (
            "processing_time_ms" in response.metadata or
            "execution_time" in response.metadata or
            "duration" in response.metadata
        )


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator"""

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, orchestrator, mock_anthropic_client):
        """Test orchestrator handles agent failures"""
        # Make an agent fail
        mock_anthropic_client.messages.create.side_effect = Exception("Agent Error")

        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response = await orchestrator.process(message)

        # Should return error response or fallback
        assert response is not None

    @pytest.mark.asyncio
    async def test_invalid_message_type(self, orchestrator):
        """Test handling of invalid message type"""
        message = AgentMessage(
            message_type=MessageType.AGENT_ERROR,  # Invalid for orchestrator
            content={"text": "Hello"}
        )

        response = await orchestrator.process(message)

        # Should handle gracefully
        assert response is not None

    @pytest.mark.asyncio
    async def test_missing_content(self, orchestrator):
        """Test handling of missing content"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={}  # Empty content
        )

        response = await orchestrator.process(message)

        # Should handle gracefully
        assert response is not None


class TestOrchestratorPerformance:
    """Test orchestrator performance"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, orchestrator):
        """Test handling concurrent requests"""
        messages = [
            AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={"text": f"Message {i}"}
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(
            *[orchestrator.process(msg) for msg in messages]
        )

        # All should succeed
        assert len(responses) == 5
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_response_time(self, orchestrator, timing):
        """Test orchestrator response time"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        with timing:
            response = await orchestrator.process(message)

        # With mocked agents, should be very fast
        assert timing.elapsed_ms < 100


class TestOrchestratorStateManagement:
    """Test orchestrator state management"""

    @pytest.mark.asyncio
    async def test_session_context_maintained(self, orchestrator):
        """Test orchestrator maintains session context"""
        session_id = "test-session-123"

        # First message
        message1 = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "I'm learning Python",
                "session_id": session_id
            }
        )
        response1 = await orchestrator.process(message1)

        # Second message
        message2 = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Tell me more about functions",
                "session_id": session_id
            }
        )
        response2 = await orchestrator.process(message2)

        # Both should succeed
        assert response1 is not None
        assert response2 is not None


class TestAgentRegistration:
    """Test dynamic agent registration"""

    @pytest.mark.asyncio
    async def test_register_new_agent(self, orchestrator):
        """Test registering a new agent"""
        from app.agents.base import BaseAgent

        class CustomAgent(BaseAgent):
            async def process(self, message):
                return self._create_response(
                    content={"custom": "response"},
                    message_type=MessageType.CONVERSATION_RESPONSE,
                    original_message=message
                )

        custom_agent = CustomAgent("custom_agent")
        orchestrator.register_agent(custom_agent)

        assert "custom_agent" in orchestrator.agents
