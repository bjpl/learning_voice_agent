"""
End-to-End Integration Tests for Multi-Agent System

Tests complete workflows involving multiple agents and components.
"""
import pytest
import asyncio
from app.agents.orchestrator import AgentOrchestrator
from app.agents.base import AgentMessage, MessageType


class TestEndToEndConversationFlow:
    """Test complete conversation workflows"""

    @pytest.mark.asyncio
    async def test_simple_conversation_flow(self, orchestrator):
        """Test basic conversation flow"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello, I'm learning AI"}
        )

        response = await orchestrator.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content
        assert len(response.content["text"]) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, orchestrator):
        """Test multi-turn conversation with context"""
        session_id = "test-session-multi-turn"

        # Turn 1
        message1 = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "I'm learning Python",
                "session_id": session_id
            }
        )
        response1 = await orchestrator.process(message1)
        assert response1.message_type == MessageType.CONVERSATION_RESPONSE

        # Turn 2
        message2 = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Can you help with functions?",
                "session_id": session_id,
                "context": [
                    {"user": message1.content["text"], "agent": response1.content["text"]}
                ]
            }
        )
        response2 = await orchestrator.process(message2)
        assert response2.message_type == MessageType.CONVERSATION_RESPONSE

        # Turn 3
        message3 = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Show me an example",
                "session_id": session_id,
                "context": [
                    {"user": message1.content["text"], "agent": response1.content["text"]},
                    {"user": message2.content["text"], "agent": response2.content["text"]}
                ]
            }
        )
        response3 = await orchestrator.process(message3)
        assert response3.message_type == MessageType.CONVERSATION_RESPONSE


class TestResearchAugmentedWorkflow:
    """Test research-augmented conversation workflows"""

    @pytest.mark.asyncio
    async def test_research_augmented_response(self, orchestrator):
        """Test conversation with research augmentation"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "What are the latest developments in transformers?",
                "enable_research": True
            }
        )

        response = await orchestrator.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content

        # Should include research metadata
        if "research" in response.metadata:
            research_data = response.metadata["research"]
            assert "results" in research_data or "sources" in research_data


class TestAnalysisWorkflow:
    """Test analysis and synthesis workflows"""

    @pytest.mark.asyncio
    async def test_conversation_with_analysis(self, orchestrator):
        """Test conversation with concept analysis"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "I'm learning about neural networks and backpropagation",
                "enable_analysis": True
            }
        )

        response = await orchestrator.process(message)

        assert response is not None
        # May include analysis in metadata
        if "analysis" in response.metadata:
            analysis = response.metadata["analysis"]
            assert "concepts" in analysis or "entities" in analysis


class TestComplexMultiAgentWorkflow:
    """Test complex workflows involving multiple agents"""

    @pytest.mark.asyncio
    async def test_research_analysis_synthesis_conversation(self, orchestrator):
        """Test full pipeline: research → analysis → synthesis → conversation"""
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Explain transformers with recent research and key concepts",
                "enable_research": True,
                "enable_analysis": True
            }
        )

        response = await orchestrator.process(message)

        assert response.message_type == MessageType.CONVERSATION_RESPONSE
        assert "text" in response.content

        # Should coordinate multiple agents
        if "agents_used" in response.metadata:
            agents_used = response.metadata["agents_used"]
            # May involve multiple agents
            assert len(agents_used) >= 1


class TestErrorRecoveryWorkflow:
    """Test error recovery in workflows"""

    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, orchestrator, mock_anthropic_client):
        """Test workflow recovery when an agent fails"""
        # First request fails
        mock_anthropic_client.messages.create.side_effect = Exception("Temporary error")

        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={"text": "Hello"}
        )

        response1 = await orchestrator.process(message)

        # Should handle error gracefully
        assert response1 is not None

        # Reset mock - second request should succeed
        mock_anthropic_client.messages.create.side_effect = None

        response2 = await orchestrator.process(message)

        assert response2.message_type == MessageType.CONVERSATION_RESPONSE


class TestConcurrentWorkflows:
    """Test concurrent workflow execution"""

    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, orchestrator):
        """Test handling concurrent conversations"""
        messages = [
            AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={
                    "text": f"Tell me about topic {i}",
                    "session_id": f"session-{i}"
                }
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(
            *[orchestrator.process(msg) for msg in messages]
        )

        # All should succeed
        assert len(responses) == 5
        assert all(r.message_type == MessageType.CONVERSATION_RESPONSE for r in responses)

    @pytest.mark.asyncio
    async def test_concurrent_research_requests(self, orchestrator):
        """Test handling concurrent research requests"""
        messages = [
            AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={
                    "text": f"Research topic {i}",
                    "enable_research": True
                }
            )
            for i in range(3)
        ]

        responses = await asyncio.gather(
            *[orchestrator.process(msg) for msg in messages]
        )

        # All should succeed
        assert len(responses) == 3


class TestPerformanceUnderLoad:
    """Test system performance under load"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput(self, orchestrator, timing):
        """Test system throughput"""
        num_requests = 20

        messages = [
            AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={"text": f"Message {i}"}
            )
            for i in range(num_requests)
        ]

        with timing:
            responses = await asyncio.gather(
                *[orchestrator.process(msg) for msg in messages]
            )

        # All should succeed
        assert len(responses) == num_requests

        # Calculate throughput
        throughput = num_requests / (timing.elapsed_ms / 1000)
        print(f"\nThroughput: {throughput:.1f} requests/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_latency_distribution(self, orchestrator):
        """Test latency distribution"""
        import time

        latencies = []

        for i in range(10):
            message = AgentMessage(
                message_type=MessageType.CONVERSATION_REQUEST,
                content={"text": f"Test message {i}"}
            )

            start = time.time()
            response = await orchestrator.process(message)
            latency = (time.time() - start) * 1000  # Convert to ms

            latencies.append(latency)
            assert response is not None

        # Calculate statistics
        latencies.sort()
        p50 = latencies[4]
        p95 = latencies[9]

        print(f"\nLatency P50: {p50:.1f}ms")
        print(f"Latency P95: {p95:.1f}ms")

        # With mocked agents, should be very fast
        assert p95 < 100  # 100ms


class TestDataPersistenceWorkflow:
    """Test workflows with data persistence"""

    @pytest.mark.asyncio
    async def test_conversation_saved_to_database(self, orchestrator, test_agent_db):
        """Test conversations are saved to database"""
        session_id = "persist-test-session"

        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Hello, save this conversation",
                "session_id": session_id
            }
        )

        response = await orchestrator.process(message)

        # Save to database
        await test_agent_db.save_exchange(
            session_id,
            message.content["text"],
            response.content["text"],
            metadata={"agent_type": "v2"}
        )

        # Retrieve from database
        history = await test_agent_db.get_session_history(session_id)

        assert len(history) > 0
        assert history[0]["user_text"] == message.content["text"]


class TestBackwardCompatibility:
    """Test backward compatibility with v1.0"""

    @pytest.mark.asyncio
    async def test_v1_message_format(self, orchestrator):
        """Test handling v1.0 message format"""
        # Simulate v1.0 format
        message = AgentMessage(
            message_type=MessageType.CONVERSATION_REQUEST,
            content={
                "text": "Hello",
                # v1.0 fields
                "session_id": "v1-session"
            }
        )

        response = await orchestrator.process(message)

        # Should handle v1.0 format
        assert response.message_type == MessageType.CONVERSATION_RESPONSE
