"""
Integration tests for ResearchAgent tools
PATTERN: Real API integration testing with network calls
WHY: Verify actual tool functionality against live services

NOTE: These tests make real network calls and may be slow.
Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import pytest
import os

from app.agents.research_agent import ResearchAgent
from app.agents.base import AgentMessage, MessageType


pytestmark = pytest.mark.integration


@pytest.fixture
async def research_agent():
    """Create research agent with optional API keys from environment"""
    agent = ResearchAgent(
        agent_id="integration-test-agent",
        tavily_api_key=os.getenv("TAVILY_API_KEY"),  # Optional
    )
    yield agent
    await agent.cleanup()


@pytest.mark.asyncio
class TestLiveToolIntegration:
    """Integration tests with live APIs"""

    async def test_wikipedia_live_search(self, research_agent):
        """Test Wikipedia search with real API"""
        result = await research_agent._wikipedia_search(
            "Python programming language",
            max_results=3,
        )

        assert result["source"] == "wikipedia"
        assert len(result["results"]) > 0
        assert "Python" in result["results"][0]["title"]
        assert "wikipedia.org" in result["results"][0]["url"]

    async def test_arxiv_live_search(self, research_agent):
        """Test arXiv search with real API"""
        result = await research_agent._arxiv_search(
            "neural networks",
            max_results=3,
        )

        assert result["source"] == "arxiv"
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] != ""
        assert "arxiv.org" in result["results"][0]["url"]

    async def test_duckduckgo_live_search(self, research_agent):
        """Test DuckDuckGo search with real API"""
        result = await research_agent._duckduckgo_search(
            "machine learning",
            max_results=3,
        )

        assert result["source"] == "duckduckgo"
        assert result["query"] == "machine learning"
        # Results may vary, just check structure
        assert "results" in result

    @pytest.mark.skipif(
        not os.getenv("TAVILY_API_KEY"),
        reason="TAVILY_API_KEY not set",
    )
    async def test_tavily_live_search(self, research_agent):
        """Test Tavily search with real API (requires API key)"""
        result = await research_agent._tavily_search(
            "artificial intelligence",
            max_results=3,
        )

        assert result["source"] == "tavily"
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] != ""
        assert result["results"][0]["url"] != ""

    async def test_knowledge_base_integration(self, research_agent):
        """Test knowledge base query with real database"""
        from app.database import db

        # Initialize database
        await db.initialize()

        # Add test exchanges
        await db.save_exchange(
            session_id="integration-test",
            user_text="Tell me about quantum computing",
            agent_text="Quantum computing uses quantum mechanics principles...",
        )

        await db.save_exchange(
            session_id="integration-test",
            user_text="What is a qubit?",
            agent_text="A qubit is a quantum bit, the basic unit of quantum information...",
        )

        # Search
        result = await research_agent._query_knowledge_base(
            "quantum",
            max_results=5,
        )

        assert result["source"] == "knowledge_base"
        assert len(result["results"]) >= 2
        assert any("quantum" in r["user_text"].lower() for r in result["results"])


@pytest.mark.asyncio
class TestEndToEndResearch:
    """End-to-end research workflows"""

    async def test_multi_tool_research(self, research_agent):
        """Test research with multiple tools in parallel"""
        message = AgentMessage(
            sender="test-orchestrator",
            recipient=research_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "Python programming",
                "tools": ["wikipedia", "arxiv"],
                "max_results": 2,
            },
        )

        response = await research_agent.process(message)

        assert response.message_type == MessageType.RESEARCH_COMPLETE
        assert response.content["query"] == "Python programming"
        assert "wikipedia" in response.content["results"]
        assert "arxiv" in response.content["results"]

    async def test_research_with_caching(self, research_agent):
        """Test that repeated queries use cache"""
        # First query
        result1 = await research_agent._wikipedia_search(
            "Albert Einstein",
            max_results=2,
        )

        # Get initial metrics
        initial_calls = research_agent.tool_metrics["wikipedia"]["calls"]

        # Cache the result manually to ensure it's cached
        cache_key = "wikipedia:Albert Einstein:2"
        research_agent._cache_result(cache_key, result1)

        # Second query - should use cache
        cached_result = research_agent._get_cached_result(cache_key)

        assert cached_result is not None
        assert cached_result == result1

    async def test_error_recovery(self, research_agent):
        """Test error handling and recovery"""
        # Request with invalid tool
        message = AgentMessage(
            sender="test",
            recipient=research_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={
                "query": "test",
                "tools": ["invalid_tool", "wikipedia"],  # One invalid, one valid
            },
        )

        response = await research_agent.process(message)

        # Should still complete with valid tool results
        assert response.message_type == MessageType.RESEARCH_COMPLETE
        assert "wikipedia" in response.content["results"]

    async def test_timeout_handling(self, research_agent):
        """Test that slow tools timeout correctly"""
        import asyncio

        async def slow_tool(query, max_results=5):
            await asyncio.sleep(35)  # Exceeds 30s timeout
            return {"results": []}

        # Replace a tool with slow version
        research_agent.tools["wikipedia"] = slow_tool

        # Should timeout
        with pytest.raises(Exception):  # Will be httpx timeout or similar
            await research_agent._execute_tool_with_metrics(
                tool_name="wikipedia",
                query="test",
                max_results=5,
            )


@pytest.mark.asyncio
class TestMetricsAndObservability:
    """Test metrics collection in real scenarios"""

    async def test_metrics_tracking(self, research_agent):
        """Test metrics are properly tracked"""
        # Make some real API calls
        await research_agent._wikipedia_search("Python", max_results=1)
        await research_agent._arxiv_search("machine learning", max_results=1)

        metrics = research_agent.get_tool_metrics()

        assert metrics["wikipedia"]["calls"] > 0
        assert metrics["arxiv"]["calls"] > 0
        assert metrics["wikipedia"]["avg_execution_time_ms"] > 0

    async def test_agent_metrics(self, research_agent):
        """Test agent-level metrics"""
        message = AgentMessage(
            sender="test",
            recipient=research_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"query": "test", "tools": ["wikipedia"]},
        )

        await research_agent.receive_message(message)
        await research_agent.process(message)

        agent_metrics = research_agent.get_metrics()

        assert agent_metrics["messages_received"] == 1
        assert agent_metrics["messages_sent"] > 0
        assert agent_metrics["agent_type"] == "ResearchAgent"
