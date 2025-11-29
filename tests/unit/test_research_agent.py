"""
Unit tests for ResearchAgent
PATTERN: Comprehensive tool testing with mocking
WHY: Ensure reliable research capabilities
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.agents.research_agent import ResearchAgent, ToolExecutionError
from app.agents.base import AgentMessage, MessageType


# Fixtures are provided by tests/unit/conftest.py


@pytest.mark.asyncio
class TestResearchAgent:
    """Test ResearchAgent functionality"""

    async def test_agent_initialization(self, research_agent):
        """Test research agent initialization"""
        assert research_agent.agent_id == "research-agent-1"
        assert research_agent.agent_type == "ResearchAgent"
        assert len(research_agent.tools) == 5
        assert "web_search" in research_agent.tools
        assert "wikipedia" in research_agent.tools
        assert "arxiv" in research_agent.tools
        assert "knowledge_base" in research_agent.tools

    async def test_process_research_request(self, research_agent):
        """Test processing a research request"""
        # Mock the tool execution
        research_agent._execute_tools_parallel = AsyncMock(
            return_value={
                "web_search": {"results": [{"title": "Test", "url": "http://test.com"}]},
            }
        )

        message = AgentMessage(
            sender="test-sender",
            recipient=research_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={"query": "test query", "tools": ["web_search"]},
        )

        response = await research_agent.process(message)

        assert response.message_type == MessageType.RESEARCH_RESPONSE
        assert response.content["query"] == "test query"
        assert "results" in response.content
        # Results is a flat list with source field indicating tool
        assert len(response.content["results"]) > 0
        assert response.content["results"][0]["source"] == "web"

    async def test_missing_query_error(self, research_agent):
        """Test error when query is missing"""
        message = AgentMessage(
            sender="test-sender",
            recipient=research_agent.agent_id,
            message_type=MessageType.REQUEST,
            content={},  # No query
        )

        response = await research_agent.process(message)

        assert response.message_type == MessageType.AGENT_ERROR
        assert "No query provided" in response.content["error"]

    async def test_parallel_tool_execution(self, research_agent):
        """Test parallel execution of multiple tools"""
        # Mock individual tools
        research_agent._execute_tool_with_metrics = AsyncMock(
            side_effect=lambda tool_name, query, max_results: {
                "source": tool_name,
                "results": [{"data": f"{tool_name}_result"}],
            }
        )

        results = await research_agent._execute_tools_parallel(
            query="test",
            tools=["web_search", "wikipedia"],
            max_results=5,
        )

        assert len(results) == 2
        assert "web_search" in results
        assert "wikipedia" in results

    async def test_tool_error_isolation(self, research_agent):
        """Test that one tool error doesn't affect others"""

        async def mock_tool(tool_name, query, max_results):
            if tool_name == "web_search":
                raise ValueError("Web search failed")
            return {"source": tool_name, "results": []}

        research_agent._execute_tool_with_metrics = AsyncMock(side_effect=mock_tool)

        results = await research_agent._execute_tools_parallel(
            query="test",
            tools=["web_search", "wikipedia"],
            max_results=5,
        )

        assert "web_search" in results
        assert "error" in results["web_search"]
        assert "wikipedia" in results
        assert "results" in results["wikipedia"]


@pytest.mark.asyncio
class TestResearchTools:
    """Test individual research tools"""

    async def test_duckduckgo_search(self, research_agent):
        """Test DuckDuckGo search"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Test abstract",
            "AbstractURL": "http://test.com",
            "Heading": "Test Heading",
            "RelatedTopics": [
                {
                    "Text": "Related topic 1",
                    "FirstURL": "http://related1.com",
                },
            ],
        }

        research_agent.http_client.get = AsyncMock(return_value=mock_response)

        result = await research_agent._duckduckgo_search("test query", max_results=5)

        assert result["source"] == "duckduckgo"
        assert result["query"] == "test query"
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] == "Test Heading"

    async def test_wikipedia_search(self, research_agent):
        """Test Wikipedia search"""
        # Mock search response
        search_response = MagicMock()
        search_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "pageid": 12345,
                        "title": "Test Article",
                        "snippet": "Test snippet",
                    }
                ]
            }
        }

        # Mock extract response
        extract_response = MagicMock()
        extract_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "extract": "Full article extract text",
                    }
                }
            }
        }

        research_agent.http_client.get = AsyncMock(
            side_effect=[search_response, extract_response]
        )

        result = await research_agent._wikipedia_search("test query", max_results=5)

        assert result["source"] == "wikipedia"
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] == "Test Article"
        assert "wikipedia.org" in result["results"][0]["url"]

    async def test_arxiv_search(self, research_agent):
        """Test arXiv search"""
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678</id>
                <title>Test Paper Title</title>
                <summary>Test paper summary</summary>
                <published>2023-01-01T00:00:00Z</published>
                <author><name>Author One</name></author>
                <author><name>Author Two</name></author>
            </entry>
        </feed>
        """

        mock_response = MagicMock()
        mock_response.text = mock_xml

        research_agent.http_client.get = AsyncMock(return_value=mock_response)

        result = await research_agent._arxiv_search("machine learning", max_results=5)

        assert result["source"] == "arxiv"
        assert len(result["results"]) > 0
        assert result["results"][0]["title"] == "Test Paper Title"
        assert len(result["results"][0]["authors"]) == 2

    async def test_knowledge_base_query(self, research_agent):
        """Test knowledge base query"""
        from app.database import db

        # Initialize database
        await db.initialize()

        # Add test data
        await db.save_exchange(
            session_id="test-session",
            user_text="What is machine learning?",
            agent_text="Machine learning is a subset of AI...",
        )

        result = await research_agent._query_knowledge_base(
            "machine learning",
            max_results=5,
        )

        assert result["source"] == "knowledge_base"
        assert len(result["results"]) > 0

    async def test_code_execution_disabled(self, research_agent):
        """Test code execution when disabled"""
        with pytest.raises(ToolExecutionError, match="Code execution is disabled"):
            await research_agent._execute_code("print('hello')")


@pytest.mark.asyncio
class TestCachingAndRateLimiting:
    """Test caching and rate limiting"""

    async def test_result_caching(self, research_agent):
        """Test result caching"""
        # First call
        result1 = {"results": ["test"]}
        research_agent._cache_result("test_key", result1)

        # Second call - should hit cache
        cached = research_agent._get_cached_result("test_key")
        assert cached == result1

    async def test_cache_expiration(self, research_agent):
        """Test cache expiration"""
        from datetime import timedelta

        # Set a very short TTL for testing
        research_agent.cache_ttl = timedelta(seconds=0)

        result = {"results": ["test"]}
        research_agent._cache_result("test_key", result)

        # Wait a moment
        await asyncio.sleep(0.1)

        # Should be expired
        cached = research_agent._get_cached_result("test_key")
        assert cached is None

    async def test_rate_limiting(self, research_agent):
        """Test rate limiting"""
        # Set low rate limit for testing
        research_agent.rate_limit_max_calls = 3

        # Make calls up to limit
        for i in range(3):
            assert research_agent._check_rate_limit("test_tool")

        # Next call should be rate limited
        assert not research_agent._check_rate_limit("test_tool")

    async def test_cache_size_management(self, research_agent):
        """Test cache size management"""
        # Fill cache beyond limit
        for i in range(150):
            research_agent._cache_result(f"key_{i}", {"data": i})

        # Cache should be pruned to ~100 entries
        assert len(research_agent.cache) <= 100


@pytest.mark.asyncio
class TestMetrics:
    """Test metrics collection"""

    async def test_tool_metrics(self, research_agent):
        """Test tool metrics tracking"""
        # Mock successful tool execution
        research_agent.tools["web_search"] = AsyncMock(
            return_value={"results": []}
        )

        await research_agent._execute_tool_with_metrics(
            tool_name="web_search",
            query="test",
            max_results=5,
        )

        metrics = research_agent.get_tool_metrics()

        assert "web_search" in metrics
        assert metrics["web_search"]["calls"] == 1
        assert metrics["web_search"]["errors"] == 0

    async def test_error_metrics(self, research_agent):
        """Test error tracking in metrics"""
        # Mock failing tool
        research_agent.tools["web_search"] = AsyncMock(
            side_effect=ValueError("Test error")
        )

        with pytest.raises(ToolExecutionError):
            await research_agent._execute_tool_with_metrics(
                tool_name="web_search",
                query="test",
                max_results=5,
            )

        metrics = research_agent.get_tool_metrics()

        assert metrics["web_search"]["errors"] == 1
        assert metrics["web_search"]["error_rate"] == 1.0
