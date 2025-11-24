"""
Tests for ResearchAgent

Tests the research agent that retrieves external knowledge.
"""
import pytest
from app.agents.research_agent import ResearchAgent
from app.agents.base import AgentMessage, MessageType


class TestResearchAgentInitialization:
    """Test ResearchAgent initialization"""

    def test_agent_id(self):
        """Test agent_id is set correctly"""
        agent = ResearchAgent()
        assert agent.agent_id == "research_agent"

    @pytest.mark.asyncio
    async def test_initialization(self, research_agent):
        """Test agent initializes properly"""
        # research_agent fixture already initializes
        assert research_agent is not None


class TestWebSearch:
    """Test web search functionality"""

    @pytest.mark.asyncio
    async def test_basic_web_search(self, research_agent):
        """Test basic web search"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "machine learning basics",
                "sources": ["web"],
                "max_results": 5
            }
        )

        response = await research_agent.process(message)

        assert response.message_type == MessageType.RESEARCH_RESPONSE
        assert "results" in response.content
        assert len(response.content["results"]) > 0
        assert len(response.content["results"]) <= 5

    @pytest.mark.asyncio
    async def test_web_search_result_structure(self, research_agent):
        """Test web search result structure"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "neural networks",
                "sources": ["web"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        assert len(results) > 0

        # Check result structure
        result = results[0]
        assert "title" in result
        assert "url" in result
        assert "snippet" in result
        assert "source" in result
        assert result["source"] == "web"

    @pytest.mark.asyncio
    async def test_web_search_max_results(self, research_agent):
        """Test max_results parameter"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "AI",
                "sources": ["web"],
                "max_results": 2
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        assert len(results) <= 2


class TestArxivSearch:
    """Test ArXiv paper search"""

    @pytest.mark.asyncio
    async def test_basic_arxiv_search(self, research_agent):
        """Test basic ArXiv search"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "transformer architecture",
                "sources": ["arxiv"],
                "max_results": 3
            }
        )

        response = await research_agent.process(message)

        assert response.message_type == MessageType.RESEARCH_RESPONSE
        assert "results" in response.content
        results = response.content["results"]
        assert len(results) > 0
        assert all(r["source"] == "arxiv" for r in results)

    @pytest.mark.asyncio
    async def test_arxiv_result_structure(self, research_agent):
        """Test ArXiv result structure"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "attention mechanisms",
                "sources": ["arxiv"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        result = results[0]

        # ArXiv results should have paper metadata
        assert "title" in result
        assert "url" in result
        assert "snippet" in result
        assert "authors" in result
        assert isinstance(result["authors"], list)

    @pytest.mark.asyncio
    async def test_arxiv_authors(self, research_agent):
        """Test ArXiv results include authors"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "deep learning",
                "sources": ["arxiv"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        # At least one result should have authors
        assert any("authors" in r and len(r["authors"]) > 0 for r in results)


class TestWikipediaSearch:
    """Test Wikipedia lookup"""

    @pytest.mark.asyncio
    async def test_wikipedia_lookup(self, research_agent):
        """Test Wikipedia article lookup"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "artificial intelligence",
                "sources": ["wikipedia"]
            }
        )

        response = await research_agent.process(message)

        assert "results" in response.content
        results = response.content["results"]
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_wikipedia_result_structure(self, research_agent):
        """Test Wikipedia result structure"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "machine learning",
                "sources": ["wikipedia"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        result = results[0]

        assert "title" in result
        assert "url" in result
        assert "snippet" in result
        assert result["source"] == "wikipedia"


class TestMultiSourceSearch:
    """Test searching multiple sources"""

    @pytest.mark.asyncio
    async def test_multi_source_search(self, research_agent):
        """Test searching multiple sources simultaneously"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "neural networks",
                "sources": ["web", "arxiv"],
                "max_results": 5
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        sources = {r["source"] for r in results}

        # Should have results from multiple sources
        assert len(sources) > 1
        assert "web" in sources or "arxiv" in sources

    @pytest.mark.asyncio
    async def test_all_sources(self, research_agent):
        """Test searching all available sources"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "deep learning",
                "sources": ["web", "arxiv", "wikipedia"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        assert len(results) > 0

        # Should have results (not necessarily from all sources if mocked)

    @pytest.mark.asyncio
    async def test_sources_metadata(self, research_agent):
        """Test sources_used metadata"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "AI",
                "sources": ["web", "arxiv"]
            }
        )

        response = await research_agent.process(message)

        # Should include which sources were used
        assert "sources_used" in response.content or "sources_used" in response.metadata


class TestSearchRelevance:
    """Test search result relevance"""

    @pytest.mark.asyncio
    async def test_relevance_scores(self, research_agent):
        """Test results include relevance scores"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "machine learning",
                "sources": ["web"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        # Results may have relevance scores
        has_relevance = any("relevance_score" in r for r in results)

        if has_relevance:
            for result in results:
                if "relevance_score" in result:
                    score = result["relevance_score"]
                    assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(self, research_agent):
        """Test results are sorted by relevance"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "transformers",
                "sources": ["web"]
            }
        )

        response = await research_agent.process(message)

        results = response.content["results"]
        scores = [r.get("relevance_score", 1.0) for r in results]

        # Should be sorted in descending order
        assert scores == sorted(scores, reverse=True) or all(s == scores[0] for s in scores)


class TestResearchAgentErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_empty_query(self, research_agent):
        """Test handling of empty query"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "",
                "sources": ["web"]
            }
        )

        response = await research_agent.process(message)

        # Should handle gracefully
        assert response.message_type in [
            MessageType.RESEARCH_RESPONSE,
            MessageType.AGENT_ERROR
        ]

    @pytest.mark.asyncio
    async def test_missing_query(self, research_agent):
        """Test handling of missing query"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "sources": ["web"]
                # Missing 'query'
            }
        )

        response = await research_agent.process(message)

        # Should handle gracefully
        assert response is not None

    @pytest.mark.asyncio
    async def test_invalid_source(self, research_agent):
        """Test handling of invalid source"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "test",
                "sources": ["invalid_source"]
            }
        )

        response = await research_agent.process(message)

        # Should handle gracefully (may return empty results or error)
        assert response is not None

    @pytest.mark.asyncio
    async def test_no_sources_specified(self, research_agent):
        """Test default sources when none specified"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "test"
                # No sources specified
            }
        )

        response = await research_agent.process(message)

        # Should use default sources
        assert response.message_type == MessageType.RESEARCH_RESPONSE
        assert "results" in response.content


class TestResearchAgentCaching:
    """Test caching functionality"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, research_agent):
        """Test cache returns same results"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "unique search query 12345",
                "sources": ["web"]
            }
        )

        # First call
        response1 = await research_agent.process(message)

        # Second call (should hit cache)
        response2 = await research_agent.process(message)

        # Results should be consistent
        assert response1.content["results"] == response2.content["results"]


class TestResearchAgentPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_search_speed(self, research_agent, timing):
        """Test search completes in reasonable time"""
        message = AgentMessage(
            message_type=MessageType.RESEARCH_REQUEST,
            content={
                "query": "test query",
                "sources": ["web"]
            }
        )

        with timing:
            response = await research_agent.process(message)

        # With mocked search, should be very fast
        assert timing.elapsed_ms < 100

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, research_agent):
        """Test handling concurrent searches"""
        import asyncio

        messages = [
            AgentMessage(
                message_type=MessageType.RESEARCH_REQUEST,
                content={
                    "query": f"query {i}",
                    "sources": ["web"]
                }
            )
            for i in range(3)
        ]

        responses = await asyncio.gather(
            *[research_agent.process(msg) for msg in messages]
        )

        # All should succeed
        assert len(responses) == 3
        assert all(r.message_type == MessageType.RESEARCH_RESPONSE for r in responses)
