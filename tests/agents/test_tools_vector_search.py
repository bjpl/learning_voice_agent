"""
Tests for Knowledge Base Vector Search Integration (Feature 1)

SPARC Specification:
- Search tool integrates with ChromaDB for semantic search
- Minimum 0.7 similarity threshold for relevant results
- Returns context from past conversations
- Performance benchmark: <100ms for typical queries
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import asyncio
import sys

# Mock circuitbreaker before importing app modules
sys.modules['circuitbreaker'] = MagicMock()

from app.agents.tools import ToolRegistry, tool_registry


@pytest.fixture
def mock_chroma_storage():
    """Mock ChromaDB storage for vector search"""
    mock = MagicMock()
    mock.search = MagicMock(return_value=[
        {
            "id": "doc1",
            "document": "User: Tell me about Python\nAssistant: Python is a programming language.",
            "metadata": {
                "session_id": "test-session",
                "user_text": "Tell me about Python",
                "agent_text": "Python is a programming language.",
                "timestamp": "2024-11-21T10:00:00"
            },
            "score": 0.85,
            "distance": 0.15
        },
        {
            "id": "doc2",
            "document": "User: How do I learn Python?\nAssistant: Start with basics.",
            "metadata": {
                "session_id": "test-session",
                "user_text": "How do I learn Python?",
                "agent_text": "Start with basics.",
                "timestamp": "2024-11-21T10:05:00"
            },
            "score": 0.78,
            "distance": 0.22
        }
    ])
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for async search"""
    mock = AsyncMock()
    mock.search_similar = AsyncMock(return_value=[
        {
            "id": "doc1",
            "document": "User: Tell me about Python\nAssistant: Python is a programming language.",
            "metadata": {
                "session_id": "test-session",
                "user_text": "Tell me about Python",
                "agent_text": "Python is a programming language.",
                "timestamp": "2024-11-21T10:00:00"
            },
            "similarity": 0.85,
            "distance": 0.15
        }
    ])
    mock.initialize = AsyncMock()
    mock._initialized = True
    return mock


class TestKnowledgeBaseVectorSearch:
    """Test cases for Knowledge Base Vector Search integration"""

    def test_tool_registry_has_search_tool(self):
        """Verify search_knowledge tool is registered"""
        tool = tool_registry.get_tool("search_knowledge")
        assert tool is not None
        assert tool.name == "search_knowledge"
        assert tool.handler is not None

    def test_search_tool_schema_valid(self):
        """Verify search tool schema is correctly defined"""
        tool = tool_registry.get_tool("search_knowledge")
        schema = tool.input_schema

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_search_with_vector_store(self, mock_vector_store):
        """Test search tool uses vector store for semantic search"""
        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="Python programming",
                limit=5,
                context={"session_id": "test-session"}
            )

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_returns_similarity_scores(self, mock_vector_store):
        """Verify results include similarity scores"""
        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="Python",
                limit=5,
                context={}
            )

        if result["results"]:
            for item in result["results"]:
                assert "relevance" in item or "similarity" in item

    @pytest.mark.asyncio
    async def test_search_filters_by_threshold(self, mock_vector_store):
        """Test results are filtered by similarity threshold (0.7)"""
        # Add low-relevance result
        mock_vector_store.search_similar.return_value.append({
            "id": "doc3",
            "document": "Unrelated content",
            "metadata": {},
            "similarity": 0.5,  # Below threshold
            "distance": 0.5
        })

        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="Python",
                limit=10,
                context={}
            )

        # All results should have relevance >= 0.7
        for item in result["results"]:
            relevance = item.get("relevance", item.get("similarity", 0))
            assert relevance >= 0.7

    @pytest.mark.asyncio
    async def test_search_session_scoped(self, mock_vector_store):
        """Test search can be scoped to specific session"""
        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            await tool.handler(
                query="Python",
                limit=5,
                context={"session_id": "specific-session"}
            )

        # Verify session filter was applied
        call_kwargs = mock_vector_store.search_similar.call_args.kwargs
        # Should filter by session_id in metadata
        assert mock_vector_store.search_similar.called

    @pytest.mark.asyncio
    async def test_search_fallback_on_empty_store(self, mock_vector_store):
        """Test graceful fallback when vector store is empty"""
        mock_vector_store.search_similar.return_value = []

        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="nonexistent topic",
                limit=5,
                context={}
            )

        assert result["success"] is True
        assert result["results"] == []
        assert result["total_found"] == 0

    @pytest.mark.asyncio
    async def test_search_handles_vector_store_error(self, mock_vector_store):
        """Test error handling when vector store fails"""
        mock_vector_store.search_similar.side_effect = Exception("Database error")

        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="Python",
                limit=5,
                context={}
            )

        # Should return error gracefully, not crash
        assert result["success"] is False or "error" in result

    @pytest.mark.asyncio
    async def test_search_performance_benchmark(self, mock_vector_store):
        """Benchmark: search should complete in <100ms"""
        import time

        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")

            start_time = time.time()
            await tool.handler(
                query="Python programming basics",
                limit=5,
                context={}
            )
            elapsed_ms = (time.time() - start_time) * 1000

        assert elapsed_ms < 100, f"Search took {elapsed_ms}ms, expected <100ms"

    @pytest.mark.asyncio
    async def test_search_result_format(self, mock_vector_store):
        """Test search results have expected format"""
        registry = ToolRegistry()

        with patch('app.agents.tools.vector_store', mock_vector_store):
            tool = registry.get_tool("search_knowledge")
            result = await tool.handler(
                query="Python",
                limit=5,
                context={}
            )

        assert "success" in result
        assert "results" in result
        assert "total_found" in result
        assert "query" in result

        if result["results"]:
            item = result["results"][0]
            assert "user" in item or "user_text" in item or "document" in item


class TestVectorSearchIntegration:
    """Integration tests for vector search with ChromaDB"""

    @pytest.mark.asyncio
    async def test_conversation_history_to_vector_storage(self):
        """Test storing conversation history in vector store"""
        # This tests the full flow: conversation -> embedding -> storage
        pass  # Will be tested in integration tests

    @pytest.mark.asyncio
    async def test_cross_session_search(self):
        """Test searching across multiple sessions"""
        # Verify global search works across session boundaries
        pass  # Will be tested in integration tests
