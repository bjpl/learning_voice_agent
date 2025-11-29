"""
Integration tests for Phase 3 RAG system

Tests the complete RAG pipeline:
- RAGRetriever
- ContextBuilder
- RAGGenerator

Run:
    pytest tests/test_rag_integration.py -v
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from app.rag import (
    RAGRetriever,
    ContextBuilder,
    RAGGenerator,
    RetrievalResult,
    rag_config
)
from app.rag.context_builder import ContextFormat
from app.database import Database
from app.search.hybrid_search import HybridSearchEngine, HybridSearchResponse


@pytest.fixture
async def mock_database():
    """Mock database for testing"""
    db = Mock(spec=Database)
    db.get_connection = AsyncMock()
    return db


@pytest.fixture
async def mock_hybrid_search():
    """Mock hybrid search engine"""
    engine = Mock(spec=HybridSearchEngine)

    # Mock search response
    async def mock_search(query, strategy=None, limit=10):
        return HybridSearchResponse(
            query=query,
            strategy="hybrid",
            results=[
                {
                    'id': 1,
                    'session_id': 'session-1',
                    'timestamp': '2024-01-01T10:00:00',
                    'user_text': 'What is machine learning?',
                    'agent_text': 'Machine learning is a subset of AI...',
                    'score': 0.95,
                    'rank': 1,
                    'source': 'hybrid',
                    'vector_score': 0.92,
                    'keyword_score': 0.88
                },
                {
                    'id': 2,
                    'session_id': 'session-1',
                    'timestamp': '2024-01-01T10:05:00',
                    'user_text': 'How do neural networks work?',
                    'agent_text': 'Neural networks are composed of layers...',
                    'score': 0.85,
                    'rank': 2,
                    'source': 'hybrid',
                    'vector_score': 0.83,
                    'keyword_score': 0.78
                }
            ],
            total_count=2,
            query_analysis={},
            execution_time_ms=50.0,
            vector_results_count=2,
            keyword_results_count=2
        )

    engine.search = mock_search
    return engine


@pytest.fixture
async def mock_anthropic_client():
    """Mock Anthropic client"""
    client = AsyncMock()

    # Mock message response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a generated response based on the provided context.")]
    mock_response.model = "claude-3-5-sonnet-20241022"
    mock_response.stop_reason = "stop"
    mock_response.usage = Mock(input_tokens=500, output_tokens=100)

    client.messages.create = AsyncMock(return_value=mock_response)

    return client


@pytest.mark.asyncio
class TestRAGRetriever:
    """Test RAGRetriever"""

    async def test_retrieve_basic(self, mock_database, mock_hybrid_search):
        """Test basic retrieval"""
        retriever = RAGRetriever(mock_database, mock_hybrid_search)

        response = await retriever.retrieve("What is ML?", top_k=5)

        assert response.query == "What is ML?"
        assert response.total_retrieved > 0
        assert response.strategy_used == "hybrid"
        assert all(isinstance(r, RetrievalResult) for r in response.results)

    async def test_retrieve_with_session(self, mock_database, mock_hybrid_search):
        """Test session-scoped retrieval"""
        retriever = RAGRetriever(mock_database, mock_hybrid_search)

        # Enable session scoping
        retriever.config.session_scoped_search = True

        response = await retriever.retrieve(
            "What is ML?",
            session_id="session-1",
            top_k=3
        )

        assert response.session_scoped is True
        assert response.session_id == "session-1"

    async def test_retrieve_filters_by_threshold(self, mock_database, mock_hybrid_search):
        """Test relevance threshold filtering"""
        retriever = RAGRetriever(mock_database, mock_hybrid_search)
        retriever.config.relevance_threshold = 0.9  # High threshold

        response = await retriever.retrieve("What is ML?")

        # All results should meet threshold
        assert all(r.score >= 0.9 for r in response.results)

    async def test_retrieve_timeout(self, mock_database, mock_hybrid_search):
        """Test retrieval timeout"""
        # Make search take too long
        async def slow_search(query, strategy=None, limit=10):
            await asyncio.sleep(10)  # Longer than timeout
            return HybridSearchResponse(
                query=query, strategy="hybrid", results=[],
                total_count=0, query_analysis={}, execution_time_ms=10000
            )

        mock_hybrid_search.search = slow_search

        retriever = RAGRetriever(mock_database, mock_hybrid_search)
        retriever.config.retrieval_timeout = 0.1  # Short timeout

        response = await retriever.retrieve("What is ML?")

        # Should return empty results on timeout
        assert response.total_retrieved == 0


@pytest.mark.asyncio
class TestContextBuilder:
    """Test ContextBuilder"""

    def create_sample_results(self):
        """Create sample retrieval results"""
        return [
            RetrievalResult(
                id=1,
                session_id="session-1",
                timestamp="2024-01-01T10:00:00",
                user_text="What is machine learning?",
                agent_text="Machine learning is a subset of AI that enables systems to learn from data.",
                score=0.95,
                rank=1,
                source="hybrid",
                final_score=0.95,
                age_days=1.0
            ),
            RetrievalResult(
                id=2,
                session_id="session-1",
                timestamp="2024-01-01T10:05:00",
                user_text="How do neural networks work?",
                agent_text="Neural networks are composed of interconnected layers of nodes.",
                score=0.85,
                rank=2,
                source="hybrid",
                final_score=0.85,
                age_days=1.0
            )
        ]

    async def test_build_context_basic(self):
        """Test basic context building"""
        builder = ContextBuilder()
        results = self.create_sample_results()

        context = await builder.build_context(
            retrieval_results=results,
            query="What is ML?"
        )

        assert context.query == "What is ML?"
        assert context.document_count == 2
        assert context.total_tokens > 0
        assert len(context.formatted_context) > 0

    async def test_build_context_structured_format(self):
        """Test structured formatting"""
        builder = ContextBuilder()
        results = self.create_sample_results()

        context = await builder.build_context(
            retrieval_results=results,
            query="What is ML?",
            format_type=ContextFormat.STRUCTURED
        )

        assert "# Relevant Conversation History" in context.formatted_context
        assert "Conversation 1" in context.formatted_context

    async def test_build_context_compact_format(self):
        """Test compact formatting"""
        builder = ContextBuilder()
        results = self.create_sample_results()

        context = await builder.build_context(
            retrieval_results=results,
            query="What is ML?",
            format_type=ContextFormat.COMPACT
        )

        # Compact format should be shorter
        assert "User:" in context.formatted_context
        assert "Assistant:" in context.formatted_context

    async def test_token_budget_enforcement(self):
        """Test context fits within token budget"""
        builder = ContextBuilder()
        results = self.create_sample_results()

        context = await builder.build_context(
            retrieval_results=results,
            query="What is ML?",
            max_tokens=100  # Very small budget
        )

        # Should truncate or summarize to fit
        assert context.total_tokens <= 100 or context.is_summarized

    async def test_empty_results(self):
        """Test with empty results"""
        builder = ContextBuilder()

        context = await builder.build_context(
            retrieval_results=[],
            query="What is ML?"
        )

        assert context.document_count == 0
        assert context.formatted_context == ""
        assert context.total_tokens == 0


@pytest.mark.asyncio
class TestRAGGenerator:
    """Test RAGGenerator"""

    async def test_generate_with_context(self, mock_anthropic_client):
        """Test generation with context"""
        generator = RAGGenerator(mock_anthropic_client)

        # Create mock context
        builder = ContextBuilder()
        results = [
            RetrievalResult(
                id=1, session_id="s1", timestamp="2024-01-01T10:00:00",
                user_text="What is ML?",
                agent_text="ML is a subset of AI.",
                score=0.95, rank=1, source="hybrid",
                final_score=0.95, age_days=1.0
            )
        ]
        context = await builder.build_context(results, "What is ML?")

        response = await generator.generate(
            query="What is ML?",
            context=context
        )

        assert response.mode == "rag"
        assert response.context_used is True
        assert len(response.response_text) > 0
        assert response.tokens_used > 0

    async def test_generate_without_context(self, mock_anthropic_client):
        """Test generation without context (basic mode)"""
        generator = RAGGenerator(mock_anthropic_client)

        response = await generator.generate(
            query="What is ML?",
            context=None
        )

        assert response.mode == "basic"
        assert response.context_used is False
        assert len(response.response_text) > 0

    async def test_generate_batch(self, mock_anthropic_client):
        """Test batch generation"""
        generator = RAGGenerator(mock_anthropic_client)

        queries = ["What is ML?", "What is AI?", "What is NLP?"]

        responses = await generator.generate_batch(queries)

        assert len(responses) == 3
        assert all(r.response_text for r in responses)

    async def test_citation_extraction(self, mock_anthropic_client):
        """Test citation extraction"""
        # Mock response with citations
        mock_response = Mock()
        mock_response.content = [Mock(text="Based on Conversation 1, ML is a subset of AI.")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.stop_reason = "stop"
        mock_response.usage = Mock(input_tokens=500, output_tokens=100)

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        generator = RAGGenerator(mock_anthropic_client)
        generator.config.enable_citations = True

        # Create context
        builder = ContextBuilder()
        results = [
            RetrievalResult(
                id=1, session_id="s1", timestamp="2024-01-01T10:00:00",
                user_text="What is ML?",
                agent_text="ML is a subset of AI.",
                score=0.95, rank=1, source="hybrid",
                final_score=0.95, age_days=1.0
            )
        ]
        context = await builder.build_context(results, "What is ML?")

        response = await generator.generate("What is ML?", context)

        # Should extract citation for "Conversation 1"
        assert len(response.citations) > 0


@pytest.mark.asyncio
class TestRAGPipeline:
    """Test complete RAG pipeline integration"""

    async def test_end_to_end_pipeline(
        self,
        mock_database,
        mock_hybrid_search,
        mock_anthropic_client
    ):
        """Test complete RAG pipeline"""
        # Initialize components
        retriever = RAGRetriever(mock_database, mock_hybrid_search)
        context_builder = ContextBuilder()
        generator = RAGGenerator(mock_anthropic_client)

        query = "What is machine learning?"

        # Execute pipeline
        # 1. Retrieve
        retrieval_response = await retriever.retrieve(query, top_k=5)
        assert retrieval_response.total_retrieved > 0

        # 2. Build context
        context = await context_builder.build_context(
            retrieval_results=retrieval_response.results,
            query=query
        )
        assert context.document_count > 0
        assert context.total_tokens > 0

        # 3. Generate
        generation_response = await generator.generate(
            query=query,
            context=context
        )
        assert generation_response.mode == "rag"
        assert generation_response.context_used is True
        assert len(generation_response.response_text) > 0

        # Verify metadata
        assert generation_response.query == query
        assert generation_response.tokens_used > 0
        # generation_time_ms may be 0 in mocked tests
        assert generation_response.generation_time_ms >= 0

    async def test_pipeline_with_fallback(
        self,
        mock_database,
        mock_hybrid_search,
        mock_anthropic_client
    ):
        """Test pipeline with fallback on retrieval failure"""
        # Make retrieval return no results
        async def empty_search(query, strategy=None, limit=10):
            return HybridSearchResponse(
                query=query, strategy="hybrid", results=[],
                total_count=0, query_analysis={}, execution_time_ms=10.0
            )

        mock_hybrid_search.search = empty_search

        retriever = RAGRetriever(mock_database, mock_hybrid_search)
        context_builder = ContextBuilder()
        generator = RAGGenerator(mock_anthropic_client)

        query = "What is quantum computing?"

        # Execute pipeline
        retrieval_response = await retriever.retrieve(query)
        assert retrieval_response.total_retrieved == 0

        # Build empty context
        context = await context_builder.build_context(
            retrieval_results=[],
            query=query
        )
        assert context.document_count == 0

        # Generate should fall back to basic mode
        generation_response = await generator.generate(query, context)
        assert generation_response.mode == "basic"
        assert generation_response.context_used is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
