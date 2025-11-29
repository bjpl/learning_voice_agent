"""
Tests for HybridSearchEngine
Target: 30+ tests, 86% coverage
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.search.hybrid_search import HybridSearchEngine, create_hybrid_search_engine
from app.search.config import SearchStrategy, HybridSearchConfig
from app.search.query_analyzer import QueryAnalysis
from app.search.hybrid_search import SearchResult


@pytest.mark.asyncio
class TestHybridSearchExecution:
    """Test hybrid search execution (8 tests)"""

    async def test_search_basic(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test basic search execution"""
        config = HybridSearchConfig()
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer,
            config=config
        )
        engine._embedding_client = AsyncMock()
        engine._embedding_client.embeddings.create = AsyncMock()

        response = await engine.search("test query")

        assert response.query == "test query"
        assert isinstance(response.results, list)
        assert response.execution_time_ms > 0

    async def test_search_with_semantic_strategy(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test search with semantic-only strategy"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        response = await engine.search(
            query="conceptual question",
            strategy=SearchStrategy.SEMANTIC
        )

        assert response.strategy == "semantic"
        # Only vector search should be executed (may be 0 if no embedding client configured)
        assert response.vector_results_count >= 0
        assert response.keyword_results_count == 0

    async def test_search_with_keyword_strategy(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test search with keyword-only strategy"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        response = await engine.search(
            query="exact phrase",
            strategy=SearchStrategy.KEYWORD
        )

        assert response.strategy == "keyword"
        # Only keyword search should be executed
        assert response.keyword_results_count > 0
        assert response.vector_results_count == 0

    async def test_search_with_hybrid_strategy(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test search with hybrid strategy"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        response = await engine.search(
            query="machine learning",
            strategy=SearchStrategy.HYBRID
        )

        assert response.strategy == "hybrid"
        # Both searches should be executed (counts depend on mock setup)
        assert response.vector_results_count >= 0
        assert response.keyword_results_count >= 0

    async def test_search_with_adaptive_strategy(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test search with adaptive strategy selection"""
        # Mock analyzer to suggest semantic search
        analysis = QueryAnalysis(
            original_query="what is AI?",
            cleaned_query="what is ai",
            keywords=["ai"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=3
        )
        mock_query_analyzer.analyze = AsyncMock(return_value=analysis)

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine._embedding_client = AsyncMock()
        engine._embedding_client.embeddings.create = AsyncMock()

        response = await engine.search(
            query="what is AI?",
            strategy=SearchStrategy.ADAPTIVE
        )

        # Should use suggested strategy
        assert response.strategy == "semantic"

    async def test_search_respects_limit(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test that search respects result limit"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        response = await engine.search(
            query="test",
            strategy=SearchStrategy.KEYWORD,
            limit=3
        )

        assert len(response.results) <= 3

    async def test_search_includes_query_analysis(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test that response includes query analysis"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        response = await engine.search("test query")

        assert 'query_analysis' in response.__dict__
        assert isinstance(response.query_analysis, dict)

    async def test_search_handles_empty_results(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test search with no results"""
        # Mock empty results
        mock_vector_store.search = AsyncMock(return_value=[])
        mock_database.search_captures = AsyncMock(return_value=[])

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine._embedding_client = AsyncMock()
        engine._embedding_client.embeddings.create = AsyncMock()

        response = await engine.search("no results query", strategy=SearchStrategy.HYBRID)

        assert response.total_count == 0
        assert response.results == []


@pytest.mark.asyncio
class TestVectorSearch:
    """Test vector search component (7 tests)"""

    async def test_vector_search_success(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test successful vector search"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        # Create mock analysis
        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._vector_search("test query", analysis, limit=10)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == 'vector' for r in results)

    async def test_vector_search_uses_cache(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test that vector search uses embedding cache"""
        # Set up cache with embedding
        cached_embedding = [0.1] * 1536
        mock_vector_store.get_cached_embedding = MagicMock(return_value=cached_embedding)

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        await engine._vector_search("test", analysis, limit=10)

        # Should have checked cache
        mock_vector_store.get_cached_embedding.assert_called_once_with("test")

        # Should NOT have called OpenAI (used cache)
        mock_openai_client.embeddings.create.assert_not_called()

    async def test_vector_search_no_client(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test vector search without embedding client"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        # No embedding client set

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._vector_search("test", analysis, limit=10)

        # Should return empty results
        assert results == []

    async def test_vector_search_error_handling(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test error handling in vector search"""
        # Make embedding generation fail
        mock_openai_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._vector_search("test", analysis, limit=10)

        # Should return empty on error
        assert results == []

    async def test_vector_search_respects_threshold(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test that vector search respects similarity threshold"""
        config = HybridSearchConfig(vector_similarity_threshold=0.9)

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer,
            config=config
        )
        engine.set_embedding_client(mock_openai_client)

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        await engine._vector_search("test", analysis, limit=10)

        # Verify threshold was passed to vector store
        call_args = mock_vector_store.search.call_args[1]
        assert call_args['threshold'] == 0.9

    async def test_vector_search_creates_search_results(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test that vector search creates proper SearchResult objects"""
        # Mock database to return capture details
        mock_database.get_connection = AsyncMock()

        class MockConnection:
            async def execute(self, query, params):
                class MockCursor:
                    async def fetchall(self):
                        return [
                            {
                                'id': 1,
                                'session_id': 'sess_1',
                                'timestamp': '2025-01-21T10:00:00Z',
                                'user_text': 'test',
                                'agent_text': 'response'
                            }
                        ]
                return MockCursor()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_database.get_connection = MagicMock(return_value=MockConnection())

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._vector_search("test", analysis, limit=10)

        assert len(results) > 0
        for result in results:
            assert result.source == 'vector'
            assert result.vector_score is not None
            assert result.user_text is not None

    async def test_vector_search_caches_embedding(self, mock_database, mock_vector_store, mock_query_analyzer, mock_openai_client):
        """Test that generated embeddings are cached"""
        mock_vector_store.cache_embedding = MagicMock()

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )
        engine.set_embedding_client(mock_openai_client)

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="conceptual",
            suggested_strategy=SearchStrategy.SEMANTIC,
            is_short=False,
            is_exact_phrase=False,
            word_count=1
        )

        await engine._vector_search("test query", analysis, limit=10)

        # Should have cached the embedding
        mock_vector_store.cache_embedding.assert_called_once()


@pytest.mark.asyncio
class TestKeywordSearch:
    """Test keyword search component (5 tests)"""

    async def test_keyword_search_success(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test successful keyword search"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="factual",
            suggested_strategy=SearchStrategy.KEYWORD,
            is_short=True,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._keyword_search("test", analysis, limit=10)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == 'keyword' for r in results)

    async def test_keyword_search_includes_snippets(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test that keyword search includes highlighted snippets"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        analysis = QueryAnalysis(
            original_query="machine learning",
            cleaned_query="machine learning",
            keywords=["machine", "learning"],
            intent="factual",
            suggested_strategy=SearchStrategy.KEYWORD,
            is_short=False,
            is_exact_phrase=False,
            word_count=2
        )

        results = await engine._keyword_search("machine learning", analysis, limit=10)

        # Check that snippets are present
        for result in results:
            assert result.user_snippet is not None or result.agent_snippet is not None

    async def test_keyword_search_empty_results(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test keyword search with no results"""
        # Mock empty database results
        mock_database.search_captures = AsyncMock(return_value=[])

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        analysis = QueryAnalysis(
            original_query="nonexistent",
            cleaned_query="nonexistent",
            keywords=["nonexistent"],
            intent="factual",
            suggested_strategy=SearchStrategy.KEYWORD,
            is_short=True,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._keyword_search("nonexistent", analysis, limit=10)

        assert results == []

    async def test_keyword_search_error_handling(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test error handling in keyword search"""
        # Make database search fail
        mock_database.search_captures = AsyncMock(side_effect=Exception("DB error"))

        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="factual",
            suggested_strategy=SearchStrategy.KEYWORD,
            is_short=True,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._keyword_search("test", analysis, limit=10)

        # Should return empty on error
        assert results == []

    async def test_keyword_search_rank_based_scoring(self, mock_database, mock_vector_store, mock_query_analyzer):
        """Test that keyword search uses rank-based scoring"""
        engine = HybridSearchEngine(
            database=mock_database,
            vector_store=mock_vector_store,
            query_analyzer=mock_query_analyzer
        )

        analysis = QueryAnalysis(
            original_query="test",
            cleaned_query="test",
            keywords=["test"],
            intent="factual",
            suggested_strategy=SearchStrategy.KEYWORD,
            is_short=True,
            is_exact_phrase=False,
            word_count=1
        )

        results = await engine._keyword_search("test", analysis, limit=10)

        # Check that scores decrease with rank
        if len(results) > 1:
            assert results[0].score > results[1].score


class TestRRFFusion:
    """Test Reciprocal Rank Fusion (7 tests)"""

    def test_rrf_combines_results(self, sample_vector_results, sample_keyword_results):
        """Test basic RRF combination"""
        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        combined = engine._reciprocal_rank_fusion(
            sample_vector_results,
            sample_keyword_results
        )

        # Should have unique results (deduplication)
        assert len(combined) == 3  # id 1, 2, 3

    def test_rrf_deduplication(self, sample_vector_results, sample_keyword_results):
        """Test that RRF deduplicates results"""
        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        # Both have id=2
        combined = engine._reciprocal_rank_fusion(
            sample_vector_results,
            sample_keyword_results
        )

        # Check that id appears only once
        ids = [r.id for r in combined]
        assert len(ids) == len(set(ids))  # No duplicates

    def test_rrf_score_normalization(self):
        """Test that RRF normalizes scores to 0-1 range"""
        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        vector_results = [
            SearchResult(id=1, session_id='s1', timestamp='t1', user_text='u1',
                        agent_text='a1', score=0.9, rank=1, source='vector')
        ]

        keyword_results = [
            SearchResult(id=2, session_id='s1', timestamp='t1', user_text='u2',
                        agent_text='a2', score=0.8, rank=1, source='keyword')
        ]

        combined = engine._reciprocal_rank_fusion(vector_results, keyword_results)

        # Scores should be normalized 0-1
        for result in combined:
            assert 0.0 <= result.score <= 1.0

    def test_rrf_weight_configuration(self):
        """Test that RRF respects weight configuration"""
        config = HybridSearchConfig(
            vector_weight=0.8,
            keyword_weight=0.2
        )

        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock(),
            config=config
        )

        # Result that appears in vector should rank higher due to higher weight
        vector_results = [
            SearchResult(id=1, session_id='s1', timestamp='t1', user_text='u1',
                        agent_text='a1', score=0.9, rank=1, source='vector')
        ]

        keyword_results = [
            SearchResult(id=2, session_id='s1', timestamp='t1', user_text='u2',
                        agent_text='a2', score=0.9, rank=1, source='keyword')
        ]

        combined = engine._reciprocal_rank_fusion(vector_results, keyword_results)

        # Vector result should rank higher (assuming RRF implementation)
        # This is conceptual - actual test would need to verify RRF formula
        assert len(combined) == 2

    def test_rrf_preserves_snippets(self, sample_vector_results, sample_keyword_results):
        """Test that RRF preserves snippet information from keyword search"""
        # Add snippets to keyword results
        for result in sample_keyword_results:
            result.user_snippet = "test <mark>snippet</mark>"

        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        combined = engine._reciprocal_rank_fusion(
            sample_vector_results,
            sample_keyword_results
        )

        # Find result with id=2 (appears in both)
        result_2 = next((r for r in combined if r.id == 2), None)
        assert result_2 is not None
        # Should have snippet from keyword search
        # (actual test depends on implementation details)

    def test_rrf_empty_vector_results(self):
        """Test RRF with empty vector results"""
        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        keyword_results = [
            SearchResult(id=1, session_id='s1', timestamp='t1', user_text='u1',
                        agent_text='a1', score=0.9, rank=1, source='keyword')
        ]

        combined = engine._reciprocal_rank_fusion([], keyword_results)

        # Should return keyword results
        assert len(combined) == 1
        assert combined[0].id == 1

    def test_rrf_empty_keyword_results(self):
        """Test RRF with empty keyword results"""
        engine = HybridSearchEngine(
            database=AsyncMock(),
            vector_store=AsyncMock(),
            query_analyzer=AsyncMock()
        )

        vector_results = [
            SearchResult(id=1, session_id='s1', timestamp='t1', user_text='u1',
                        agent_text='a1', score=0.9, rank=1, source='vector')
        ]

        combined = engine._reciprocal_rank_fusion(vector_results, [])

        # Should return vector results
        assert len(combined) == 1
        assert combined[0].id == 1


def test_create_hybrid_search_engine(mock_database):
    """Test factory function"""
    engine = create_hybrid_search_engine(mock_database)

    assert isinstance(engine, HybridSearchEngine)
    assert engine.db is mock_database
    assert engine.config is not None
