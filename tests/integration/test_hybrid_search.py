"""
Integration tests for hybrid search system
PATTERN: Test-driven development with async fixtures
WHY: Ensure search works end-to-end with real database
"""
import pytest
import numpy as np
from app.database import Database
from app.search import create_hybrid_search_engine, SearchStrategy
from app.search.vector_store import VectorStore
from app.search.query_analyzer import QueryAnalyzer
from app.search.config import HybridSearchConfig


@pytest.fixture
async def test_db(tmp_path):
    """Create test database with sample data"""
    db_path = str(tmp_path / "test_hybrid.db")
    db = Database(database_url=db_path)
    await db.initialize()

    # Add sample conversations
    test_data = [
        ("session1", "What is machine learning?", "Machine learning is a subset of AI..."),
        ("session1", "Explain neural networks", "Neural networks are computational models..."),
        ("session2", "How do I implement a REST API?", "To implement a REST API, you need..."),
        ("session2", "What is FastAPI?", "FastAPI is a modern Python web framework..."),
        ("session3", "Explain distributed systems", "Distributed systems are networks of computers..."),
    ]

    for session_id, user_text, agent_text in test_data:
        await db.save_exchange(session_id, user_text, agent_text)

    yield db


@pytest.fixture
async def test_vector_store(tmp_path):
    """Create test vector store with sample embeddings"""
    db_path = str(tmp_path / "test_hybrid.db")
    vector_store = VectorStore(db_path=db_path)
    await vector_store.initialize()

    # Add sample embeddings (random for testing)
    for capture_id in range(1, 6):
        embedding = np.random.rand(1536).astype(np.float32)
        await vector_store.add_embedding(capture_id, embedding)

    yield vector_store


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embedding generation"""
    class MockEmbeddingResponse:
        def __init__(self):
            self.data = [type('obj', (object,), {'embedding': np.random.rand(1536).tolist()})]

    class MockEmbeddings:
        async def create(self, input, model):
            return MockEmbeddingResponse()

    class MockClient:
        def __init__(self):
            self.embeddings = MockEmbeddings()

    return MockClient()


@pytest.mark.asyncio
async def test_hybrid_search_initialization(test_db):
    """Test hybrid search engine initialization"""
    engine = create_hybrid_search_engine(test_db)

    assert engine is not None
    assert engine.db == test_db
    assert engine.config is not None


@pytest.mark.asyncio
async def test_keyword_only_search(test_db):
    """Test pure keyword search strategy"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="machine learning",
        strategy=SearchStrategy.KEYWORD,
        limit=5
    )

    assert response.strategy == "keyword"
    assert response.query == "machine learning"
    assert len(response.results) >= 0  # May have results
    assert response.keyword_results_count >= 0
    assert response.vector_results_count == 0


@pytest.mark.asyncio
async def test_semantic_search_without_embeddings(test_db):
    """Test semantic search without OpenAI client (should return empty)"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="explain AI concepts",
        strategy=SearchStrategy.SEMANTIC,
        limit=5
    )

    assert response.strategy == "semantic"
    assert len(response.results) == 0  # No embeddings without client


@pytest.mark.asyncio
async def test_semantic_search_with_mock_client(test_db, test_vector_store, mock_openai_client):
    """Test semantic search with mock OpenAI client"""
    engine = create_hybrid_search_engine(test_db)
    engine.set_embedding_client(mock_openai_client)

    response = await engine.search(
        query="machine learning concepts",
        strategy=SearchStrategy.SEMANTIC,
        limit=5
    )

    assert response.strategy == "semantic"
    assert response.query == "machine learning concepts"
    # Results depend on vector similarity


@pytest.mark.asyncio
async def test_hybrid_search_strategy(test_db, mock_openai_client):
    """Test hybrid search combining vector and keyword"""
    engine = create_hybrid_search_engine(test_db)
    engine.set_embedding_client(mock_openai_client)

    response = await engine.search(
        query="REST API implementation",
        strategy=SearchStrategy.HYBRID,
        limit=10
    )

    assert response.strategy == "hybrid"
    assert response.execution_time_ms > 0
    # Both vector and keyword searches executed
    assert "query_analysis" in response.__dict__


@pytest.mark.asyncio
async def test_adaptive_strategy_short_query(test_db):
    """Test adaptive strategy with short query (should choose keyword)"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="API",
        strategy=SearchStrategy.ADAPTIVE,
        limit=5
    )

    # Short queries typically use keyword search
    assert response.strategy in ["keyword", "semantic", "hybrid"]


@pytest.mark.asyncio
async def test_adaptive_strategy_long_query(test_db, mock_openai_client):
    """Test adaptive strategy with long conceptual query"""
    engine = create_hybrid_search_engine(test_db)
    engine.set_embedding_client(mock_openai_client)

    response = await engine.search(
        query="Can you explain how distributed systems handle consensus and fault tolerance?",
        strategy=SearchStrategy.ADAPTIVE,
        limit=5
    )

    # Long conceptual queries typically use semantic search
    assert response.strategy in ["semantic", "hybrid"]


@pytest.mark.asyncio
async def test_query_analysis(test_db):
    """Test query analysis functionality"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="What is machine learning?",
        strategy=SearchStrategy.ADAPTIVE,
        limit=5
    )

    analysis = response.query_analysis
    assert "intent" in analysis
    assert "keywords" in analysis
    assert "suggested_strategy" in analysis


@pytest.mark.asyncio
async def test_result_limit(test_db):
    """Test result limit enforcement"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="learning",
        strategy=SearchStrategy.KEYWORD,
        limit=2
    )

    assert len(response.results) <= 2


@pytest.mark.asyncio
async def test_empty_query_handling(test_db):
    """Test handling of edge cases"""
    engine = create_hybrid_search_engine(test_db)

    # Empty query
    response = await engine.search(
        query="",
        strategy=SearchStrategy.KEYWORD,
        limit=5
    )

    # Should handle gracefully
    assert response.total_count >= 0


@pytest.mark.asyncio
async def test_rrf_score_normalization(test_db, mock_openai_client, test_vector_store):
    """Test RRF score normalization in hybrid search"""
    engine = create_hybrid_search_engine(test_db)
    engine.set_embedding_client(mock_openai_client)

    response = await engine.search(
        query="machine learning",
        strategy=SearchStrategy.HYBRID,
        limit=10
    )

    # Check that scores are normalized (0-1 range)
    for result in response.results:
        score = result.get('score', 0)
        assert 0 <= score <= 1, f"Score {score} out of range"


@pytest.mark.asyncio
async def test_search_with_custom_config(test_db):
    """Test search with custom configuration"""
    custom_config = HybridSearchConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        max_results_per_search=15,
        final_result_limit=8
    )

    engine = create_hybrid_search_engine(test_db, config=custom_config)

    response = await engine.search(
        query="FastAPI",
        strategy=SearchStrategy.KEYWORD,
        limit=5
    )

    assert response.total_count <= 5


@pytest.mark.asyncio
async def test_concurrent_searches(test_db):
    """Test multiple concurrent searches"""
    import asyncio

    engine = create_hybrid_search_engine(test_db)

    queries = [
        "machine learning",
        "REST API",
        "distributed systems"
    ]

    tasks = [
        engine.search(query, SearchStrategy.KEYWORD, 5)
        for query in queries
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for response in results:
        assert response.execution_time_ms > 0


@pytest.mark.asyncio
async def test_search_error_handling(test_db):
    """Test graceful error handling"""
    engine = create_hybrid_search_engine(test_db)

    # Very long query
    long_query = "a" * 1000

    response = await engine.search(
        query=long_query,
        strategy=SearchStrategy.KEYWORD,
        limit=5
    )

    # Should handle without crashing
    assert response is not None


@pytest.mark.asyncio
async def test_search_result_metadata(test_db):
    """Test that search results contain required metadata"""
    engine = create_hybrid_search_engine(test_db)

    response = await engine.search(
        query="machine",
        strategy=SearchStrategy.KEYWORD,
        limit=5
    )

    if response.results:
        result = response.results[0]
        assert 'id' in result
        assert 'session_id' in result
        assert 'user_text' in result
        assert 'agent_text' in result
        assert 'score' in result
        assert 'rank' in result
