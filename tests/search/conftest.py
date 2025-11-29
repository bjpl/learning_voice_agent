"""
Pytest fixtures for search tests
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
import numpy as np


@pytest.fixture
def mock_database():
    """Mock Database instance for search tests"""
    db = AsyncMock()

    # Mock search_captures for FTS5
    db.search_captures = AsyncMock(return_value=[
        {
            'id': 1,
            'session_id': 'sess_1',
            'timestamp': '2025-01-21T10:00:00Z',
            'user_text': 'What is machine learning?',
            'agent_text': 'Machine learning is a subset of AI...',
            'user_snippet': 'What is <mark>machine</mark> <mark>learning</mark>?',
            'agent_snippet': '<mark>Machine</mark> <mark>learning</mark> is...'
        },
        {
            'id': 2,
            'session_id': 'sess_1',
            'timestamp': '2025-01-21T10:05:00Z',
            'user_text': 'How does neural network work?',
            'agent_text': 'Neural networks consist of layers...',
            'user_snippet': 'How does <mark>neural</mark> <mark>network</mark> work?',
            'agent_snippet': '<mark>Neural</mark> <mark>networks</mark> consist...'
        }
    ])

    db.get_all_captures = AsyncMock(return_value=[])

    # Mock get_connection as async context manager
    class MockConnection:
        async def execute(self, query, params=None):
            class MockCursor:
                async def fetchall(self):
                    return [
                        {
                            'id': 1,
                            'session_id': 'sess_1',
                            'timestamp': '2025-01-21T10:00:00Z',
                            'user_text': 'What is machine learning?',
                            'agent_text': 'Machine learning is a subset of AI...'
                        },
                        {
                            'id': 2,
                            'session_id': 'sess_1',
                            'timestamp': '2025-01-21T10:05:00Z',
                            'user_text': 'How does neural network work?',
                            'agent_text': 'Neural networks consist of layers...'
                        },
                        {
                            'id': 3,
                            'session_id': 'sess_2',
                            'timestamp': '2025-01-21T10:10:00Z',
                            'user_text': 'Deep learning basics',
                            'agent_text': 'Deep learning uses multiple layers...'
                        }
                    ]
            return MockCursor()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    db.get_connection = MagicMock(return_value=MockConnection())

    return db


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for search tests"""
    store = AsyncMock()

    # Mock search method
    async def mock_search(query_embedding, limit=10, threshold=0.7):
        # Return list of (id, score) tuples
        return [
            (1, 0.95),
            (2, 0.87),
            (3, 0.82)
        ]

    store.search = AsyncMock(side_effect=mock_search)

    # Mock cache methods
    store.get_cached_embedding = Mock(return_value=None)
    store.cache_embedding = Mock()

    return store


@pytest.fixture
def mock_query_analyzer():
    """Mock QueryAnalyzer for search tests"""
    from app.search.config import SearchStrategy
    from app.search.query_analyzer import QueryAnalysis

    analyzer = AsyncMock()

    # Default analysis
    default_analysis = QueryAnalysis(
        original_query="test query",
        cleaned_query="test query",
        keywords=["test", "query"],
        intent="conceptual",
        suggested_strategy=SearchStrategy.HYBRID,
        is_short=False,
        is_exact_phrase=False,
        word_count=2
    )

    analyzer.analyze = AsyncMock(return_value=default_analysis)

    return analyzer


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embeddings"""
    client = AsyncMock()

    # Mock embeddings.create
    async def mock_create(input, model):
        class MockEmbeddingData:
            def __init__(self):
                self.embedding = np.random.rand(1536).tolist()

        class MockEmbeddingResponse:
            def __init__(self):
                self.data = [MockEmbeddingData()]

        return MockEmbeddingResponse()

    client.embeddings = AsyncMock()
    client.embeddings.create = AsyncMock(side_effect=mock_create)

    return client


@pytest.fixture
def sample_search_query():
    """Sample search query for testing"""
    return "machine learning concepts"


@pytest.fixture
def sample_vector_results():
    """Sample vector search results"""
    from app.search.hybrid_search import SearchResult

    return [
        SearchResult(
            id=1,
            session_id='sess_1',
            timestamp='2025-01-21T10:00:00Z',
            user_text='What is machine learning?',
            agent_text='Machine learning is...',
            score=0.95,
            rank=1,
            source='vector',
            vector_score=0.95
        ),
        SearchResult(
            id=2,
            session_id='sess_1',
            timestamp='2025-01-21T10:05:00Z',
            user_text='Explain neural networks',
            agent_text='Neural networks are...',
            score=0.87,
            rank=2,
            source='vector',
            vector_score=0.87
        )
    ]


@pytest.fixture
def sample_keyword_results():
    """Sample keyword search results"""
    from app.search.hybrid_search import SearchResult

    return [
        SearchResult(
            id=2,
            session_id='sess_1',
            timestamp='2025-01-21T10:05:00Z',
            user_text='Explain neural networks',
            agent_text='Neural networks are...',
            score=0.90,
            rank=1,
            source='keyword',
            keyword_score=0.90
        ),
        SearchResult(
            id=3,
            session_id='sess_2',
            timestamp='2025-01-21T10:10:00Z',
            user_text='Deep learning basics',
            agent_text='Deep learning uses...',
            score=0.85,
            rank=2,
            source='keyword',
            keyword_score=0.85
        )
    ]
