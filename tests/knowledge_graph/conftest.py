"""
Pytest fixtures for knowledge graph tests
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j AsyncDriver for testing"""
    driver = AsyncMock()

    # Mock session
    session = AsyncMock()

    # Mock query results for concept operations (includes 'id' for entity operations)
    concept_result = AsyncMock()
    concept_result.single = AsyncMock(return_value={
        'name': 'test_concept',
        'frequency': 5,
        'description': 'Test description',
        'id': 'entity_12345'  # For add_entity operations
    })

    # Mock query results for relationship queries
    related_result = AsyncMock()
    related_result.values = AsyncMock(return_value=[
        ['related_concept', 'description', 3, ['RELATES_TO'], [0.8], 1],
        ['another_concept', 'another desc', 2, ['BUILDS_ON'], [0.9], 2]
    ])

    # Mock stats result
    stats_result = AsyncMock()
    stats_result.single = AsyncMock(return_value={
        'concept_count': 10,
        'relationship_count': 15,
        'entity_count': 5,
        'session_count': 3,
        'topic_count': 2
    })

    # Default to concept result
    session.run = AsyncMock(return_value=concept_result)

    # Context manager for session - must return fresh context manager each call
    @asynccontextmanager
    async def get_session(database=None):
        yield session

    # Use side_effect to return a new context manager each time session() is called
    driver.session = MagicMock(side_effect=lambda database=None: get_session(database))
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()

    return driver, session, concept_result, related_result, stats_result


@pytest.fixture
def mock_knowledge_graph_store(mock_neo4j_driver):
    """Mock KnowledgeGraphStore for testing"""
    from app.knowledge_graph.graph_store import KnowledgeGraphStore
    from app.knowledge_graph.config import KnowledgeGraphConfig

    driver, session, _, _, _ = mock_neo4j_driver

    config = KnowledgeGraphConfig()
    store = KnowledgeGraphStore(config)
    store.driver = driver
    store._initialized = True

    return store


@pytest.fixture
def sample_concept_data():
    """Sample concept data for testing"""
    return {
        'name': 'machine learning',
        'description': 'AI field focused on learning from data',
        'metadata': {'category': 'technology', 'difficulty': 'intermediate'},
        'topic': 'artificial intelligence'
    }


@pytest.fixture
def sample_relationship_data():
    """Sample relationship data for testing"""
    return {
        'from_concept': 'neural networks',
        'to_concept': 'machine learning',
        'relationship_type': 'BUILDS_ON',
        'strength': 0.9,
        'context': 'Prerequisites discussion'
    }


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        'session_id': 'sess_123',
        'concepts': ['machine learning', 'neural networks', 'deep learning'],
        'entities': [
            ('TensorFlow', 'PRODUCT'),
            ('Google', 'ORG'),
            ('Andrew Ng', 'PERSON')
        ],
        'metadata': {
            'exchange_count': 5,
            'duration': 300,
            'user_satisfaction': 4.5
        }
    }
