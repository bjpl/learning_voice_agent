"""
Pytest fixtures for vector tests
"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import numpy as np
from typing import Dict, Any


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing"""
    client = MagicMock()

    # Mock collection
    collection = MagicMock()
    collection.count.return_value = 100
    collection.add = MagicMock()
    collection.delete = MagicMock()
    collection.query = MagicMock(return_value={
        'ids': [['id1', 'id2', 'id3']],
        'documents': [['Document 1 text', 'Document 2 text', 'Document 3 text']],
        'metadatas': [[
            {'session_id': 'sess_1', 'timestamp': '2025-01-21T10:00:00Z'},
            {'session_id': 'sess_1', 'timestamp': '2025-01-21T10:05:00Z'},
            {'session_id': 'sess_2', 'timestamp': '2025-01-21T10:10:00Z'}
        ]],
        'distances': [[0.1, 0.2, 0.3]]
    })

    client.get_or_create_collection.return_value = collection
    client.list_collections.return_value = [collection]
    client.delete_collection = MagicMock()

    return client


@pytest.fixture
def mock_vector_store(mock_chroma_client, monkeypatch):
    """Mock VectorStore instance"""
    from app.vector.vector_store import VectorStore

    store = VectorStore()
    monkeypatch.setattr(store, 'client', mock_chroma_client)
    store._initialized = True
    store.collections = {
        'conversations': mock_chroma_client.get_or_create_collection()
    }

    return store


@pytest.fixture
def mock_embedding_generator():
    """Mock EmbeddingGenerator for testing"""
    from app.vector.embeddings import EmbeddingGenerator

    generator = MagicMock(spec=EmbeddingGenerator)
    generator._initialized = True

    # Mock generate_embedding to return random 384-dim vector
    async def mock_generate(text, use_cache=True):
        return np.random.rand(384).astype(np.float32)

    generator.generate_embedding = AsyncMock(side_effect=mock_generate)

    # Mock generate_batch
    async def mock_generate_batch(texts, batch_size=None, show_progress=False):
        return [np.random.rand(384).astype(np.float32) for _ in texts]

    generator.generate_batch = AsyncMock(side_effect=mock_generate_batch)

    generator.get_model_info = MagicMock(return_value={
        'status': 'initialized',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimensions': 384,
        'cache': {'size': 10, 'max_size': 1000}
    })

    generator.clear_cache = MagicMock()

    return generator


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing"""
    return {
        'text1': np.random.rand(384).astype(np.float32),
        'text2': np.random.rand(384).astype(np.float32),
        'text3': np.random.rand(384).astype(np.float32)
    }


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing"""
    return [
        {
            'session_id': 'sess_1',
            'timestamp': '2025-01-21T10:00:00Z',
            'exchange_type': 'user',
            'speaker': 'user'
        },
        {
            'session_id': 'sess_1',
            'timestamp': '2025-01-21T10:01:00Z',
            'exchange_type': 'agent',
            'speaker': 'agent'
        },
        {
            'session_id': 'sess_2',
            'timestamp': '2025-01-21T10:05:00Z',
            'exchange_type': 'user',
            'speaker': 'user'
        }
    ]


@pytest.fixture
def vector_config():
    """Sample vector configuration"""
    from app.vector.config import VectorConfig, CollectionConfig
    from pathlib import Path
    import tempfile

    temp_dir = tempfile.mkdtemp()

    config = VectorConfig(
        persist_directory=Path(temp_dir) / "chromadb",
        default_n_results=10,
        similarity_threshold=0.7
    )

    return config
