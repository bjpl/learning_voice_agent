"""
Test Vector Database Integration

This test suite validates the vector database layer functionality,
including embedding generation, vector storage, and semantic search.

Author: Claude Code Agent
Version: 2.0.0
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from app.vector import VectorStore, EmbeddingGenerator, VectorConfig
from app.vector.config import EmbeddingModelConfig, CollectionConfig


@pytest.fixture
async def vector_config():
    """Create test vector configuration with temporary directory"""
    config = VectorConfig(
        persist_directory=Path("./data/chromadb_test")
    )
    yield config
    # Cleanup is handled by ChromaDB


@pytest.fixture
async def embedding_generator(vector_config):
    """Create and initialize embedding generator"""
    generator = EmbeddingGenerator(config=vector_config)
    await generator.initialize()
    yield generator
    await generator.close()


@pytest.fixture
async def vector_store(vector_config, embedding_generator):
    """Create and initialize vector store"""
    store = VectorStore(config=vector_config)
    await store.initialize()
    yield store
    await store.close()


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator"""

    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_generator):
        """Test single embedding generation"""
        text = "Hello, this is a test sentence."
        embedding = await embedding_generator.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension
        assert not np.any(np.isnan(embedding))

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedding_generator):
        """Test batch embedding generation"""
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence",
        ]
        embeddings = await embedding_generator.generate_batch(texts)

        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] == 384

    @pytest.mark.asyncio
    async def test_embedding_cache(self, embedding_generator):
        """Test embedding cache functionality"""
        text = "Cached test sentence"

        # First generation should not be cached
        embedding1 = await embedding_generator.generate_embedding(text)

        # Second generation should use cache
        embedding2 = await embedding_generator.generate_embedding(text)

        # Should be identical
        assert np.array_equal(embedding1, embedding2)

        # Verify cache hit
        cache_stats = embedding_generator.cache.stats()
        assert cache_stats["size"] > 0

    @pytest.mark.asyncio
    async def test_model_info(self, embedding_generator):
        """Test getting model information"""
        info = embedding_generator.get_model_info()

        assert info["status"] == "initialized"
        assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info["dimensions"] == 384
        assert "cache" in info


class TestVectorStore:
    """Test suite for VectorStore"""

    @pytest.mark.asyncio
    async def test_add_embedding(self, vector_store):
        """Test adding a single embedding"""
        doc_id = await vector_store.add_embedding(
            collection_name="conversations",
            text="Hello world",
            metadata={"session_id": "test123", "speaker": "user"},
        )

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    @pytest.mark.asyncio
    async def test_add_batch_embeddings(self, vector_store):
        """Test adding multiple embeddings in batch"""
        texts = [
            "First conversation message",
            "Second conversation message",
            "Third conversation message",
        ]
        metadatas = [
            {"session_id": "test123", "speaker": "user"},
            {"session_id": "test123", "speaker": "agent"},
            {"session_id": "test123", "speaker": "user"},
        ]

        doc_ids = await vector_store.add_batch(
            collection_name="conversations",
            texts=texts,
            metadatas=metadatas,
        )

        assert len(doc_ids) == len(texts)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)

    @pytest.mark.asyncio
    async def test_search_similar(self, vector_store):
        """Test semantic similarity search"""
        # Add test data
        texts = [
            "The weather is sunny today",
            "I love programming in Python",
            "The sky is blue and clear",
            "Machine learning is fascinating",
        ]

        await vector_store.add_batch(
            collection_name="conversations",
            texts=texts,
            metadatas=[{"session_id": "test123"} for _ in texts],
        )

        # Search for similar texts
        results = await vector_store.search_similar(
            collection_name="conversations",
            query_text="What's the weather like?",
            n_results=2,
        )

        assert len(results) > 0
        assert "document" in results[0]
        assert "similarity" in results[0]
        assert "metadata" in results[0]

        # Weather-related texts should rank higher
        assert any(
            "weather" in result["document"].lower() or "sky" in result["document"].lower()
            for result in results[:2]
        )

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, vector_store):
        """Test search with metadata filtering"""
        # Add test data with different session IDs
        texts = ["Message 1", "Message 2", "Message 3"]
        metadatas = [
            {"session_id": "session1"},
            {"session_id": "session2"},
            {"session_id": "session1"},
        ]

        await vector_store.add_batch(
            collection_name="conversations",
            texts=texts,
            metadatas=metadatas,
        )

        # Search with filter
        results = await vector_store.search_similar(
            collection_name="conversations",
            query_text="Message",
            n_results=10,
            metadata_filter={"session_id": "session1"},
        )

        # Should only return results from session1
        assert all(result["metadata"]["session_id"] == "session1" for result in results)

    @pytest.mark.asyncio
    async def test_delete_embedding(self, vector_store):
        """Test deleting an embedding"""
        # Add an embedding
        doc_id = await vector_store.add_embedding(
            collection_name="conversations",
            text="Test deletion",
            metadata={"session_id": "test123"},
        )

        # Delete it
        success = await vector_store.delete_embedding(
            collection_name="conversations",
            document_id=doc_id,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_collection_stats(self, vector_store):
        """Test getting collection statistics"""
        # Add some data
        await vector_store.add_batch(
            collection_name="conversations",
            texts=["Text 1", "Text 2", "Text 3"],
            metadatas=[{"session_id": "test123"} for _ in range(3)],
        )

        # Get stats
        stats = await vector_store.get_collection_stats("conversations")

        assert "name" in stats
        assert "count" in stats
        assert stats["count"] >= 3

    @pytest.mark.asyncio
    async def test_list_collections(self, vector_store):
        """Test listing all collections"""
        collections = await vector_store.list_collections()

        assert isinstance(collections, list)
        assert "conversations" in collections

    @pytest.mark.asyncio
    async def test_create_custom_collection(self, vector_store):
        """Test creating a custom collection"""
        await vector_store.create_collection(
            name="custom_test",
            metadata_schema={"custom_field": "str"},
            distance_metric="cosine",
        )

        collections = await vector_store.list_collections()
        assert "custom_test" in collections


class TestVectorConfig:
    """Test suite for VectorConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = VectorConfig()

        assert config.persist_directory.exists()
        assert config.default_n_results == 10
        assert config.similarity_threshold == 0.7
        assert "conversations" in config.collections

    def test_collection_schema(self):
        """Test collection schema definition"""
        config = VectorConfig()
        schema = config.get_collection_config("conversations")

        assert schema is not None
        assert schema.name == "conversations"
        assert "session_id" in schema.metadata_schema
        assert schema.distance_metric == "cosine"

    def test_custom_collection(self):
        """Test adding custom collection"""
        config = VectorConfig()

        custom_config = CollectionConfig(
            name="custom",
            metadata_schema={"field1": "str", "field2": "int"},
            distance_metric="l2",
        )

        config.add_collection(custom_config)

        assert "custom" in config.collections
        assert config.get_collection_config("custom") == custom_config

    def test_config_to_dict(self):
        """Test configuration export to dictionary"""
        config = VectorConfig()
        config_dict = config.to_dict()

        assert "persist_directory" in config_dict
        assert "embedding_model" in config_dict
        assert "collections" in config_dict
        assert "search" in config_dict


class TestIntegration:
    """Integration tests for the complete vector database layer"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, vector_store):
        """Test complete workflow from adding to searching"""
        # Add conversation data
        conversation_texts = [
            "User: What's the weather like today?",
            "Agent: The weather is sunny and warm.",
            "User: Should I bring an umbrella?",
            "Agent: No need for an umbrella today.",
            "User: What about tomorrow?",
            "Agent: Tomorrow will be cloudy with possible rain.",
        ]

        metadatas = [
            {
                "session_id": "session123",
                "exchange_type": "user" if i % 2 == 0 else "agent",
                "speaker": "user" if i % 2 == 0 else "agent",
            }
            for i in range(len(conversation_texts))
        ]

        # Add in batch
        doc_ids = await vector_store.add_batch(
            collection_name="conversations",
            texts=conversation_texts,
            metadatas=metadatas,
        )

        assert len(doc_ids) == len(conversation_texts)

        # Search for weather-related queries
        results = await vector_store.search_similar(
            collection_name="conversations",
            query_text="Will it rain?",
            n_results=3,
        )

        assert len(results) > 0

        # Verify results contain weather-related content
        weather_keywords = ["weather", "rain", "sunny", "cloudy", "umbrella"]
        assert any(
            any(keyword in result["document"].lower() for keyword in weather_keywords)
            for result in results
        )

        # Get collection stats
        stats = await vector_store.get_collection_stats("conversations")
        assert stats["count"] >= len(conversation_texts)

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test that data persists across store instances"""
        # Create first store and add data
        config = VectorConfig(persist_directory=Path("./data/chromadb_persistence_test"))
        store1 = VectorStore(config=config)
        await store1.initialize()

        test_text = "This should persist across restarts"
        doc_id = await store1.add_embedding(
            collection_name="conversations",
            text=test_text,
            metadata={"test": "persistence"},
        )

        await store1.close()

        # Create new store instance and verify data exists
        store2 = VectorStore(config=config)
        await store2.initialize()

        results = await store2.search_similar(
            collection_name="conversations",
            query_text=test_text,
            n_results=1,
        )

        assert len(results) > 0
        assert test_text in results[0]["document"]

        await store2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
