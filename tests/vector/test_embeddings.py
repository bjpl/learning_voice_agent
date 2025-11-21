"""
Tests for EmbeddingGenerator
Target: 20+ tests, 93% coverage
"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from app.vector.embeddings import EmbeddingGenerator, EmbeddingCache
from datetime import datetime, timedelta


class TestEmbeddingCache:
    """Test embedding cache functionality (4 tests)"""

    def test_cache_set_and_get(self):
        """Test basic cache set and get"""
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        embedding = np.array([1.0] * 384, dtype=np.float32)

        cache.set("test_text", embedding)
        result = cache.get("test_text")

        assert result is not None
        np.testing.assert_array_equal(result, embedding)

    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = EmbeddingCache(max_size=10)

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = EmbeddingCache(max_size=2, ttl_seconds=3600)

        emb1 = np.array([1.0] * 384, dtype=np.float32)
        emb2 = np.array([2.0] * 384, dtype=np.float32)
        emb3 = np.array([3.0] * 384, dtype=np.float32)

        cache.set("text1", emb1)
        cache.set("text2", emb2)
        cache.set("text3", emb3)  # Should evict text1

        assert cache.get("text1") is None  # Evicted
        assert cache.get("text2") is not None
        assert cache.get("text3") is not None

    def test_cache_ttl_expiration(self):
        """Test TTL expiration"""
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)

        embedding = np.array([1.0] * 384, dtype=np.float32)
        cache.set("test", embedding)

        # Should exist immediately
        assert cache.get("test") is not None

        # Manually expire by modifying timestamp
        key = "test"
        if key in cache._cache:
            emb, timestamp = cache._cache[key]
            # Set timestamp to old value
            cache._cache[key] = (emb, datetime.now() - timedelta(seconds=3601))

        # Should be expired now
        assert cache.get("test") is None

    def test_cache_clear(self):
        """Test cache clear"""
        cache = EmbeddingCache(max_size=10)

        cache.set("text1", np.array([1.0] * 384, dtype=np.float32))
        cache.set("text2", np.array([2.0] * 384, dtype=np.float32))

        assert cache.stats()['size'] == 2

        cache.clear()

        assert cache.stats()['size'] == 0

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)

        stats = cache.stats()

        assert 'size' in stats
        assert 'max_size' in stats
        assert 'ttl_seconds' in stats
        assert stats['max_size'] == 100
        assert stats['ttl_seconds'] == 3600


@pytest.mark.asyncio
class TestEmbeddingGeneratorInitialization:
    """Test embedding generator initialization (3 tests)"""

    async def test_initialize_success(self, monkeypatch):
        """Test successful initialization"""
        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.max_seq_length = 256

        def mock_sentence_transformer(model_name, device):
            return mock_model

        monkeypatch.setattr('app.vector.embeddings.SentenceTransformer', mock_sentence_transformer)

        generator = EmbeddingGenerator()
        await generator.initialize()

        assert generator._initialized is True
        assert generator.model is not None

    async def test_initialize_idempotent(self):
        """Test that initialize is idempotent"""
        generator = EmbeddingGenerator()
        generator._initialized = True

        # Should not reinitialize
        await generator.initialize()

        assert generator._initialized is True

    async def test_initialize_creates_cache(self, monkeypatch):
        """Test that initialization creates cache when enabled"""
        mock_model = MagicMock()
        mock_model.max_seq_length = 256

        monkeypatch.setattr('app.vector.embeddings.SentenceTransformer', lambda *args, **kwargs: mock_model)

        generator = EmbeddingGenerator()
        generator.model_config.enable_cache = True

        await generator.initialize()

        assert generator.cache is not None
        assert isinstance(generator.cache, EmbeddingCache)


@pytest.mark.asyncio
class TestGenerateEmbedding:
    """Test single embedding generation (5 tests)"""

    async def test_generate_embedding_basic(self, monkeypatch):
        """Test basic embedding generation"""
        mock_model = MagicMock()
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode = MagicMock(return_value=mock_embedding)

        generator = EmbeddingGenerator()
        generator.model = mock_model
        generator._initialized = True

        result = await generator.generate_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        np.testing.assert_array_equal(result, mock_embedding)

    async def test_generate_embedding_with_cache_hit(self, monkeypatch):
        """Test embedding generation with cache hit"""
        generator = EmbeddingGenerator()
        generator._initialized = True

        # Set up cache with entry
        cached_embedding = np.random.rand(384).astype(np.float32)
        generator.cache = EmbeddingCache()
        generator.cache.set("test text", cached_embedding)

        # Mock model to verify it's NOT called on cache hit
        generator.model = MagicMock()
        generator.model.encode = MagicMock()

        result = await generator.generate_embedding("test text", use_cache=True)

        # Should return cached embedding
        np.testing.assert_array_equal(result, cached_embedding)

        # Model should NOT have been called
        generator.model.encode.assert_not_called()

    async def test_generate_embedding_cache_disabled(self, monkeypatch):
        """Test embedding generation with cache disabled"""
        mock_model = MagicMock()
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode = MagicMock(return_value=mock_embedding)

        generator = EmbeddingGenerator()
        generator.model = mock_model
        generator._initialized = True
        generator.cache = EmbeddingCache()

        result = await generator.generate_embedding("test", use_cache=False)

        # Should not use cache
        assert generator.cache.get("test") is None

    async def test_generate_embedding_auto_initializes(self):
        """Test that generate_embedding initializes if needed"""
        generator = EmbeddingGenerator()
        generator._initialized = False

        # Mock initialize
        init_called = False
        async def mock_init():
            nonlocal init_called
            init_called = True
            generator._initialized = True
            generator.model = MagicMock()
            generator.model.encode = MagicMock(return_value=np.random.rand(384).astype(np.float32))

        generator.initialize = mock_init

        await generator.generate_embedding("test")

        assert init_called

    async def test_generate_embedding_error_handling(self):
        """Test error handling in embedding generation"""
        generator = EmbeddingGenerator()
        generator._initialized = True
        generator.model = MagicMock()
        generator.model.encode = MagicMock(side_effect=Exception("Model error"))

        with pytest.raises(Exception, match="Model error"):
            await generator.generate_embedding("test")


@pytest.mark.asyncio
class TestGenerateBatch:
    """Test batch embedding generation (5 tests)"""

    async def test_generate_batch_basic(self, monkeypatch):
        """Test basic batch generation"""
        texts = ["text1", "text2", "text3"]

        mock_model = MagicMock()
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode = MagicMock(return_value=mock_embeddings)

        generator = EmbeddingGenerator()
        generator.model = mock_model
        generator._initialized = True

        results = await generator.generate_batch(texts)

        assert len(results) == 3
        assert all(isinstance(emb, np.ndarray) for emb in results)
        assert all(emb.shape == (384,) for emb in results)

    async def test_generate_batch_empty_input(self):
        """Test batch generation with empty input"""
        generator = EmbeddingGenerator()
        generator._initialized = True

        results = await generator.generate_batch([])

        assert results == []

    async def test_generate_batch_with_cache(self, monkeypatch):
        """Test batch generation with partial cache hits"""
        texts = ["text1", "text2", "text3"]

        generator = EmbeddingGenerator()
        generator._initialized = True

        # Cache text1 and text2
        generator.cache = EmbeddingCache()
        cached_emb1 = np.array([1.0] * 384, dtype=np.float32)
        cached_emb2 = np.array([2.0] * 384, dtype=np.float32)
        generator.cache.set("text1", cached_emb1)
        generator.cache.set("text2", cached_emb2)

        # Mock model for uncached text
        mock_model = MagicMock()
        new_emb3 = np.array([[3.0] * 384], dtype=np.float32)
        mock_model.encode = MagicMock(return_value=new_emb3)
        generator.model = mock_model

        results = await generator.generate_batch(texts)

        # Should have 3 results
        assert len(results) == 3

        # First two should be from cache
        np.testing.assert_array_equal(results[0], cached_emb1)
        np.testing.assert_array_equal(results[1], cached_emb2)

        # Model should only be called for text3
        mock_model.encode.assert_called_once()

    async def test_generate_batch_custom_batch_size(self):
        """Test batch generation with custom batch size"""
        texts = ["text"] * 100

        generator = EmbeddingGenerator()
        generator._initialized = True

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=np.random.rand(100, 384).astype(np.float32)
        )
        generator.model = mock_model

        await generator.generate_batch(texts, batch_size=32)

        # Verify batch_size was passed
        call_args = mock_model.encode.call_args[1]
        assert call_args['batch_size'] == 32

    async def test_generate_batch_with_progress(self):
        """Test batch generation with progress bar"""
        texts = ["text"] * 10

        generator = EmbeddingGenerator()
        generator._initialized = True

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=np.random.rand(10, 384).astype(np.float32)
        )
        generator.model = mock_model

        await generator.generate_batch(texts, show_progress=True)

        # Verify show_progress_bar was passed
        call_args = mock_model.encode.call_args[1]
        assert call_args['show_progress_bar'] is True


@pytest.mark.asyncio
class TestEmbeddingGeneratorUtilities:
    """Test utility methods (4 tests)"""

    def test_get_model_info_initialized(self):
        """Test get_model_info when initialized"""
        generator = EmbeddingGenerator()
        generator._initialized = True
        generator.cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)

        info = generator.get_model_info()

        assert info['status'] == 'initialized'
        assert 'model_name' in info
        assert 'dimensions' in info
        assert 'cache' in info
        assert info['cache']['max_size'] == 1000

    def test_get_model_info_not_initialized(self):
        """Test get_model_info when not initialized"""
        generator = EmbeddingGenerator()
        generator._initialized = False

        info = generator.get_model_info()

        assert info['status'] == 'not_initialized'

    def test_clear_cache(self):
        """Test cache clearing"""
        generator = EmbeddingGenerator()
        generator._initialized = True
        generator.cache = EmbeddingCache()

        # Add some entries
        generator.cache.set("test1", np.random.rand(384).astype(np.float32))
        generator.cache.set("test2", np.random.rand(384).astype(np.float32))

        assert generator.cache.stats()['size'] == 2

        generator.clear_cache()

        assert generator.cache.stats()['size'] == 0

    async def test_close_cleanup(self):
        """Test that close cleans up resources"""
        generator = EmbeddingGenerator()
        generator._initialized = True
        generator.cache = EmbeddingCache()
        generator.model = MagicMock()

        # Add cache entry
        generator.cache.set("test", np.random.rand(384).astype(np.float32))

        await generator.close()

        assert generator._initialized is False
        assert generator.model is None
        assert generator.cache.stats()['size'] == 0
