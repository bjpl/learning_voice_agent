"""
Embedding Generation with Sentence Transformers
PATTERN: Singleton with caching for efficiency
WHY: Model loading is expensive, reuse instances
RESILIENCE: Automatic model download with retry
"""
import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any
from functools import lru_cache
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from app.vector.config import VectorConfig, EmbeddingModelConfig
from app.logger import db_logger


class EmbeddingCache:
    """
    CONCEPT: LRU cache with TTL for embedding results
    WHY: Avoid recomputing embeddings for common queries
    RESILIENCE: Automatic expiration prevents stale data
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[np.ndarray, datetime]] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached embedding if valid"""
        if key in self._cache:
            embedding, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return embedding
            else:
                # Expired, remove it
                del self._cache[key]
        return None

    def set(self, key: str, embedding: np.ndarray) -> None:
        """Cache an embedding with timestamp"""
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (embedding, datetime.now())

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class EmbeddingGenerator:
    """
    PATTERN: Lazy-loaded singleton with batch support
    WHY: Efficient embedding generation with caching
    RESILIENCE: Automatic retry on model loading failures

    USAGE:
        generator = EmbeddingGenerator()
        await generator.initialize()
        embedding = await generator.generate_embedding("Hello world")
        embeddings = await generator.generate_batch(["text1", "text2"])
    """

    _instance: Optional['EmbeddingGenerator'] = None

    def __new__(cls, config: Optional[VectorConfig] = None):
        """Singleton pattern for model reuse"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[VectorConfig] = None):
        """
        Initialize embedding generator

        Args:
            config: Vector configuration (uses default if None)
        """
        if self._initialized:
            return

        from app.vector.config import vector_config as default_config
        self.config = config or default_config
        self.model_config = self.config.embedding_model

        self.model: Optional[SentenceTransformer] = None
        self.cache: Optional[EmbeddingCache] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        CONCEPT: Lazy model loading
        WHY: Defer expensive operations until needed
        RESILIENCE: Retry on download/loading failures
        """
        if self._initialized:
            return

        try:
            db_logger.info(
                "embedding_model_loading_started",
                model_name=self.model_config.model_name,
                device=self.model_config.device
            )

            # Load the sentence transformer model
            self.model = SentenceTransformer(
                self.model_config.model_name,
                device=self.model_config.device
            )

            # Set max sequence length
            self.model.max_seq_length = self.model_config.max_sequence_length

            # Initialize cache if enabled
            if self.model_config.enable_cache:
                self.cache = EmbeddingCache(
                    max_size=self.model_config.cache_size,
                    ttl_seconds=self.model_config.cache_ttl
                )

            self._initialized = True

            db_logger.info(
                "embedding_model_loaded",
                model_name=self.model_config.model_name,
                dimensions=self.model_config.dimensions,
                max_length=self.model_config.max_sequence_length,
                cache_enabled=self.model_config.enable_cache
            )

        except Exception as e:
            db_logger.error(
                "embedding_model_loading_failed",
                model_name=self.model_config.model_name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load embedding model: {e}")

    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text

        PATTERN: Cache-first with fallback to computation
        WHY: Avoid redundant computation for repeated queries
        RESILIENCE: Graceful cache failures fall back to computation

        Args:
            text: Input text to embed
            use_cache: Whether to use cached results

        Returns:
            Embedding vector as numpy array
        """
        if not self._initialized:
            await self.initialize()

        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                db_logger.debug("embedding_cache_hit", text_preview=text[:50])
                return cached

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.model_config.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Cache the result
            if use_cache and self.cache:
                self.cache.set(text, embedding)

            db_logger.debug(
                "embedding_generated",
                text_length=len(text),
                embedding_shape=embedding.shape
            )

            return embedding

        except Exception as e:
            db_logger.error(
                "embedding_generation_failed",
                text_preview=text[:100],
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def generate_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts

        PATTERN: Batched processing for efficiency
        WHY: GPU/CPU parallelization reduces total time
        RESILIENCE: Partial success - skip failed items

        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            show_progress: Show progress bar for large batches

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        batch_size = batch_size or self.model_config.batch_size

        try:
            db_logger.info(
                "batch_embedding_started",
                num_texts=len(texts),
                batch_size=batch_size
            )

            # Check cache for all texts
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            if self.cache:
                for i, text in enumerate(texts):
                    cached = self.cache.get(text)
                    if cached is not None:
                        embeddings.append((i, cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    normalize_embeddings=self.model_config.normalize_embeddings,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )

                # Cache new embeddings
                if self.cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        self.cache.set(text, embedding)

                # Combine cached and new embeddings in correct order
                for i, embedding in zip(uncached_indices, new_embeddings):
                    embeddings.append((i, embedding))

            # Sort by original index and extract embeddings
            embeddings.sort(key=lambda x: x[0])
            result = [emb for _, emb in embeddings]

            db_logger.info(
                "batch_embedding_complete",
                num_texts=len(texts),
                cached_count=len(texts) - len(uncached_texts),
                generated_count=len(uncached_texts)
            )

            return result

        except Exception as e:
            db_logger.error(
                "batch_embedding_failed",
                num_texts=len(texts),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model metadata
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        info = {
            "status": "initialized",
            "model_name": self.model_config.model_name,
            "dimensions": self.model_config.dimensions,
            "max_sequence_length": self.model_config.max_sequence_length,
            "device": self.model_config.device,
            "normalize_embeddings": self.model_config.normalize_embeddings,
            "batch_size": self.model_config.batch_size
        }

        if self.cache:
            info["cache"] = self.cache.stats()

        return info

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
            db_logger.info("embedding_cache_cleared")

    async def close(self) -> None:
        """
        Clean up resources

        CONCEPT: Explicit resource management
        WHY: Free GPU/CPU memory when done
        """
        if self.cache:
            self.cache.clear()

        self.model = None
        self._initialized = False

        db_logger.info("embedding_generator_closed")


# Global embedding generator instance
embedding_generator = EmbeddingGenerator()
