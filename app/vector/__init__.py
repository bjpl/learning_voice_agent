"""
Vector Database Layer for Learning Voice Agent
PATTERN: Semantic search with ChromaDB and sentence-transformers
WHY: Enable similarity-based retrieval beyond keyword matching
RESILIENCE: Persistent storage with automatic recovery
"""
from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .config import VectorConfig

__all__ = ['VectorStore', 'EmbeddingGenerator', 'VectorConfig']
