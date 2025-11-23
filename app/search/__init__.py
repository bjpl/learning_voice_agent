"""
Hybrid Search Module
Combines vector similarity search with SQLite FTS5 keyword search
"""
from app.search.config import (
    HybridSearchConfig,
    SearchStrategy,
    DEFAULT_SEARCH_CONFIG
)
from app.search.vector_store import VectorStore, vector_store
from app.search.query_analyzer import QueryAnalyzer, query_analyzer, QueryAnalysis
from app.search.hybrid_search import (
    HybridSearchEngine,
    SearchResult,
    HybridSearchResponse,
    create_hybrid_search_engine
)

__all__ = [
    # Config
    'HybridSearchConfig',
    'SearchStrategy',
    'DEFAULT_SEARCH_CONFIG',
    # Vector Store
    'VectorStore',
    'vector_store',
    # Query Analysis
    'QueryAnalyzer',
    'query_analyzer',
    'QueryAnalysis',
    # Hybrid Search
    'HybridSearchEngine',
    'SearchResult',
    'HybridSearchResponse',
    'create_hybrid_search_engine',
]
