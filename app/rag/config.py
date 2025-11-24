"""
RAG Configuration

SPECIFICATION:
- Centralized RAG system configuration
- Embedding models configuration
- Retrieval parameters (top-k, thresholds)
- Context window management
- Model settings for generation

ARCHITECTURE:
- Pydantic-based configuration
- Environment variable support
- Sensible defaults with overrides
- Type-safe settings

WHY:
- Single source of truth for RAG parameters
- Easy to tune performance vs quality
- Production-ready configuration management
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class RAGConfig(BaseSettings):
    """
    RAG System Configuration

    PATTERN: Centralized configuration with environment overrides
    WHY: Easy tuning and deployment flexibility
    """

    # ========== EMBEDDING CONFIGURATION ==========

    # Embedding model for vector generation
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="RAG_EMBEDDING_MODEL",
        description="OpenAI embedding model (text-embedding-3-small or text-embedding-3-large)"
    )

    # Embedding dimensions (1536 for small, 3072 for large)
    embedding_dimension: int = Field(
        default=1536,
        env="RAG_EMBEDDING_DIMENSION",
        description="Embedding vector dimension"
    )

    # Batch size for embedding generation
    embedding_batch_size: int = Field(
        default=100,
        env="RAG_EMBEDDING_BATCH_SIZE",
        description="Number of texts to embed in one batch"
    )

    # ========== RETRIEVAL CONFIGURATION ==========

    # Number of documents to retrieve
    retrieval_top_k: int = Field(
        default=5,
        env="RAG_RETRIEVAL_TOP_K",
        description="Number of most relevant documents to retrieve"
    )

    # Minimum relevance score (0.0 to 1.0)
    relevance_threshold: float = Field(
        default=0.7,
        env="RAG_RELEVANCE_THRESHOLD",
        description="Minimum similarity score for retrieval (0.0-1.0)"
    )

    # Use hybrid search (combines vector + keyword)
    use_hybrid_search: bool = Field(
        default=True,
        env="RAG_USE_HYBRID_SEARCH",
        description="Enable hybrid search (vector + keyword)"
    )

    # Hybrid search weighting (0.0 = only keyword, 1.0 = only vector)
    hybrid_alpha: float = Field(
        default=0.7,
        env="RAG_HYBRID_ALPHA",
        description="Vector search weight in hybrid mode (0.0-1.0)"
    )

    # Search within session only (vs across all sessions)
    session_scoped_search: bool = Field(
        default=False,
        env="RAG_SESSION_SCOPED_SEARCH",
        description="Restrict search to current session only"
    )

    # Maximum age of documents to retrieve (in days, None = no limit)
    max_document_age_days: Optional[int] = Field(
        default=None,
        env="RAG_MAX_DOCUMENT_AGE_DAYS",
        description="Maximum age of documents to retrieve (days)"
    )

    # ========== CONTEXT BUILDING CONFIGURATION ==========

    # Maximum tokens for retrieval context
    max_context_tokens: int = Field(
        default=4000,
        env="RAG_MAX_CONTEXT_TOKENS",
        description="Maximum tokens for retrieved context"
    )

    # Context summarization if too large
    enable_context_summarization: bool = Field(
        default=True,
        env="RAG_ENABLE_CONTEXT_SUMMARIZATION",
        description="Summarize context if exceeds max tokens"
    )

    # Prioritize recent documents
    prioritize_recent: bool = Field(
        default=True,
        env="RAG_PRIORITIZE_RECENT",
        description="Give higher weight to recent documents"
    )

    # Recency decay factor (exponential decay)
    recency_decay_days: int = Field(
        default=7,
        env="RAG_RECENCY_DECAY_DAYS",
        description="Days for recency weight to decay by ~63%"
    )

    # Deduplicate similar retrieved documents
    deduplicate_context: bool = Field(
        default=True,
        env="RAG_DEDUPLICATE_CONTEXT",
        description="Remove duplicate/similar retrieved documents"
    )

    # Similarity threshold for deduplication (cosine similarity)
    deduplication_threshold: float = Field(
        default=0.95,
        env="RAG_DEDUPLICATION_THRESHOLD",
        description="Cosine similarity threshold for deduplication"
    )

    # ========== GENERATION CONFIGURATION ==========

    # Claude model for RAG generation
    generation_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        env="RAG_GENERATION_MODEL",
        description="Claude model for RAG-enhanced generation"
    )

    # Maximum tokens for generation
    generation_max_tokens: int = Field(
        default=1500,
        env="RAG_GENERATION_MAX_TOKENS",
        description="Maximum tokens for Claude response"
    )

    # Temperature for generation
    generation_temperature: float = Field(
        default=0.7,
        env="RAG_GENERATION_TEMPERATURE",
        description="Temperature for Claude generation (0.0-1.0)"
    )

    # Enable citation generation
    enable_citations: bool = Field(
        default=True,
        env="RAG_ENABLE_CITATIONS",
        description="Generate citations linking to source documents"
    )

    # ========== PERFORMANCE CONFIGURATION ==========

    # Cache embeddings (in-memory cache)
    cache_embeddings: bool = Field(
        default=True,
        env="RAG_CACHE_EMBEDDINGS",
        description="Cache embeddings to reduce API calls"
    )

    # Embedding cache TTL (seconds)
    embedding_cache_ttl: int = Field(
        default=3600,
        env="RAG_EMBEDDING_CACHE_TTL",
        description="Embedding cache TTL in seconds"
    )

    # Retrieval timeout (seconds)
    retrieval_timeout: float = Field(
        default=5.0,
        env="RAG_RETRIEVAL_TIMEOUT",
        description="Maximum time for retrieval operation (seconds)"
    )

    # Context building timeout (seconds)
    context_timeout: float = Field(
        default=2.0,
        env="RAG_CONTEXT_TIMEOUT",
        description="Maximum time for context building (seconds)"
    )

    # ========== STORAGE CONFIGURATION ==========

    # ChromaDB persist directory
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        env="RAG_CHROMA_PERSIST_DIRECTORY",
        description="ChromaDB persistence directory"
    )

    # ChromaDB collection name
    chroma_collection_name: str = Field(
        default="conversation_memory",
        env="RAG_CHROMA_COLLECTION_NAME",
        description="ChromaDB collection name"
    )

    # Enable ChromaDB persistence
    chroma_enable_persistence: bool = Field(
        default=True,
        env="RAG_CHROMA_ENABLE_PERSISTENCE",
        description="Enable ChromaDB disk persistence"
    )

    # ========== FALLBACK CONFIGURATION ==========

    # Graceful degradation if retrieval fails
    enable_fallback: bool = Field(
        default=True,
        env="RAG_ENABLE_FALLBACK",
        description="Continue without RAG if retrieval fails"
    )

    # Use basic conversation mode as fallback
    fallback_to_basic_mode: bool = Field(
        default=True,
        env="RAG_FALLBACK_TO_BASIC_MODE",
        description="Fall back to non-RAG conversation on errors"
    )

    # ========== DEBUGGING CONFIGURATION ==========

    # Log retrieval details
    log_retrieval_details: bool = Field(
        default=True,
        env="RAG_LOG_RETRIEVAL_DETAILS",
        description="Log detailed retrieval information"
    )

    # Include retrieval metadata in response
    include_retrieval_metadata: bool = Field(
        default=False,
        env="RAG_INCLUDE_RETRIEVAL_METADATA",
        description="Include retrieval metadata in responses"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Singleton instance
rag_config = RAGConfig()


# ========== HELPER FUNCTIONS ==========

def get_rag_config() -> RAGConfig:
    """
    Get RAG configuration instance

    Returns:
        RAGConfig singleton instance
    """
    return rag_config


def update_rag_config(**kwargs) -> None:
    """
    Update RAG configuration dynamically

    Args:
        **kwargs: Configuration parameters to update

    Example:
        update_rag_config(retrieval_top_k=10, relevance_threshold=0.8)
    """
    global rag_config
    for key, value in kwargs.items():
        if hasattr(rag_config, key):
            setattr(rag_config, key, value)
        else:
            raise ValueError(f"Invalid RAG configuration parameter: {key}")


def get_performance_profile(profile: str = "balanced") -> dict:
    """
    Get predefined performance profiles

    PATTERN: Performance vs quality trade-off presets
    WHY: Easy tuning for different use cases

    Args:
        profile: One of "fast", "balanced", "quality"

    Returns:
        Dictionary of configuration overrides
    """
    profiles = {
        "fast": {
            "retrieval_top_k": 3,
            "relevance_threshold": 0.65,
            "max_context_tokens": 2000,
            "enable_context_summarization": False,
            "deduplicate_context": False,
            "generation_max_tokens": 1000,
        },
        "balanced": {
            "retrieval_top_k": 5,
            "relevance_threshold": 0.7,
            "max_context_tokens": 4000,
            "enable_context_summarization": True,
            "deduplicate_context": True,
            "generation_max_tokens": 1500,
        },
        "quality": {
            "retrieval_top_k": 10,
            "relevance_threshold": 0.75,
            "max_context_tokens": 6000,
            "enable_context_summarization": True,
            "deduplicate_context": True,
            "generation_max_tokens": 2000,
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Choose from: {list(profiles.keys())}")

    return profiles[profile]
