"""
Search Configuration
PATTERN: Centralized configuration for hybrid search system
WHY: Single source of truth for search parameters
"""
from typing import Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SearchStrategy(str, Enum):
    """Search strategy enumeration"""
    SEMANTIC = "semantic"  # Pure vector search for conceptual queries
    KEYWORD = "keyword"   # Pure FTS5 for exact phrase matching
    HYBRID = "hybrid"     # Combined search with RRF (default)
    ADAPTIVE = "adaptive"  # Automatically choose strategy based on query


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search engine"""

    # Vector search configuration
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid mode (0-1)"
    )

    # Keyword search configuration
    keyword_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid mode (0-1)"
    )

    # Result limits
    max_results_per_search: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results to fetch from each search type"
    )

    final_result_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Final number of results to return"
    )

    # RRF configuration
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant for rank fusion (typically 60)"
    )

    # Vector search configuration
    vector_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for vector results"
    )

    # Adaptive strategy configuration
    adaptive_keyword_threshold: int = Field(
        default=3,
        ge=1,
        description="Max words for keyword-only in adaptive mode"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="OpenAI embedding model name"
    )

    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions"
    )

    # Cache configuration
    enable_embedding_cache: bool = Field(
        default=True,
        description="Enable caching of embeddings"
    )

    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )

    # Query analysis configuration
    enable_spell_correction: bool = Field(
        default=True,
        description="Enable fuzzy spell correction"
    )

    enable_query_expansion: bool = Field(
        default=False,
        description="Enable query expansion with synonyms"
    )

    max_query_length: int = Field(
        default=500,
        ge=1,
        description="Maximum query length in characters"
    )

    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0"""
        total = self.vector_weight + self.keyword_weight
        return abs(total - 1.0) < 0.01

    class Config:
        validate_assignment = True


# Default configuration
DEFAULT_SEARCH_CONFIG = HybridSearchConfig()


# Intent detection patterns
INTENT_PATTERNS: Dict[str, list] = {
    "conceptual": [
        "what is", "explain", "how does", "why", "concept",
        "understand", "learn about", "tell me about", "difference between"
    ],
    "factual": [
        "when", "where", "who", "which", "specific",
        "exact", "date", "time", "name", "number"
    ],
    "comparison": [
        "vs", "versus", "compare", "better", "difference",
        "similar", "like", "alternative"
    ],
    "procedural": [
        "how to", "steps", "guide", "tutorial", "process",
        "implement", "create", "build"
    ]
}


# Stop words for keyword extraction (common English words to filter)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "he", "in", "is", "it", "its", "of", "on",
    "that", "the", "to", "was", "will", "with", "the", "this",
    "but", "they", "have", "had", "what", "when", "where", "who",
    "which", "why", "how"
}
