"""
Vector Database Configuration
PATTERN: Centralized configuration with sensible defaults
WHY: Easy tuning without code changes
RESILIENCE: Validation and fallbacks for all settings
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingModelConfig:
    """
    Configuration for sentence-transformer embedding model

    CONCEPT: all-MiniLM-L6-v2 balances speed and quality
    WHY: 384 dimensions, 80MB model, fast inference
    ALTERNATIVES: all-mpnet-base-v2 (higher quality, slower)
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    max_sequence_length: int = 256
    device: str = "cpu"  # Use "cuda" if GPU available
    batch_size: int = 32
    normalize_embeddings: bool = True

    # Cache settings
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour in seconds


@dataclass
class CollectionConfig:
    """
    Configuration for a ChromaDB collection

    PATTERN: Schema definition with metadata
    WHY: Consistent structure across collections
    """
    name: str
    metadata_schema: Dict[str, str] = field(default_factory=dict)
    distance_metric: str = "cosine"  # cosine, l2, or ip (inner product)

    def __post_init__(self):
        """Validate configuration"""
        valid_metrics = {"cosine", "l2", "ip"}
        if self.distance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid distance_metric: {self.distance_metric}. "
                f"Must be one of {valid_metrics}"
            )


@dataclass
class VectorConfig:
    """
    Main vector database configuration

    CONCEPT: Single source of truth for vector settings
    WHY: Consistent configuration across application
    RESILIENCE: Path validation and creation
    """
    persist_directory: Path = field(default_factory=lambda: Path("./data/chromadb"))
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)

    # Collection definitions
    collections: Dict[str, CollectionConfig] = field(default_factory=lambda: {
        "conversations": CollectionConfig(
            name="conversations",
            metadata_schema={
                "session_id": "str",
                "timestamp": "str",
                "exchange_type": "str",  # user, agent, system
                "speaker": "str",
                "exchange_id": "int"
            },
            distance_metric="cosine"
        ),
        "knowledge": CollectionConfig(
            name="knowledge",
            metadata_schema={
                "source": "str",
                "category": "str",
                "timestamp": "str",
                "confidence": "float"
            },
            distance_metric="cosine"
        )
    })

    # Search settings
    default_n_results: int = 10
    max_n_results: int = 100
    similarity_threshold: float = 0.7  # Minimum similarity score (0-1)

    # Performance settings
    enable_batch_operations: bool = True
    max_batch_size: int = 100

    def __post_init__(self):
        """
        CONCEPT: Automatic directory creation
        WHY: Prevent runtime errors from missing directories
        RESILIENCE: Graceful handling of permission errors
        """
        # Convert string to Path if needed
        if isinstance(self.persist_directory, str):
            self.persist_directory = Path(self.persist_directory)

        # Create directory if it doesn't exist
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create persist directory {self.persist_directory}: {e}"
            )

    def get_collection_config(self, name: str) -> Optional[CollectionConfig]:
        """Get configuration for a specific collection"""
        return self.collections.get(name)

    def add_collection(self, config: CollectionConfig) -> None:
        """
        Add a new collection configuration

        PATTERN: Dynamic collection management
        WHY: Allow runtime collection creation
        """
        self.collections[config.name] = config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "persist_directory": str(self.persist_directory),
            "embedding_model": {
                "model_name": self.embedding_model.model_name,
                "dimensions": self.embedding_model.dimensions,
                "device": self.embedding_model.device,
                "batch_size": self.embedding_model.batch_size
            },
            "collections": {
                name: {
                    "name": config.name,
                    "distance_metric": config.distance_metric,
                    "metadata_schema": config.metadata_schema
                }
                for name, config in self.collections.items()
            },
            "search": {
                "default_n_results": self.default_n_results,
                "max_n_results": self.max_n_results,
                "similarity_threshold": self.similarity_threshold
            }
        }


# Global configuration instance
vector_config = VectorConfig()
